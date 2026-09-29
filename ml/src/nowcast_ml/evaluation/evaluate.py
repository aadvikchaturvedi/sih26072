"""``nowcast-eval``: score the model and baselines with identical code.

    nowcast-eval --model artifacts/models/nowcast/latest --events data/events --split test \
                 --baselines all --out reports/run1

Rows: each baseline, ``model_full`` and ``model_satellite_only`` (radar forced
unavailable). Writes ``metrics.json``, ``skill.md`` and PNG plots to ``--out``.
Splits come from the model's training run (same event-hash assignment).
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from nowcast_ml.baselines.base import ForecastInput
from nowcast_ml.config import Config, load_config
from nowcast_ml.data import labels as lb
from nowcast_ml.data.event import Event
from nowcast_ml.data.zarr_events import WindowSpec, make_sample, valid_t0_indices
from nowcast_ml.evaluation import metrics as M
from nowcast_ml.evaluation.forecasters import build_forecasters, load_runner, parse_baselines
from nowcast_ml.utils.io import write_json
from nowcast_ml.utils.logging import get_logger

log = get_logger(__name__)
CASE_LEADS_MIN = (30, 60, 120)


class _State:
    def __init__(self, n_leads, n_thr, n_scales, ltg_leads, n_bins):
        self.cont = np.zeros((n_leads, n_thr, 4), np.int64)
        self.fss = np.zeros((n_leads, n_scales, 2))
        self.prob = {L: M.ProbAccumulator(n_bins) for L in ltg_leads}
        self.ff_parts: list[dict] = []
        self.warnings: dict[int, np.ndarray] = {}
        self.sharp: list[float] = []
        self.ms: list[float] = []


class Evaluator:
    def __init__(self, cfg: Config, first_flash_threshold: float = 0.5):
        self.cfg = cfg
        e = cfg.eval
        self.thr = list(e.thresholds_dbz)
        self.scales = list(e.fss_scales_px)
        self.leads = cfg.data.lead_minutes
        self.ltg_leads = list(cfg.data.lightning_leads_min)
        self.ff_thr = first_flash_threshold
        self.state: dict[str, _State] = {}
        self.n_samples = 0
        self.events: list[str] = []

    def _st(self, name) -> _State:
        if name not in self.state:
            e = self.cfg.eval
            self.state[name] = _State(
                len(self.leads), len(self.thr), len(self.scales), self.ltg_leads, e.reliability_bins
            )
        return self.state[name]

    def update(self, name: str, fc, sample: dict, grid_spacing_km: float, ms: float) -> None:
        st = self._st(name)
        st.ms.append(ms)
        y, v = sample["y_refl"], sample["y_refl_valid"]
        for li in range(len(self.leads)):
            for ti, thr in enumerate(self.thr):
                st.cont[li, ti] += M.contingency(fc.reflectivity[li], y[li], thr, v[li])
            for si, sc in enumerate(self.scales):
                st.fss[li, si] += M.fss_terms(
                    fc.reflectivity[li], y[li], self.cfg.eval.fss_threshold_dbz, sc, v[li]
                )
        l60 = self.leads.index(60) if 60 in self.leads else len(self.leads) // 2
        if np.isfinite(fc.reflectivity[l60]).all() and v[l60].all():
            st.sharp.append(
                M.high_freq_power_ratio(
                    fc.reflectivity[l60], np.nan_to_num(y[l60]), grid_spacing_km
                )
            )
        past = sample["past_ltg"].astype(float)
        for i, L in enumerate(self.ltg_leads):
            st.prob[L].update(
                fc.lightning[i],
                sample["y_ltg"][i],
                sample["y_ltg_valid"][i],
                refs={"persistence": past},
            )
        i30 = self.ltg_leads.index(30) if 30 in self.ltg_leads else 0
        st.warnings[int(sample["t0_index"])] = (
            np.nan_to_num(fc.lightning[i30]) >= self.ff_thr
        ) & ~sample["past_ltg"]

    def end_event(self, event_id: str, occ_dilated: np.ndarray, step_minutes: int) -> None:
        self.events.append(event_id)
        for st in self.state.values():
            if st.warnings:
                st.ff_parts.append(M.first_flash_scores(st.warnings, occ_dilated, step_minutes))
            st.warnings = {}

    def results(self) -> dict:
        out = {}
        for name, st in self.state.items():
            refl = {"csi": {}, "pod": {}, "far": {}, "fss": {}}
            for ti, thr in enumerate(self.thr):
                sc = [M.scores_from_table(st.cont[li, ti]) for li in range(len(self.leads))]
                for k in ("csi", "pod", "far"):
                    refl[k][str(int(thr))] = [s[k] for s in sc]
            for si, s in enumerate(self.scales):
                a, b = st.fss[:, si, 0], st.fss[:, si, 1]
                refl["fss"][str(s)] = [
                    float(1 - x / y) if y > 0 else float("nan") for x, y in zip(a, b, strict=True)
                ]
            out[name] = {
                "reflectivity": refl,
                "lightning": {str(L): st.prob[L].result() for L in self.ltg_leads},
                "first_flash": M.summarize_first_flash(st.ff_parts),
                "sharpness_hf_power_ratio_60min": float(np.nanmean(st.sharp))
                if st.sharp
                else float("nan"),
                "ms_per_forecast": float(np.mean(st.ms)) if st.ms else float("nan"),
            }
        return out


def evaluate(
    cfg: Config,
    forecasters: list,
    events: list[tuple[str, Event]],
    out_dir: str | Path | None = None,
    first_flash_threshold: float = 0.5,
    meta: dict | None = None,
) -> dict:
    spec = WindowSpec.from_config(cfg.data)
    ev_cfg = cfg.eval
    evaluator = Evaluator(cfg, first_flash_threshold)
    case = None
    k = 0
    for eid, ev in events:
        radius = lb.radius_px(spec.lightning_radius_km, ev.grid_spacing_km)
        t0s = valid_t0_indices(ev.n_times, spec, ev_cfg.sample_stride)
        for t0 in t0s:
            if ev_cfg.max_samples is not None and k >= ev_cfg.max_samples:
                break
            s = make_sample(ev, t0, spec)
            inp = ForecastInput.from_sample(
                s, cfg.data.channels, ev.grid_spacing_km, spec.step_minutes
            )
            fcs = {}
            for f in forecasters:
                t = time.perf_counter()
                fc = f.forecast(inp, spec.t_out, spec.leads_min)
                ms = 1000 * (time.perf_counter() - t)
                evaluator.update(f.name, fc, s, ev.grid_spacing_km, ms)
                fcs[f.name] = fc
            if k == ev_cfg.case_study_index:
                case = _case_study(eid, ev, t0, s, fcs, cfg)
            k += 1
        evaluator.end_event(eid, lb.dilate(ev.lightning_occurrence, radius), spec.step_minutes)
        evaluator.n_samples = k
    results = {
        "meta": {
            **(meta or {}),
            "n_samples": k,
            "n_events": len(evaluator.events),
            "events": evaluator.events,
            "leads_min": cfg.data.lead_minutes,
            "thresholds_dbz": ev_cfg.thresholds_dbz,
            "fss_threshold_dbz": ev_cfg.fss_threshold_dbz,
            "fss_scales_px": ev_cfg.fss_scales_px,
            "lightning_leads_min": cfg.data.lightning_leads_min,
            "first_flash_threshold": first_flash_threshold,
            "sample_stride": ev_cfg.sample_stride,
            "climatology_note": "BSS reference climatology = base rate of the evaluated samples (sample climatology)",
        },
        "forecasters": evaluator.results(),
    }
    if out_dir is not None:
        from nowcast_ml.evaluation.report import write_report

        out = Path(out_dir)
        out.mkdir(parents=True, exist_ok=True)
        write_json(out / "metrics.json", results)
        write_report(results, out, case)
    return results


def _case_study(eid, ev, t0, s, fcs, cfg) -> dict:
    leads = cfg.data.lead_minutes
    idx = [leads.index(L) for L in CASE_LEADS_MIN if L in leads]
    return {
        "event_id": eid,
        "t0": str(ev.times[t0]),
        "leads_min": [leads[i] for i in idx],
        "observed": [np.nan_to_num(s["y_refl"][i]) for i in idx],
        "forecasts": {n: [fc.reflectivity[i] for i in idx] for n, fc in fcs.items()},
        "lightning_60": {n: fc.lightning[-1] for n, fc in fcs.items()},
        "lightning_obs_60": s["y_ltg"][-1],
    }


def resolve_events(cfg: Config, events_dir: str | None, split: str, splits: dict | None):
    """(event_id, Event) pairs for ``split``, using the training split assignment if known."""
    from nowcast_ml.training.datamodule import EventSources

    if events_dir:
        cfg = cfg.model_copy(deep=True)
        cfg.data.events_dir = str(events_dir)
        if cfg.data.source == "sevir":
            cfg.data.source = "india"
    src = EventSources(cfg)
    if split == "all":
        ids = sorted(src.openers)
    elif (
        splits is not None
        and all(e in src.openers for e in splits.get(split, []))
        and splits.get(split)
    ):
        ids = splits[split]
    else:
        ids = src.splits[split]
    out = []
    for i in ids:
        op = src.openers[i]
        out.append((i, op if isinstance(op, Event) else op()))
    return out


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(
        prog="nowcast-eval",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--model", help="registry artifact dir (or .../latest) or a Lightning .ckpt")
    ap.add_argument("--config", help="train config (only needed without --model)")
    ap.add_argument(
        "--events", help="directory of event Zarr stores (default: from the model config)"
    )
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument(
        "--baselines",
        default="all",
        help="'all', 'none' or comma list of persistence,extrapolation,steps",
    )
    ap.add_argument(
        "--no-satellite-only", action="store_true", help="skip the satellite-only model row"
    )
    ap.add_argument("--out", default="reports/eval")
    ap.add_argument("--device", default="auto")
    ap.add_argument(
        "overrides",
        nargs="*",
        help="config overrides, e.g. eval.sample_stride=1 eval.max_samples=50",
    )
    args = ap.parse_args(argv)

    if not args.model and not args.config:
        ap.error("need --model or --config")
    runner, splits, ff_thr, model_meta = None, None, 0.5, {}
    if args.model:
        runner, cfg, splits = load_runner(args.model, args.device)
        ff_thr = cfg.inference.first_flash_threshold
        model_meta = {"model": str(args.model), "model_name": runner.name}
        if args.overrides:
            from omegaconf import OmegaConf

            merged = OmegaConf.merge(
                OmegaConf.create(cfg.model_dump(mode="json")),
                OmegaConf.from_dotlist(args.overrides),
            )
            cfg = Config.model_validate(OmegaConf.to_container(merged))
    else:
        cfg = load_config(args.config, args.overrides)
    events = resolve_events(cfg, args.events, args.split, splits)
    source = {str(ev.ds.attrs.get("source", cfg.data.source)) for _, ev in events}
    fs = build_forecasters(
        runner, parse_baselines(args.baselines), cfg.eval.steps_members, not args.no_satellite_only
    )
    log.info("evaluating %d forecasters on %d %s events", len(fs), len(events), args.split)
    res = evaluate(
        cfg,
        fs,
        events,
        args.out,
        ff_thr,
        meta={
            **model_meta,
            "split": args.split,
            "data_sources": sorted(source),
            "synthetic": source == {"synthetic"},
        },
    )
    print(f"wrote {args.out}/skill.md ({res['meta']['n_samples']} samples)")


if __name__ == "__main__":
    main()
