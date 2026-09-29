"""Model artifact registry.

Layout::

    <root>/<name>/<version>/
        model.pt            state dict (eager weights)
        refiner.pt          diffusion refiner weights (only if model.refiner == "diffusion")
        model.ts            TorchScript (nowcast-export)
        model.onnx          ONNX (nowcast-export, if export succeeds)
        config.yaml         full training config
        channels.json       ordered model channels + groups + input contract version
        norm_stats.json     normalization statistics (must match channels.json)
        calibrator.pkl      isotonic lightning calibrator (nowcast-calibrate)
        splits.json         event ids per split used in training
        metrics.json        training + evaluation metrics (nowcast-eval)
        model_card.md       data, splits, metrics, known limits
        manifest.json       SHA-256 of every file + library versions
    <root>/<name>/latest    text file naming the current version

Loading verifies the manifest hashes and that channels, normalization and
config agree, raising :class:`ModelLoadError` on any mismatch.
"""

from __future__ import annotations

import platform
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import torch

from nowcast_ml import __version__
from nowcast_ml.config import Config, config_from_yaml_text, dump_config
from nowcast_ml.data import channels as ch
from nowcast_ml.data.schema import CONTRACT_VERSION
from nowcast_ml.data.transforms import NormStats, NormStatsMismatchError
from nowcast_ml.inference.errors import ModelLoadError
from nowcast_ml.inference.schema import OUTPUT_CONTRACT_VERSION
from nowcast_ml.utils.io import atomic_write_text, read_json, sha256_file, sha256_text, write_json

REQUIRED_FILES = ("model.pt", "config.yaml", "channels.json", "norm_stats.json", "manifest.json")
LATEST = "latest"


@dataclass
class Artifact:
    path: Path
    name: str
    version: str
    config: Config
    channels: list[str]
    norm_stats: NormStats
    state_dict: dict
    calibrator: object | None
    refiner_state: dict | None
    splits: dict | None
    metrics: dict
    manifest: dict

    def file(self, name: str) -> Path:
        return self.path / name


# ------------------------------------------------------------------ paths


def resolve_version_dir(path: str | Path) -> Path:
    """Accept ``<name>/latest``, ``<name>/`` (uses latest) or ``<name>/<version>/``."""
    p = Path(path)
    if p.name == LATEST and p.is_file():
        v = p.read_text().strip()
        target = p.parent / v
    elif p.is_dir() and (p / LATEST).is_file() and not (p / "model.pt").exists():
        target = p / (p / LATEST).read_text().strip()
    else:
        target = p
    if not target.is_dir():
        raise ModelLoadError(f"model artifact directory not found: {target} (from {path})")
    return target


def list_versions(root: str | Path, name: str) -> list[str]:
    d = Path(root) / name
    return sorted(p.name for p in d.iterdir() if p.is_dir()) if d.is_dir() else []


def set_latest(version_dir: Path) -> None:
    atomic_write_text(version_dir.parent / LATEST, version_dir.name + "\n")


# ------------------------------------------------------------------ save


def _lib_versions() -> dict:
    import numpy
    import xarray

    return {
        "nowcast_ml": __version__,
        "torch": torch.__version__,
        "numpy": numpy.__version__,
        "xarray": xarray.__version__,
        "python": platform.python_version(),
    }


def refresh_manifest(version_dir: str | Path) -> dict:
    """Recompute hashes of all files (call after adding calibrator/exports/metrics)."""
    d = Path(version_dir)
    old = read_json(d / "manifest.json") if (d / "manifest.json").exists() else {}
    files = {
        p.name: sha256_file(p)
        for p in sorted(d.iterdir())
        if p.is_file() and p.name != "manifest.json" and not p.name.startswith(".")
    }
    manifest = {
        **old,
        "files": files,
        "updated": datetime.now(UTC).isoformat(),
        "libraries": {**old.get("libraries", {}), "last_writer": _lib_versions()},
    }
    write_json(d / "manifest.json", manifest)
    return manifest


def save_artifact(
    model: torch.nn.Module,
    cfg: Config,
    norm_stats: NormStats,
    root: str | Path = "artifacts/models",
    name: str | None = None,
    splits: dict | None = None,
    train_metrics: dict | None = None,
    calibrator=None,
    refiner: torch.nn.Module | None = None,
    make_latest: bool = True,
) -> Path:
    name = name or cfg.model.name
    norm_stats.check_channels(cfg.data.channels)
    cfg_text = dump_config(cfg)
    stamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S")
    version = f"v{stamp}-{sha256_text(cfg_text)[:8]}"
    d = Path(root) / name / version
    n = 1
    while d.exists():
        d = Path(root) / name / f"{version}-{n}"
        n += 1
    d.mkdir(parents=True)
    torch.save({k: v.detach().cpu() for k, v in model.state_dict().items()}, d / "model.pt")
    if refiner is not None:
        torch.save({k: v.detach().cpu() for k, v in refiner.state_dict().items()}, d / "refiner.pt")
    (d / "config.yaml").write_text(cfg_text)
    write_json(
        d / "channels.json",
        {
            "channels": list(cfg.data.channels),
            "groups": {c: ch.group_of(c) for c in cfg.data.channels},
            "input_contract_version": CONTRACT_VERSION,
            "grid_spacing_km": cfg.data.grid_spacing_km,
            "t_in": cfg.data.t_in,
            "t_out": cfg.data.t_out,
            "step_minutes": cfg.data.step_minutes,
        },
    )
    norm_stats.save(d / "norm_stats.json")
    if splits is not None:
        write_json(d / "splits.json", splits)
    write_json(d / "metrics.json", {"training": train_metrics or {}})
    if calibrator is not None:
        calibrator.save(d / "calibrator.pkl")
    write_json(
        d / "manifest.json",
        {
            "name": name,
            "version": version,
            "created": datetime.now(UTC).isoformat(),
            "config_sha256": sha256_text(cfg_text),
            "input_contract_version": CONTRACT_VERSION,
            "output_contract_version": OUTPUT_CONTRACT_VERSION,
            "libraries": {"creator": _lib_versions()},
        },
    )
    write_model_card(d)
    refresh_manifest(d)
    if make_latest:
        set_latest(d)
    return d


# ------------------------------------------------------------------ load


def load_artifact(path: str | Path, verify_hashes: bool = True) -> Artifact:
    d = resolve_version_dir(path)
    missing = [f for f in REQUIRED_FILES if not (d / f).exists()]
    if missing:
        raise ModelLoadError(f"artifact {d} is missing required file(s): {missing}")
    manifest = read_json(d / "manifest.json")
    if verify_hashes:
        for fname, digest in manifest.get("files", {}).items():
            f = d / fname
            if not f.exists():
                raise ModelLoadError(f"artifact {d}: file {fname} listed in manifest is missing")
            if sha256_file(f) != digest:
                raise ModelLoadError(
                    f"artifact {d}: {fname} does not match manifest hash (modified after saving?)"
                )
    try:
        cfg = config_from_yaml_text((d / "config.yaml").read_text())
    except Exception as e:  # noqa: BLE001
        raise ModelLoadError(f"artifact {d}: invalid config.yaml: {e}") from e
    chan_info = read_json(d / "channels.json")
    channels = list(chan_info["channels"])
    problems = ch.validate_channel_list(channels)
    if problems:
        raise ModelLoadError(f"artifact {d}: channels.json invalid: {problems}")
    if channels != list(cfg.data.channels):
        raise ModelLoadError(
            f"artifact {d}: channels.json {channels} does not match config channels {cfg.data.channels}"
        )
    for c, g in chan_info.get("groups", {}).items():
        if ch.group_of(c) != g:
            raise ModelLoadError(
                f"artifact {d}: channel {c!r} group {g!r} != registry {ch.group_of(c)!r}"
            )
    if (
        str(chan_info.get("input_contract_version", "")).split(".")[0]
        != CONTRACT_VERSION.split(".")[0]
    ):
        raise ModelLoadError(
            f"artifact {d}: input contract {chan_info.get('input_contract_version')} incompatible with {CONTRACT_VERSION}"
        )
    try:
        stats = NormStats.load(d / "norm_stats.json")
        stats.check_channels(channels)
    except NormStatsMismatchError as e:
        raise ModelLoadError(f"artifact {d}: normalization mismatch: {e}") from e
    try:
        state = torch.load(d / "model.pt", map_location="cpu", weights_only=True)
    except Exception as e:  # noqa: BLE001
        raise ModelLoadError(f"artifact {d}: cannot read model.pt: {e}") from e
    refiner_state = None
    if cfg.model.refiner == "diffusion":
        if not (d / "refiner.pt").exists():
            raise ModelLoadError(
                f"artifact {d}: config uses the diffusion refiner but refiner.pt is missing"
            )
        try:
            refiner_state = torch.load(d / "refiner.pt", map_location="cpu", weights_only=True)
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"artifact {d}: cannot read refiner.pt: {e}") from e
    calibrator = None
    if (d / "calibrator.pkl").exists():
        from nowcast_ml.calibration.isotonic import IsotonicCalibrator

        try:
            calibrator = IsotonicCalibrator.load(d / "calibrator.pkl")
        except Exception as e:  # noqa: BLE001
            raise ModelLoadError(f"artifact {d}: invalid calibrator.pkl: {e}") from e
        if list(calibrator.leads_min) != list(cfg.data.lightning_leads_min):
            raise ModelLoadError(f"artifact {d}: calibrator leads {calibrator.leads_min} != config")
    return Artifact(
        path=d,
        name=manifest.get("name", d.parent.name),
        version=manifest.get("version", d.name),
        config=cfg,
        channels=channels,
        norm_stats=stats,
        state_dict=state,
        calibrator=calibrator,
        refiner_state=refiner_state,
        splits=read_json(d / "splits.json") if (d / "splits.json").exists() else None,
        metrics=read_json(d / "metrics.json") if (d / "metrics.json").exists() else {},
        manifest=manifest,
    )


def update_metrics(version_dir: str | Path, key: str, metrics: dict) -> None:
    d = resolve_version_dir(version_dir)
    m = read_json(d / "metrics.json") if (d / "metrics.json").exists() else {}
    m[key] = metrics
    write_json(d / "metrics.json", m)
    write_model_card(d)
    refresh_manifest(d)


# ------------------------------------------------------------------ model card


def _fmt(x) -> str:
    return "–" if x is None or x != x else f"{x:.3f}"


def write_model_card(version_dir: str | Path) -> None:
    d = Path(version_dir)
    cfg = config_from_yaml_text((d / "config.yaml").read_text())
    man = read_json(d / "manifest.json") if (d / "manifest.json").exists() else {}
    splits = read_json(d / "splits.json") if (d / "splits.json").exists() else None
    metrics = read_json(d / "metrics.json") if (d / "metrics.json").exists() else {}
    cal = (
        read_json(d / "calibration_report.json")
        if (d / "calibration_report.json").exists()
        else None
    )
    exp = read_json(d / "export_report.json") if (d / "export_report.json").exists() else None
    synthetic = cfg.data.source == "synthetic"
    L = [
        f"# Model card: {man.get('name', d.parent.name)} / {man.get('version', d.name)}",
        "",
    ]
    if synthetic:
        L += [
            "> **Trained on SYNTHETIC data.** For pipeline testing only; not for operational use.",
            "",
        ]
    L += [
        "## Intended use",
        "",
        "0–120 min nowcasts of column-max reflectivity (10-min steps) and probability of lightning "
        "within 10 km in the next 30 / 60 min, for IMD thunderstorm nowcasting (SIH 2026, PS 26072). "
        "Decision support for forecasters; not a replacement for warnings issued by forecasters.",
        "",
        "## Model",
        "",
        f"- Backbone: `{cfg.model.backbone}` (hid_s={cfg.model.hid_s}, hid_t={cfg.model.hid_t}, "
        f"n_s={cfg.model.n_s}, n_t={cfg.model.n_t}); lightning head: {cfg.model.lightning_head.enabled}",
        f"- Ensemble refiner: `{cfg.model.refiner or 'none'}`"
        + (
            f" (DDIM {cfg.model.refiner_params.sample_steps} steps, base {cfg.model.refiner_params.base_channels} ch)"
            if cfg.model.refiner
            else ""
        ),
        f"- Inputs: {cfg.data.t_in} frames × {len(cfg.data.channels)} channels + availability masks; "
        f"outputs {cfg.data.t_out} frames; grid {cfg.data.grid_spacing_km} km",
        f"- Channels: {', '.join(cfg.data.channels)}",
        f"- Training stage: `{cfg.train.stage}`; init_from: `{cfg.train.init_from}`; "
        f"modality dropout p_radar={cfg.train.modality_dropout.p_radar}",
        "",
        "## Data",
        "",
        f"- Source: `{cfg.data.source}`"
        + (f" (events_dir `{cfg.data.events_dir}`)" if cfg.data.events_dir else ""),
    ]
    if splits:
        L.append("- Split by event: " + ", ".join(f"{k}={len(v)}" for k, v in splits.items()))
    L += ["", "## Metrics", ""]
    ev = {k: v for k, v in metrics.items() if k.startswith("eval")}
    if not ev:
        L.append("No evaluation run recorded yet (run `nowcast-eval --model <this artifact>`).")
    for key, res in ev.items():
        meta = res.get("meta", {})
        L.append(
            f"**{key}** — split `{meta.get('split')}`, {meta.get('n_events')} events, {meta.get('n_samples')} samples, "
            f"data {meta.get('data_sources')}{' (SYNTHETIC)' if meta.get('synthetic') else ''}"
        )
        L += [
            "",
            "| Forecaster | CSI≥35 +30 | CSI≥35 +60 | BSS(clim) +30 | BSS(clim) +60 | ROC-AUC +60 |",
            "|---|---|---|---|---|---|",
        ]
        leads = meta.get("leads_min", [])
        i30 = leads.index(30) if 30 in leads else None
        i60 = leads.index(60) if 60 in leads else None
        for n, r in res.get("forecasters", {}).items():
            csi = r["reflectivity"]["csi"].get("35", [])
            lt = r["lightning"]
            L.append(
                f"| {n} | {_fmt(csi[i30] if i30 is not None and csi else None)} | "
                f"{_fmt(csi[i60] if i60 is not None and csi else None)} | "
                f"{_fmt(lt.get('30', {}).get('bss_climatology'))} | {_fmt(lt.get('60', {}).get('bss_climatology'))} | "
                f"{_fmt(lt.get('60', {}).get('roc_auc'))} |"
            )
        L.append("")
    if cal:
        L += [
            "## Calibration",
            "",
            "Isotonic, per mode and lead (Brier on the fitting split, in-sample):",
            "",
        ]
        for mode, r in cal.items():
            for lead, v in r.items():
                L.append(
                    f"- {mode} +{lead} min: {v['brier_raw']:.4f} → {v['brier_calibrated_in_sample']:.4f} (n={v['n']})"
                )
        L.append("")
    if exp:
        L += ["## Export", ""]
        for k, v in exp.items():
            L.append(f"- {k}: {v}")
        L.append("")
    L += [
        "## Known limits",
        "",
        "- The deterministic forecast blurs with lead time (see the sharpness row in the skill report). "
        "Diffusion-refiner members are sharper but their spread is only as good as the training data; "
        "check CRPS and spread in the skill report.",
        "- SEVIR pretraining maps VIL to an approximate reflectivity (Greene–Clark inversion); intensity "
        "calibration relies on India fine-tuning.",
        "- Satellite-only mode cannot observe precipitation directly; expect lower skill than with radar.",
        "- Lightning probabilities are calibrated on the validation split; recalibrate when the lightning "
        "network, season or region changes.",
        "- Skill numbers apply only to the data listed above.",
        "",
        f"Input contract v{man.get('input_contract_version', CONTRACT_VERSION)}, output contract "
        f"v{man.get('output_contract_version', OUTPUT_CONTRACT_VERSION)}.",
    ]
    (d / "model_card.md").write_text("\n".join(L) + "\n")
