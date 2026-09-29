"""Write ``skill.md`` + PNG plots from evaluation results.

Plots: CSI vs lead (one panel per threshold), reliability diagram (one panel per
lightning lead), and a case-study strip (observed vs every forecaster at +30/+60/+120).
Colors follow a fixed categorical order per forecaster, so a forecaster keeps its
color across all figures and runs.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, ListedColormap  # noqa: E402

SURFACE = "#fcfcfb"
TEXT = "#0b0b0b"
TEXT_2 = "#52514e"
GRID = "#e4e3df"
NEUTRAL = "#8a8984"
# Validated categorical order (blue, orange, aqua, yellow, magenta, ...); fixed per forecaster.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
ORDER = ["model_full", "model_satellite_only", "extrapolation", "steps", "persistence"]
LABELS = {
    "model_full": "Model (full)",
    "model_satellite_only": "Model (satellite-only)",
    "extrapolation": "Extrapolation",
    "steps": "STEPS ensemble mean",
    "persistence": "Persistence",
}
# One-hue sequential ramps; lowest bin recedes to the surface ("no echo").
BLUE_RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
REFL_CMAP = ListedColormap([SURFACE] + BLUE_RAMP)
REFL_BOUNDS = [0, 15, 20, 25, 30, 35, 40, 45, 70]
PROB_CMAP = LinearSegmentedColormap.from_list("prob", [SURFACE, "#f6c3ad", "#eb6834", "#9c3a12"])


def _ordered(names):
    known = [n for n in ORDER if n in names]
    return known + sorted(n for n in names if n not in ORDER)


def _color(name, names):
    return SERIES[_ordered(names).index(name) % len(SERIES)]


def _style(ax):
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(GRID)
    ax.tick_params(colors=TEXT_2, labelsize=8)
    ax.grid(True, color=GRID, linewidth=0.6)
    ax.set_axisbelow(True)


def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return "–"
    return f"{x:.{nd}f}"


# ---------------------------------------------------------------- plots


def plot_csi(results: dict, path: Path) -> None:
    fc = results["forecasters"]
    names = _ordered(fc)
    leads = results["meta"]["leads_min"]
    thrs = list(next(iter(fc.values()))["reflectivity"]["csi"])
    fig, axes = plt.subplots(
        1, len(thrs), figsize=(4.2 * len(thrs), 3.6), sharey=True, facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, thr in zip(axes, thrs, strict=True):
        _style(ax)
        for n in names:
            y = fc[n]["reflectivity"]["csi"][thr]
            ax.plot(
                leads, y, color=_color(n, names), lw=2, marker="o", ms=4, label=LABELS.get(n, n)
            )
        ax.set_title(f"CSI ≥ {thr} dBZ", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("lead (min)", color=TEXT_2, fontsize=9)
        ax.set_xticks(leads[1::2])
        ax.set_ylim(0, 1)
    axes[0].set_ylabel("CSI", color=TEXT_2, fontsize=9)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper right")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(results: dict, path: Path) -> None:
    fc = results["forecasters"]
    names = _ordered(fc)
    ltg_leads = [str(x) for x in results["meta"]["lightning_leads_min"]]
    fig, axes = plt.subplots(
        1, len(ltg_leads), figsize=(4.2 * len(ltg_leads), 4.0), facecolor=SURFACE
    )
    axes = np.atleast_1d(axes)
    for ax, L in zip(axes, ltg_leads, strict=True):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=NEUTRAL, lw=1, ls="--", label="Perfect reliability")
        for n in names:
            r = fc[n]["lightning"].get(L, {}).get("reliability")
            if not r:
                continue
            mf, of, cnt = (
                np.asarray(r[k], float) for k in ("mean_forecast", "observed_freq", "count")
            )
            m = cnt >= 20  # hide bins with too few samples to estimate a frequency
            ax.plot(
                mf[m], of[m], color=_color(n, names), lw=2, marker="o", ms=4, label=LABELS.get(n, n)
            )
        ax.set_title(f"Lightning within 10 km, +{L} min", color=TEXT, fontsize=10, loc="left")
        ax.set_xlabel("forecast probability", color=TEXT_2, fontsize=9)
        ax.set_ylabel("observed frequency", color=TEXT_2, fontsize=9)
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    axes[-1].legend(frameon=False, fontsize=8, labelcolor=TEXT_2, loc="upper left")
    fig.tight_layout()
    fig.savefig(path, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def plot_case(case: dict, path: Path) -> None:
    names = _ordered(case["forecasts"])
    rows = [("Observed", case["observed"])] + [
        (LABELS.get(n, n), case["forecasts"][n]) for n in names
    ]
    ncol = len(case["leads_min"]) + 1
    fig, axes = plt.subplots(
        len(rows), ncol, figsize=(2.3 * ncol, 2.2 * len(rows)), facecolor=SURFACE
    )
    axes = np.atleast_2d(axes)
    from matplotlib.colors import BoundaryNorm

    norm = BoundaryNorm(REFL_BOUNDS, REFL_CMAP.N)
    im = pim = None
    for r, (label, frames) in enumerate(rows):
        for c, L in enumerate(case["leads_min"]):
            ax = axes[r, c]
            im = ax.imshow(
                np.nan_to_num(frames[c]), cmap=REFL_CMAP, norm=norm, interpolation="nearest"
            )
            ax.set_xticks([])
            ax.set_yticks([])
            for s in ax.spines.values():
                s.set_color(GRID)
            if r == 0:
                ax.set_title(f"+{L} min", color=TEXT, fontsize=9)
            if c == 0:
                ax.set_ylabel(label, color=TEXT_2, fontsize=8)
        ax = axes[r, -1]
        p = case["lightning_obs_60"] if r == 0 else case["lightning_60"][names[r - 1]]
        pim = ax.imshow(p, cmap=PROB_CMAP, vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
        for s in ax.spines.values():
            s.set_color(GRID)
        if r == 0:
            ax.set_title("Lightning +60\n(obs / prob)", color=TEXT, fontsize=9)
    fig.suptitle(
        f"Case study: {case['event_id']}  t0 = {case['t0']} UTC",
        color=TEXT,
        fontsize=10,
        x=0.01,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    cax = fig.add_axes((0.08, 0.025, 0.5, 0.012))
    cb = fig.colorbar(im, cax=cax, orientation="horizontal")
    cb.set_label("reflectivity (dBZ)", color=TEXT_2, fontsize=8)
    cb.ax.tick_params(labelsize=7, colors=TEXT_2)
    cax2 = fig.add_axes((0.68, 0.025, 0.25, 0.012))
    cb2 = fig.colorbar(pim, cax=cax2, orientation="horizontal")
    cb2.set_label("probability", color=TEXT_2, fontsize=8)
    cb2.ax.tick_params(labelsize=7, colors=TEXT_2)
    fig.savefig(path, dpi=110, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------- markdown


def skill_markdown(results: dict, has_case: bool) -> str:
    meta = results["meta"]
    fc = results["forecasters"]
    names = _ordered(fc)
    leads = meta["leads_min"]
    show = [L for L in (30, 60, 120) if L in leads]
    li = [leads.index(L) for L in show]
    out = ["# Nowcast skill report", ""]
    if meta.get("synthetic"):
        out += [
            "> **SYNTHETIC DATA — not indicative of real skill.** These scores come from the "
            "synthetic storm generator and only show that the pipeline works end to end. In synthetic "
            "data the satellite anvil is a deterministic function of the radar core, so the satellite-only "
            "row is unrealistically close to the full model.",
            "",
        ]
    out += [
        f"- Model: `{meta.get('model', 'none')}`",
        f"- Split: `{meta.get('split')}` · events: {meta['n_events']} · samples: {meta['n_samples']} "
        f"(t0 stride {meta['sample_stride']} frames)",
        f"- Data sources: {', '.join(meta.get('data_sources', []))}",
        f"- {meta['climatology_note']}.",
        "",
        "## Reflectivity: CSI / POD / FAR",
        "",
    ]
    for thr in meta["thresholds_dbz"]:
        t = str(int(thr))
        head = (
            "| Forecaster | "
            + " | ".join(f"CSI +{L}" for L in show)
            + " | "
            + " | ".join(f"POD +{L}" for L in show)
        )
        head += " | " + " | ".join(f"FAR +{L}" for L in show) + " |"
        out += [f"**≥ {t} dBZ**", "", head, "|" + "---|" * (1 + 3 * len(show))]
        for n in names:
            r = fc[n]["reflectivity"]
            cells = [_fmt(r[k][t][i]) for k in ("csi", "pod", "far") for i in li]
            out.append(f"| {LABELS.get(n, n)} | " + " | ".join(cells) + " |")
        out.append("")
    out += [f"## FSS at {meta['fss_threshold_dbz']:.0f} dBZ", ""]
    scales = [str(s) for s in meta["fss_scales_px"]]
    head = "| Forecaster | " + " | ".join(f"{s}px +{L}" for s in scales for L in show) + " |"
    out += [head, "|" + "---|" * (1 + len(scales) * len(show))]
    for n in names:
        f = fc[n]["reflectivity"]["fss"]
        out.append(
            f"| {LABELS.get(n, n)} | "
            + " | ".join(_fmt(f[s][i]) for s in scales for i in li)
            + " |"
        )
    out += ["", "## Lightning (P ≥1 flash within 10 km)", ""]
    out += [
        "| Forecaster | Lead | Brier | BSS vs climatology | BSS vs persists | ROC-AUC | base rate |",
        "|---|---|---|---|---|---|---|",
    ]
    for n in names:
        for L, r in fc[n]["lightning"].items():
            if not r:
                continue
            out.append(
                f"| {LABELS.get(n, n)} | +{L} | {_fmt(r['brier'], 4)} | {_fmt(r['bss_climatology'])} | "
                f"{_fmt(r.get('bss_persistence'))} | {_fmt(r.get('roc_auc'))} | {_fmt(r['base_rate'], 4)} |"
            )
    if {"extrapolation", "steps"} <= set(fc):
        out += [
            "",
            "Extrapolation and STEPS share the same lightning forecast (past-30-min flashes advected with the "
            "optical-flow motion), so their lightning rows are identical.",
        ]
    out += [
        "",
        f"## First flash (warning = P(+30) ≥ {meta['first_flash_threshold']} and no flash within 10 km in past 30 min)",
        "",
        "Pixel-based: an onset is a pixel that gets a flash within 10 km after 30 quiet minutes. "
        "Lead-time resolution equals the t0 stride.",
        "",
        "| Forecaster | hit rate | false-alarm ratio | median lead (min) | onsets |",
        "|---|---|---|---|---|",
    ]
    for n in names:
        f = fc[n]["first_flash"]
        out.append(
            f"| {LABELS.get(n, n)} | {_fmt(f['hit_rate'])} | {_fmt(f['false_alarm_ratio'])} | "
            f"{_fmt(f['median_lead_min'], 0)} | {f['n_onsets']} |"
        )
    out += [
        "",
        "## Sharpness and speed",
        "",
        "High-frequency power ratio at +60 min: power at wavelengths < 16 km, forecast / observed "
        "(1 = as sharp as observed, << 1 = blurred).",
        "",
        "| Forecaster | HF power ratio | ms / forecast |",
        "|---|---|---|",
    ]
    for n in names:
        out.append(
            f"| {LABELS.get(n, n)} | {_fmt(fc[n]['sharpness_hf_power_ratio_60min'])} | "
            f"{_fmt(fc[n]['ms_per_forecast'], 1)} |"
        )
    out += [
        "",
        "## Figures",
        "",
        "![CSI vs lead](csi_vs_lead.png)",
        "",
        "![Reliability](reliability.png)",
        "",
    ]
    if has_case:
        out += ["![Case study](case_study.png)", ""]
    return "\n".join(out)


def write_report(results: dict, out_dir: Path, case: dict | None = None) -> None:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    plot_csi(results, out_dir / "csi_vs_lead.png")
    plot_reliability(results, out_dir / "reliability.png")
    if case is not None:
        plot_case(case, out_dir / "case_study.png")
    (out_dir / "skill.md").write_text(skill_markdown(results, case is not None))
