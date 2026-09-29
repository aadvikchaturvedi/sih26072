"""Verification metrics.

All scores are *pooled*: accumulators sum counts over every sample and pixel,
and scores are computed once at the end (not averaged per sample).

Reflectivity: CSI/POD/FAR at dBZ thresholds per lead, FSS per lead and scale,
radially averaged power spectrum (sharpness).
Lightning: Brier score, BSS vs a reference, reliability diagram, ROC-AUC,
first-flash hit rate / false-alarm ratio / lead time.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import uniform_filter

# ---------------------------------------------------------------- contingency


def contingency(
    forecast: np.ndarray, observed: np.ndarray, threshold: float, valid=None
) -> np.ndarray:
    """[hits, misses, false_alarms, correct_negatives] for ``>= threshold`` events."""
    f = np.nan_to_num(forecast, nan=-np.inf) >= threshold
    o = np.nan_to_num(observed, nan=-np.inf) >= threshold
    # NaN forecasts count as "no event" (not excluded), so forecasters are scored on the same pixels.
    v = np.ones_like(f) if valid is None else np.asarray(valid, bool)
    return np.array(
        [(f & o & v).sum(), (~f & o & v).sum(), (f & ~o & v).sum(), (~f & ~o & v).sum()],
        dtype=np.int64,
    )


def _div(a, b):
    return float(a / b) if b > 0 else float("nan")


def scores_from_table(t) -> dict:
    h, m, fa, _ = (float(x) for x in t)
    return {"csi": _div(h, h + m + fa), "pod": _div(h, h + m), "far": _div(fa, h + fa)}


# ---------------------------------------------------------------- FSS


def fractions(binary: np.ndarray, scale: int) -> np.ndarray:
    return uniform_filter(binary.astype(np.float64), size=scale, mode="constant")


def fss_terms(forecast, observed, threshold: float, scale: int, valid=None) -> np.ndarray:
    """[sum (Pf-Po)^2, sum Pf^2 + Po^2] so FSS can be pooled: 1 - a/b."""
    f = np.nan_to_num(forecast, nan=-np.inf) >= threshold
    o = np.nan_to_num(observed, nan=-np.inf) >= threshold
    if valid is not None:
        f &= valid
        o &= valid
    pf, po = fractions(f, scale), fractions(o, scale)
    return np.array([((pf - po) ** 2).sum(), (pf**2).sum() + (po**2).sum()])


def fss(forecast, observed, threshold: float, scale: int, valid=None) -> float:
    a, b = fss_terms(forecast, observed, threshold, scale, valid)
    return 1.0 - a / b if b > 0 else float("nan")


# ---------------------------------------------------------------- probabilistic


def brier(p: np.ndarray, o: np.ndarray) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(o, float)) ** 2))


def brier_skill(bs: float, bs_ref: float) -> float:
    return 1.0 - bs / bs_ref if bs_ref > 0 else float("nan")


def reliability(p: np.ndarray, o: np.ndarray, n_bins: int = 10) -> dict:
    """Per-bin mean forecast probability, observed frequency and count."""
    p = np.clip(np.asarray(p, float).ravel(), 0, 1)
    o = np.asarray(o, float).ravel()
    idx = np.minimum((p * n_bins).astype(int), n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    sp = np.bincount(idx, weights=p, minlength=n_bins)
    so = np.bincount(idx, weights=o, minlength=n_bins)
    with np.errstate(invalid="ignore", divide="ignore"):
        return {
            "bin_edges": np.linspace(0, 1, n_bins + 1).tolist(),
            "mean_forecast": (sp / cnt).tolist(),
            "observed_freq": (so / cnt).tolist(),
            "count": cnt.tolist(),
        }


def roc_auc(p: np.ndarray, o: np.ndarray) -> float:
    o = np.asarray(o).ravel().astype(bool)
    if o.all() or not o.any():
        return float("nan")
    from sklearn.metrics import roc_auc_score

    return float(roc_auc_score(o, np.asarray(p).ravel()))


class ProbAccumulator:
    """Pools probabilistic forecasts; keeps a bounded random subsample for ROC-AUC."""

    def __init__(self, n_bins: int = 10, max_keep: int = 2_000_000, seed: int = 0):
        self.n_bins = n_bins
        self.max_keep = max_keep
        self.rng = np.random.default_rng(seed)
        self.sq = 0.0
        self.n = 0
        self.sum_o = 0.0
        self.sq_ref: dict[str, float] = {}
        self.cnt = np.zeros(n_bins)
        self.sp = np.zeros(n_bins)
        self.so = np.zeros(n_bins)
        self.keep_p: list[np.ndarray] = []
        self.keep_o: list[np.ndarray] = []
        self.kept = 0

    def update(self, p, o, valid=None, refs: dict[str, np.ndarray] | None = None):
        p = np.clip(np.asarray(p, float), 0, 1)
        o = np.asarray(o, float)
        m = np.ones(p.shape, bool) if valid is None else np.asarray(valid, bool)
        p, o = p[m], o[m]
        self.sq += ((p - o) ** 2).sum()
        self.n += p.size
        self.sum_o += o.sum()
        for k, r in (refs or {}).items():
            r = np.asarray(r, float)[m]
            self.sq_ref[k] = self.sq_ref.get(k, 0.0) + ((r - o) ** 2).sum()
        idx = np.minimum((p * self.n_bins).astype(int), self.n_bins - 1)
        self.cnt += np.bincount(idx, minlength=self.n_bins)
        self.sp += np.bincount(idx, weights=p, minlength=self.n_bins)
        self.so += np.bincount(idx, weights=o, minlength=self.n_bins)
        room = self.max_keep - self.kept
        if room > 0 and p.size:
            take = p.size if p.size <= room else room
            sel = self.rng.choice(p.size, take, replace=False) if take < p.size else slice(None)
            self.keep_p.append(p[sel])
            self.keep_o.append(o[sel])
            self.kept += take

    def result(self, climatology: float | None = None) -> dict:
        if self.n == 0:
            return {}
        bs = self.sq / self.n
        base = self.sum_o / self.n if climatology is None else climatology
        # Brier of a constant climatological forecast c: mean (c - o)^2
        bs_clim = base**2 - 2 * base * (self.sum_o / self.n) + self.sum_o / self.n
        out = {
            "brier": bs,
            "base_rate": self.sum_o / self.n,
            "climatology": base,
            "bss_climatology": brier_skill(bs, bs_clim),
            "n": int(self.n),
        }
        for k, v in self.sq_ref.items():
            out[f"bss_{k}"] = brier_skill(bs, v / self.n)
        with np.errstate(invalid="ignore", divide="ignore"):
            out["reliability"] = {
                "bin_edges": np.linspace(0, 1, self.n_bins + 1).tolist(),
                "mean_forecast": (self.sp / self.cnt).tolist(),
                "observed_freq": (self.so / self.cnt).tolist(),
                "count": self.cnt.astype(int).tolist(),
            }
        if self.keep_p:
            out["roc_auc"] = roc_auc(np.concatenate(self.keep_p), np.concatenate(self.keep_o))
        return out


# ---------------------------------------------------------------- first flash


def first_flash_onsets(occ_dilated: np.ndarray, quiet_frames: int = 3) -> np.ndarray:
    """(T, H, W) bool: flash within radius at t, none in the previous ``quiet_frames`` frames."""
    T = occ_dilated.shape[0]
    on = np.zeros_like(occ_dilated, dtype=bool)
    for t in range(quiet_frames, T):
        on[t] = occ_dilated[t] & ~occ_dilated[t - quiet_frames : t].any(axis=0)
    return on


def first_flash_scores(
    warnings: dict[int, np.ndarray],
    occ_dilated: np.ndarray,
    step_minutes: int = 10,
    max_lookback_min: int = 60,
    horizon_min: int = 30,
) -> dict:
    """Pixel-based first-flash verification for one event.

    ``warnings`` maps t0 index -> (H, W) first_flash map issued at t0.

    * An onset (t, y, x) is *hit* if some warning at that pixel was issued at a t0 in
      ``[t - max_lookback, t - 1]``; its lead time is ``t - (earliest such t0)``.
      Only onsets whose look-back window contains at least one scored t0 are counted.
    * A warning pixel is a *false alarm* if no flash occurs within radius during
      ``(t0, t0 + horizon]``.
    """
    T = occ_dilated.shape[0]
    onsets = first_flash_onsets(occ_dilated)
    back = max_lookback_min // step_minutes
    hor = horizon_min // step_minutes
    t0s = sorted(warnings)
    hits = misses = 0
    leads: list[float] = []
    for t in range(T):
        if not onsets[t].any():
            continue
        cands = [t0 for t0 in t0s if t - back <= t0 < t]
        if not cands:
            continue
        yy, xx = np.nonzero(onsets[t])
        first = np.full(yy.shape, -1)
        for t0 in cands:  # ascending: earliest first
            w = warnings[t0][yy, xx] & (first < 0)
            first[w] = t0
        hit = first >= 0
        hits += int(hit.sum())
        misses += int((~hit).sum())
        leads += ((t - first[hit]) * step_minutes).tolist()
    issued = fa = 0
    for t0 in t0s:
        w = warnings[t0]
        if t0 + hor >= T:
            continue
        fut = occ_dilated[t0 + 1 : t0 + 1 + hor].any(axis=0)
        issued += int(w.sum())
        fa += int((w & ~fut).sum())
    return {
        "hits": hits,
        "misses": misses,
        "issued": issued,
        "false_alarms": fa,
        "lead_times": leads,
    }


def summarize_first_flash(parts: list[dict]) -> dict:
    h = sum(p["hits"] for p in parts)
    m = sum(p["misses"] for p in parts)
    iss = sum(p["issued"] for p in parts)
    fa = sum(p["false_alarms"] for p in parts)
    leads = [x for p in parts for x in p["lead_times"]]
    return {
        "hit_rate": _div(h, h + m),
        "false_alarm_ratio": _div(fa, iss),
        "median_lead_min": float(np.median(leads)) if leads else float("nan"),
        "n_onsets": h + m,
        "n_warned_pixels": iss,
    }


# ---------------------------------------------------------------- sharpness


def radial_psd(field: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Radially averaged power spectrum of a 2-D field -> (wavenumber in cycles/px, power)."""
    f = np.nan_to_num(np.asarray(field, float))
    f = f - f.mean()
    H, W = f.shape
    win = np.outer(np.hanning(H), np.hanning(W))
    P = np.abs(np.fft.fftshift(np.fft.fft2(f * win))) ** 2
    ky = np.fft.fftshift(np.fft.fftfreq(H))
    kx = np.fft.fftshift(np.fft.fftfreq(W))
    k = np.sqrt(ky[:, None] ** 2 + kx[None, :] ** 2)
    nb = min(H, W) // 2
    bins = np.linspace(0, 0.5, nb + 1)
    idx = np.clip(np.digitize(k.ravel(), bins) - 1, 0, nb - 1)
    power = np.bincount(idx, weights=P.ravel(), minlength=nb) / np.maximum(
        np.bincount(idx, minlength=nb), 1
    )
    return 0.5 * (bins[1:] + bins[:-1]), power


def high_freq_power_ratio(
    forecast, observed, grid_spacing_km: float, max_wavelength_km: float = 16.0
) -> float:
    """Power at wavelengths < ``max_wavelength_km`` of forecast / observation (1 = as sharp)."""
    k, pf = radial_psd(forecast)
    _, po = radial_psd(observed)
    wavelength = grid_spacing_km / np.maximum(k, 1e-9)
    m = (wavelength < max_wavelength_km) & (k > 0)
    return _div(pf[m].sum(), po[m].sum())


# ---------------------------------------------------------------- ensembles


def crps_ensemble(members: np.ndarray, obs: np.ndarray) -> np.ndarray:
    """Per-pixel CRPS of an ensemble (M, ...) against ``obs`` (...).

    CRPS = mean_i |X_i - y| - 1/(2 M^2) sum_ij |X_i - X_j|, using the sorted-member
    identity sum_ij |X_i - X_j| = 2 sum_k (2k - M - 1) X_(k). With M = 1 it is the
    absolute error, so deterministic forecasts are scored on the same scale.
    """
    x = np.sort(np.asarray(members, float), axis=0)
    M = x.shape[0]
    y = np.asarray(obs, float)
    term1 = np.abs(x - y[None]).mean(axis=0)
    k = np.arange(1, M + 1).reshape((M,) + (1,) * (x.ndim - 1))
    pair = 2.0 * ((2 * k - M - 1) * x).sum(axis=0)
    return term1 - pair / (2.0 * M * M)
