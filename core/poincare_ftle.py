"""
core/poincare_ftle.py
─────────────────────
Single source of truth for all Poincaré-map FTLE logic.

This module implements the Geometry-Guided ML-FTLE framework. It translates 
topological deformations of reconstructed delay-coordinate attractors into a 
continuous Lyapunov-scale instability proxy ($\hat{\lambda}_{\mathrm{geo}}$).

Edit HERE once — every notebook that imports this module gets the fix.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import spearmanr
from sklearn.linear_model import LinearRegression as _LR
from sklearn.cross_decomposition import PLSRegression as _PLS
from sklearn.preprocessing import StandardScaler as _Scaler, QuantileTransformer as _QT
from sklearn.ensemble import GradientBoostingRegressor as _GBR
from scipy.ndimage import uniform_filter1d as _uf1d

EPS = 1e-12


# ══════════════════════════════════════════════════════════════════════════════
# 1.  I/O helpers
# ══════════════════════════════════════════════════════════════════════════════

def _detect_sep(line: str) -> str:
    """Auto-detect delimiter to handle varying export formats."""
    c, s = line.count(","), line.count(";")
    return "whitespace" if (c == 0 and s == 0) else ("," if c >= s else ";")


def read_table_any_sep(path: Path) -> pd.DataFrame:
    """Read CSV / TSV / whitespace-delimited file; always returns int-indexed columns."""
    first = ""
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            if ln.strip() and not ln.lstrip().startswith("#"):
                first = ln; break
    if not first:
        return pd.DataFrame()
    sep = _detect_sep(first)
    sep_arg = r"\s+" if sep == "whitespace" else sep
    parts = first.strip().split() if sep == "whitespace" else first.strip().split(sep)
    has_header = False
    try:
        [float(p) for p in parts if p.strip()]
    except ValueError:
        has_header = True
    df = pd.read_csv(path, sep=sep_arg, header=0 if has_header else None,
                     engine="python", comment="#")
    df.columns = range(df.shape[1])
    return df.dropna(axis=1, how="all")


def load_ftle(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Return valid time and Lyapunov exponent arrays from ML-FTLE or QR-FTLE files."""
    df = pd.read_csv(path)
    t_col = "time" if "time" in df.columns else df.columns[0]
    l_col = ("ftle_smooth" if "ftle_smooth" in df.columns
              else next((c for c in df.columns if "lam" in c.lower()), "ftle"))
    t   = df[t_col].values.astype(float)
    lam = df[l_col].values.astype(float)
    valid = np.isfinite(lam)
    n_nan = int((~valid).sum())
    if n_nan:
        print(f"  load_ftle: dropping {n_nan}/{len(lam)} NaN rows")
    if valid.sum() < 10:          # fallback to raw ftle column
        lam   = df["ftle"].values.astype(float) if "ftle" in df.columns else lam
        valid = np.isfinite(lam)
        print(f"  load_ftle: fell back to raw 'ftle' col ({valid.sum()} valid rows)")
    return t[valid], lam[valid]


def load_binary_grid(path: Path) -> np.ndarray:
    """Load a saved Poincaré PNG as a binary occupancy grid representing phase-space."""
    arr = plt.imread(str(path)).astype(float)
    if arr.ndim == 3:
        arr = arr[..., 0]
    if arr.max() > 1.5:
        arr = arr / 255.0
    return (arr > 0.5).astype(float)


# ══════════════════════════════════════════════════════════════════════════════
# 2.  Grid construction
# ══════════════════════════════════════════════════════════════════════════════

def _bresenham(x0, y0, x1, y1):
    """
    Interpolates consecutive state vectors to maintain topological 
    continuity of the trajectory in low-resolution phase spaces.
    """
    dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
    dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
    err = dx + dy; pts = []
    while True:
        pts.append((x0, y0))
        if x0 == x1 and y0 == y1:
            break
        e2 = 2 * err
        if e2 >= dy:
            err += dy; x0 += sx
        if e2 <= dx:
            err += dx; y0 += sy
    return pts


def _to_bin(v: float, vmin: float, vmax: float, B: int) -> int:
    """Maps a scalar value to a discrete spatial bin for the occupancy grid."""
    den = vmax - vmin
    if den <= 0:
        return B // 2
    return max(0, min(B - 1, int(np.floor((v - vmin) / den * (B - 1)))))


def poincare_grid_and_hist(
    window: np.ndarray,
    *,
    bins: int = 32,
    lag: int = 1,
    connect_lines: bool = True,
    line_thick: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Constructs a 2D delay-coordinate occupancy grid mapping the structural geometry 
    of the local phase space. This per-window normalization ensures comparisons 
    are invariant to amplitude drift.

    Returns
    -------
    grid : float ndarray [bins × bins]  (Binary topological representation)
    hist : float ndarray [bins²]        (Normalized empirical probability distribution)
    """
    N = max(int(lag), 1)
    B = bins
    if len(window) <= N + 1:
        g = np.zeros((B, B), float)
        return g, g.ravel() + EPS
        
    x = window.astype(float)
    px, py = x[:-N], x[N:]
    xmn, xmx = float(np.min(px)), float(np.max(px))
    ymn, ymx = float(np.min(py)), float(np.max(py))
    
    grid = np.zeros((B, B), bool)
    ix = np.array([_to_bin(v, xmn, xmx, B) for v in px])
    iy = np.array([_to_bin(v, ymn, ymx, B) for v in py])
    
    for a, b in zip(ix, iy):
        grid[a, b] = True
        
    if connect_lines and len(ix) >= 2:
        for k in range(len(ix) - 1):
            for a, b in _bresenham(ix[k], iy[k], ix[k + 1], iy[k + 1]):
                grid[a, b] = True
            if line_thick > 0:
                for da in range(-line_thick, line_thick + 1):
                    for db in range(-line_thick, line_thick + 1):
                        aa, bb = a + da, b + db
                        if 0 <= aa < B and 0 <= bb < B:
                            grid[aa, bb] = True
                            
    g = grid.astype(float)
    h = g.ravel() + EPS
    h /= h.sum()
    return g, h


# ══════════════════════════════════════════════════════════════════════════════
# 3.  Metric functions
# ══════════════════════════════════════════════════════════════════════════════
# These metrics evaluate distinct dimensions of phase-space deformation:
# JSD (Distributional), SSIM (Structural), HDF (Geometric Bounds), IOU (Overlap).

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Jensen-Shannon Divergence: Captures shifts in probability mass distribution."""
    try:
        from scipy.spatial.distance import jensenshannon
        return float(jensenshannon(p, q, base=2))
    except Exception:
        pass
    p = np.asarray(p, float); p /= max(p.sum(), EPS)
    q = np.asarray(q, float); q /= max(q.sum(), EPS)
    m = 0.5 * (p + q)
    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return math.sqrt(max(0.5 * kl(p, m) + 0.5 * kl(q, m), 0.0))


def _ssim(ga: np.ndarray, gb: np.ndarray) -> float:
    """Structural Similarity Index: Highly sensitive to structured density deformations."""
    a = np.clip(np.asarray(ga, float), 0, 1)
    b = np.clip(np.asarray(gb, float), 0, 1)
    try:
        from skimage.metrics import structural_similarity as sk
        mn = int(min(a.shape)); ws = min(7, mn)
        ws = ws - 1 if ws % 2 == 0 else ws
        if ws >= 3:
            return float(np.clip(sk(a, b, data_range=1.0, win_size=ws), -1, 1))
    except Exception:
        pass
    mu_a, mu_b = np.mean(a), np.mean(b)
    va, vb = np.mean((a - mu_a) ** 2), np.mean((b - mu_b) ** 2)
    cov = np.mean((a - mu_a) * (b - mu_b))
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    num = (2 * mu_a * mu_b + c1) * (2 * cov + c2)
    den = (mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2)
    return float(np.clip(num / den if den > 0 else
                         (1.0 if np.allclose(a, b) else 0.0), -1, 1))


def _iou(ga: np.ndarray, gb: np.ndarray) -> float:
    """Intersection over Union: Evaluates raw spatial overlap of the attractors."""
    a = np.asarray(ga) > 0; b = np.asarray(gb) > 0
    u = int(np.logical_or(a, b).sum())
    return 1.0 if u == 0 else float(np.logical_and(a, b).sum() / u)


def _hdf(ga: np.ndarray, gb: np.ndarray, bins: int) -> float:
    """
    Hausdorff Distance: Evaluates the maximum mismatch between bounding sets.
    Exceptionally responsive to sudden boundary-crisis-like phase space collapses.
    """
    ap = np.argwhere(np.asarray(ga) > 0)
    bp = np.argwhere(np.asarray(gb) > 0)
    if len(ap) == 0 and len(bp) == 0:
        return 0.0
    if len(ap) == 0 or len(bp) == 0:
        return float(math.sqrt(2.0) * max(bins - 1, 1))
    diff  = ap[:, None, :] - bp[None, :, :]
    dists = np.sqrt(np.sum(diff.astype(float) ** 2, axis=2))
    return max(float(np.max(np.min(dists, axis=1))),
               float(np.max(np.min(dists, axis=0))))


def raw_metric_value(
    method: str, ga: np.ndarray, gb: np.ndarray, *, bins: int = 32,
) -> float:
    """Compute the raw (un-normalised) similarity / distance for one method."""
    if method == "JSD":  return _jsd(ga.ravel() + EPS, gb.ravel() + EPS)
    if method == "SSIM": return _ssim(ga, gb)
    if method == "HDF":  return _hdf(ga, gb, bins)
    if method == "IOU":  return _iou(ga, gb)
    raise ValueError(f"Unknown method: {method!r}")


def difference_score(method: str, raw: float) -> float:
    """Map raw metric to a normalised topological distance ∈ [0, 1]."""
    if method in ("JSD", "HDF"): return float(max(raw, 0.0))
    if method == "IOU":          return float(np.clip(1.0 - raw, 0, 1))
    if method == "SSIM":         return float(np.clip((1.0 - raw) / 2.0, 0, 1))
    raise ValueError(f"Unknown method: {method!r}")


# ══════════════════════════════════════════════════════════════════════════════
# 4.  Greedy max-min representative selection
# ══════════════════════════════════════════════════════════════════════════════

def _select_reps(
    diff_fn, n: int, nn: int, min_rel_sep_pct: float,
) -> tuple[list[int], float]:
    """
    Constructs a compact basis of structurally diverse temporal anchors.
    
    Iteratively extracts representatives by maximizing their minimum dissimilarity 
    to the existing basis set, ensuring the dictionary spans the full geometric 
    diversity of the evolving trajectory.
    """
    sel = [n // 2]
    base_val = diff_fn(0, n - 1)
    min_sep  = max(1, int(n * min_rel_sep_pct / 100))

    while len(sel) < nn:
        best_i, best_d = -1, -1.0
        for i in range(n):
            if i in sel: continue
            # Enforce minimum temporal separation between basis states
            if not all(abs(i - s) >= min_sep for s in sel): continue
            
            d = min(diff_fn(i, s) for s in sel)
            if d > best_d:
                best_d, best_i = d, i

        if best_i < 0:
            if min_sep > 1:
                min_sep = max(1, min_sep // 2)  # Relax constraint and retry
                continue
            else:
                break
        sel.append(best_i)

    return sorted(sel), base_val


# ══════════════════════════════════════════════════════════════════════════════
# 5.  Phase 1 + 2 — sliding window grids + closeness matrix
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class _Frame:
    idx: int
    center_sample: int
    path: Path


def run_phase1_2(dataset_dir: Path, P: Any) -> tuple[dict, dict]:
    """
    Phase 1: Generates delay-coordinate occupancy grids across sliding windows.
    Phase 2: Extracts basis attractors and computes the closeness matrix, projecting
             the evolving morphology into a multivariate structural trajectory.
    """
    WINDOW  = P.T_WINDOW_STEPS
    STEP    = P.T_STEP
    BINS    = P.BINS
    LAG     = P.POINCARE_LAG
    CL      = P.CONNECT_LINES
    THICK   = P.LINE_THICK
    METHODS = P.POINCARE_METHODS
    NN      = P.NN_POINCARE
    SEP     = P.MIN_REL_SEPARATION_PCT

    csv_dir   = dataset_dir / "csv_mkg"
    csv_files = sorted(csv_dir.glob("*.csv")) if csv_dir.exists() else []
    if not csv_files:
        print(f"  No CSVs in {dataset_dir.name}/csv_mkg"); return None, None

    all_frames: dict = {}
    recalc:     dict = {}

    for csv_path in csv_files:
        df_raw = read_table_any_sep(csv_path)
        if df_raw.empty: continue
        
        signal  = df_raw.iloc[:, 1].values.astype(float)
        n_wins  = (len(signal) - WINDOW) // STEP + 1
        frames  = [
            _Frame(idx=i, center_sample=i * STEP + WINDOW // 2,
                   path=dataset_dir / "poincare_maps" / f"{csv_path.stem}_w{i:05d}.png")
            for i in range(n_wins)
        ]
        all_frames[csv_path.stem] = frames

        # ── build all grids once (method-independent) ───────────────────────
        grids = []
        for fr in frames:
            s   = fr.center_sample - WINDOW // 2
            seg = signal[max(0, s): max(0, s) + WINDOW]
            g, _ = poincare_grid_and_hist(
                seg, bins=BINS, lag=LAG, connect_lines=CL, line_thick=THICK)
            grids.append(g)

        # ── per-method closeness series (design matrix generation) ───────────
        for method in METHODS:
            out_dir = dataset_dir / "METHODS" / method
            out_dir.mkdir(parents=True, exist_ok=True)
            cache: dict = {}

            def get_diff(i, j, _m=method):
                k = (min(i, j), max(i, j))
                if k not in cache:
                    raw = raw_metric_value(_m, grids[k[0]], grids[k[1]], bins=BINS)
                    cache[k] = difference_score(_m, raw)
                return cache[k]

            sel, base_val = _select_reps(get_diff, len(frames), NN, SEP)
            base = max(base_val, EPS)

            rows = []
            for fr in frames:
                dists = [get_diff(fr.idx, s) for s in sel]
                rows.append({
                    "center_index": fr.center_sample,
                    **{f"rep{i+1}_closeness": 1.0 - d / base for i, d in enumerate(dists)},
                })
            series = pd.DataFrame(rows)
            series.to_csv(out_dir / "recalc_series_to_representatives.csv", index=False)
            recalc[method] = {"sel": sel, "base": base, "series": series}
            print(f"  [{method}] {csv_path.stem}: {len(sel)} reps, base={base:.4g}")

    return all_frames, recalc


# ══════════════════════════════════════════════════════════════════════════════
# 6a.  Phase 3 — OLS proxy
# ══════════════════════════════════════════════════════════════════════════════

def align_and_regress(
    t_ftle: np.ndarray, lam: np.ndarray, t_m: np.ndarray, cl: np.ndarray,
) -> tuple:
    """Aligns topological closeness sequences to the predictive ML-FTLE timeframe."""
    n = cl.shape[1]
    cl_aln = np.column_stack([
        np.interp(t_ftle, t_m, cl[:, i]) for i in range(n)
    ])
    if np.std(lam) < 1e-10:
        return cl_aln, np.zeros_like(lam), None, 1.0, 0.0
    model = _LR().fit(cl_aln, lam)
    pred  = model.predict(cl_aln)
    rho,  _ = spearmanr(lam, pred)
    rmse    = float(np.sqrt(np.mean((lam - pred) ** 2)))
    return cl_aln, pred, model, rho, rmse


def run_phase3_ols(
    dataset_dir: Path, recalc: dict, t_ftle: np.ndarray, lam_ml: np.ndarray,
) -> dict:
    fit: dict = {}
    methods_root = dataset_dir / "METHODS"
    for method in (recalc.keys() if recalc else [d.name for d in methods_root.iterdir() if d.is_dir()]):
        scsv = methods_root / method / "recalc_series_to_representatives.csv"
        if not scsv.exists(): continue
        try:
            df  = pd.read_csv(scsv)
            t_m = df["center_index"].values
            cl  = df[[c for c in df.columns if c.endswith("_closeness")]].values
            _, pred, model, rho, rmse = align_and_regress(t_ftle, lam_ml, t_m, cl)
            fit[method] = dict(t_ftle=t_ftle, lam_ml=lam_ml, t_m=t_m, cl=cl, pred=pred, rho=rho, rmse=rmse)
        except Exception as e:
            pass
    return fit


# ══════════════════════════════════════════════════════════════════════════════
# 6b.  Phase 3 — PLSR proxy
# ══════════════════════════════════════════════════════════════════════════════

def guided_pca_proxy(
    t_ftle: np.ndarray, lam_ml: np.ndarray, t_m: np.ndarray, cl: np.ndarray, n_components: int = 1,
) -> tuple:
    """
    Partial Least Squares Regression (PLSR) proxy bridge.

    Extracts a supervised latent geometric component from the high-dimensional 
    closeness matrix that maximally covaries with the ML-FTLE scale. This filters 
    out redundant structural variations and isolates the topological shift driving 
    the macroscopic loss of instability.
    """
    scaler    = _Scaler()
    cl_scaled = scaler.fit_transform(cl)

    # Time-alignment ensures consistent calibration targets
    cl_aln = np.column_stack([
        np.interp(t_ftle, t_m, cl_scaled[:, i])
        for i in range(cl_scaled.shape[1])
    ])

    if np.std(lam_ml) < 1e-10:
        return np.zeros_like(lam_ml), np.zeros(len(t_m)), np.zeros(len(t_m)), 1.0, 0.0

    pls = _PLS(n_components=n_components)
    pls.fit(cl_aln, lam_ml)

    # Project sequences onto the extracted latent topology vector
    guided_pc1   = pls.transform(cl_scaled).flatten()
    pred_proxy   = pls.predict(cl_scaled).flatten()
    pred_aligned = np.interp(t_ftle, t_m, pred_proxy)

    rho,  _ = spearmanr(lam_ml, pred_aligned)
    rmse    = float(np.sqrt(np.mean((lam_ml - pred_aligned) ** 2)))
    return pred_aligned, pred_proxy, guided_pc1, rho, rmse


def run_phase3_plsr(
    dataset_dir: Path, recalc: dict, t_ftle: np.ndarray, lam_ml: np.ndarray, n_components: int = 1,
) -> dict:
    """Executes PLSR regression to yield the geometry-to-instability calibration."""
    fit: dict = {}
    methods_root = dataset_dir / "METHODS"
    for method in (recalc.keys() if recalc else [d.name for d in methods_root.iterdir() if d.is_dir()]):
        scsv = methods_root / method / "recalc_series_to_representatives.csv"
        if not scsv.exists(): continue
        try:
            df  = pd.read_csv(scsv)
            t_m = df["center_index"].values
            cl  = df[[c for c in df.columns if c.endswith("_closeness")]].values
            pred_aln, pred_proxy, pc1, rho, rmse = guided_pca_proxy(
                t_ftle, lam_ml, t_m, cl, n_components=n_components)
            fit[method] = dict(t_ftle=t_ftle, lam_ml=lam_ml, t_m=t_m, cl=cl,
                               pred=pred_aln, pred_proxy=pred_proxy, pc1=pc1, rho=rho, rmse=rmse)
        except Exception as e:
            pass
    return fit


# ==============================================================================
# 6c.  Upgrade helpers  (multi-lag + nonlinear calibration)
# ==============================================================================

def poincare_multilag_grids(
    window, *, bins=32, lags=None, connect_lines=True, line_thick=0
):
    """Generates a composite set of grids spanning multiple delay embeddings."""
    if lags is None:
        lags = [1, 2, 4]
    return [poincare_grid_and_hist(window, bins=bins, lag=lg,
            connect_lines=connect_lines, line_thick=line_thick)[0]
            for lg in lags]


def _fit_calibrator(X, y, mode='plsr'):
    """Fits the closeness -> FTLE proxy mapping using linear or nonlinear bases."""
    if mode == 'gbr':
        qt = _QT(output_distribution='normal', n_quantiles=min(200, len(y)))
        Xt = qt.fit_transform(X)
        gbr = _GBR(n_estimators=80, max_depth=3, learning_rate=0.1, subsample=0.8, random_state=0)
        gbr.fit(Xt, y)
        return ('gbr', qt, gbr)
    if mode == 'plsr':
        sc = _Scaler()
        pls = _PLS(n_components=1)
        pls.fit(sc.fit_transform(X), y)
        return ('plsr', sc, pls)
    return ('ols', _LR().fit(X, y))


def _predict_calibrator(fitted, X):
    """Executes the projection mapping established by _fit_calibrator."""
    kind = fitted[0]
    if kind == 'gbr':
        _, qt, gbr = fitted
        return gbr.predict(qt.transform(X))
    if kind == 'plsr':
        _, sc, pls = fitted
        return pls.predict(sc.transform(X)).flatten()
    return fitted[1].predict(X)


def _unsupervised_proxy(cl_mat, lam_ml):
    """Fallback proxy scaling when valid calibration points are insufficient."""
    mean_cl = cl_mat.mean(axis=1)
    lo = float(np.nanpercentile(lam_ml, 5))
    hi = float(np.nanpercentile(lam_ml, 95))
    return lo + (1.0 - mean_cl) * (hi - lo)


# ══════════════════════════════════════════════════════════════════════════════
# 7.  Inline proxy series
# ══════════════════════════════════════════════════════════════════════════════

def poincare_proxy_series(
    signal: np.ndarray, tsig: np.ndarray, lam_ml: np.ndarray, tml: np.ndarray,
    tref: np.ndarray | None = None, window: int | None = None, step: int | None = None,
    methods: list[str] | None = None, bins: int | None = None, lag: int | None = None,
    connect_lines: bool | None = None, nn: int | None = None, min_rel_sep_pct: float | None = None,
    regressor: str = "plsr", n_components: int = 1, extra_lags: list | None = None, smooth_proxy: int = 0,
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """
    Computes Poincaré-map FTLE sequences inline.
    
    Robustness sweeps utilize this function to validate proxy reliability under 
    varying SNR constraints and dataset alterations without relying on disk I/O.
    """
    import builtins as _b

    def _bget(name, default): return getattr(_b, name, default)
    
    _tref = tref if tref is not None else tml

    W   = window          if window          is not None else _bget("TWINDOWSTEPS", 250)
    ST  = step            if step            is not None else _bget("TSTEP",          30)
    MTH = methods         if methods         is not None else _bget("POINCAREMETHODS", ["JSD", "SSIM", "HDF", "IOU"])
    B   = bins            if bins            is not None else _bget("BINS",            32)
    LAG = lag             if lag             is not None else _bget("POINCARELAG",      5)
    CL  = connect_lines   if connect_lines   is not None else _bget("CONNECTLINES",  True)
    NN  = nn              if nn              is not None else _bget("NNP",              3)
    SEP = min_rel_sep_pct if min_rel_sep_pct is not None else _bget("MINRELSEPARATIONPCT", 5.0)

    if len(signal) < W + 1:
        return {m: (_tref, np.full_like(_tref, np.nan)) for m in MTH}

    idx_starts = np.arange(0, signal.size - W + 1, ST, dtype=int)
    if len(idx_starts) < 2:
        return {m: (_tref, np.full_like(_tref, np.nan)) for m in MTH}
    centers = tsig[idx_starts + W // 2]

    _lags = extra_lags if extra_lags else [LAG]
    grids_per_lag = [
        [poincare_grid_and_hist(signal[s: s + W], bins=B, lag=_lg, connect_lines=CL)[0]
         for s in idx_starts]
        for _lg in _lags
    ]
    grids = grids_per_lag[0]
    n = len(grids)

    result: dict = {}
    for method in MTH:
        cache: dict = {}

        def get_diff(i, j, _m=method):
            k = (min(i, j), max(i, j))
            if k not in cache:
                raw = raw_metric_value(_m, grids[k[0]], grids[k[1]], bins=B)
                cache[k] = difference_score(_m, raw)
            return cache[k]

        sel, base_val = _select_reps(get_diff, n, NN, SEP)
        base = max(base_val, EPS)

        cl_parts = []
        for _lag_grids in grids_per_lag:
            _lc = {}
            def _gd(_i, _j, _lg=_lag_grids, _m=method, _c=_lc):
                _k = (min(_i, _j), max(_i, _j))
                if _k not in _c:
                    _c[_k] = difference_score(_m, raw_metric_value(_m, _lg[_k[0]], _lg[_k[1]], bins=B))
                return _c[_k]
            cl_parts.append(np.column_stack([
                np.array([1.0 - _gd(i, rep) / base for i in range(n)])
                for rep in sel
            ]))
        cl_mat = np.column_stack(cl_parts)

        cl_on_ml = np.column_stack([
            np.interp(tml, centers, cl_mat[:, c]) for c in range(cl_mat.shape[1])
        ])
        valid = np.all(np.isfinite(cl_on_ml), axis=1) & np.isfinite(lam_ml)
        if valid.sum() >= 10:
            fitted    = _fit_calibrator(cl_on_ml[valid], lam_ml[valid], mode=regressor)
            cl_on_ref = np.column_stack([
                np.interp(_tref, centers, cl_mat[:, c]) for c in range(cl_mat.shape[1])
            ])
            proxy = _predict_calibrator(fitted, cl_on_ref)
        else:
            proxy = np.interp(_tref, centers, _unsupervised_proxy(cl_mat, lam_ml))
        if smooth_proxy and smooth_proxy > 1:
            proxy = _uf1d(np.where(np.isfinite(proxy), proxy, 0.0), size=smooth_proxy)
        result[method] = (centers, proxy)
    return result


# ══════════════════════════════════════════════════════════════════════════════
# 8.  Backward-compatible public aliases
# ══════════════════════════════════════════════════════════════════════════════

POINCARE_METHODS: list[str] = ["JSD", "SSIM", "HDF", "IOU"]

def poincare_grid_fast(
    window, *, bins: int = 32, lag: int = 1, connect_lines: bool = True, line_thick: int = 0,
):
    """Alias for :func:`poincare_grid_and_hist` (returns grid only)."""
    g, _ = poincare_grid_and_hist(
        window, bins=bins, lag=lag, connect_lines=connect_lines, line_thick=line_thick,
    )
    return g

diff_score = difference_score

def select_representatives(
    grids: list, method: str, nn: int, *, bins: int = 32, min_rel_sep_pct: float = 5.0,
) -> tuple[list[int], float]:
    """Public wrapper for topological basis selection."""
    cache: dict = {}
    def diff_fn(i, j):
        k = (min(i, j), max(i, j))
        if k not in cache:
            raw      = raw_metric_value(method, grids[k[0]], grids[k[1]], bins=bins)
            cache[k] = difference_score(method, raw)
        return cache[k]
    return _select_reps(diff_fn, len(grids), nn, min_rel_sep_pct)