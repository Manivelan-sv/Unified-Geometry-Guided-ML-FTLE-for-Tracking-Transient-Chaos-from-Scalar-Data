import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.neighbors import KNeighborsRegressor
from scipy.stats import gmean, linregress


def robust_scale(x: np.ndarray) -> np.ndarray:
    """
    Zero-median, robust scaling to ensure dimensionless calculations 
    invariant to local amplitude drift.
    
    Implements a two-stage scaling rule:
    1. Primary: Interquartile Range (IQR = Q_0.75 - Q_0.25).
    2. Fallback: Sample standard deviation + epsilon if IQR collapses below 1e-12.
    
    This prevents division by zero in highly stable/flat temporal windows.
    """
    med = float(np.median(x))
    iqr = float(np.percentile(x, 75) - np.percentile(x, 25))
    if iqr < 1e-12:
        iqr = float(np.std(x)) + 1e-12
    return (x - med) / iqr


def embed_for_ml(
    x: np.ndarray, m: int, tau: int, h: int
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Delay-coordinate embedding for a univariate time series.
    
    Establishes a localized, short-history temporal context for the forecast.
    Maps a historical state vector X (dimension m, lag tau) to a target value y 
    located exactly h steps ahead.
    """
    n_vecs = len(x) - (m - 1) * tau - h
    if n_vecs < 1:
        return None, None
    X = np.column_stack([x[i * tau : i * tau + n_vecs] for i in range(m)])
    y = x[(m - 1) * tau + h : (m - 1) * tau + h + n_vecs]
    return X, y


def _base_meta(
    m: int,
    tau: int,
    H_ftle: int,
    dt: float,
    n_valid: int = 0,
    r2: float = 0.0,
    reason: str = "",
) -> Dict:
    """Utility to return a consistent 7-key metadata dictionary."""
    return {
        "m":                int(m),
        "lag":              int(tau),
        "H_ftle":           int(H_ftle),
        "n_valid_horizons": int(n_valid),
        "r2":               float(r2),
        "dt":               float(dt),
        "reason":           str(reason),
    }


def ftle_for_window(
    x: np.ndarray,
    dt: float,
    *,
    m:            Optional[int]   = None,
    lag:          Optional[int]   = None,
    k_neighbors:  Optional[int]   = None,
    H_ftle:       Optional[int]   = None,
    # kept for API compatibility — silently ignored
    auto_lag:     Optional[bool]  = None,
    lag_max:      Optional[int]   = None,
    theiler_min:  Optional[int]   = None,
    max_basepts:  Optional[int]   = None,
    winsor_frac:  Optional[float] = None,
    n_bootstrap:  Optional[int]   = None,
    random_state: Optional[int]   = None,
) -> Tuple[float, float, float, Dict]:
    """
    Data-driven Machine Learning Finite Time Lyapunov Exponent (ML-FTLE) estimator.
    
    Quantifies local chaotic divergence by tracking the temporal growth rate of 
    out-of-sample forecast errors using distance-weighted k-NN regression.
    
    Pipeline:
      1. Pre-scale the data window robustly.
      2. For each horizon h, delay-embed the data.
      3. Apply a chronological train-test split to prevent look-ahead leakage.
      4. Forecast out-of-sample targets and compute the Geometric Mean Absolute Error (GMAE).
      5. Extract the Lyapunov proxy via OLS regression of ln(GMAE) vs h*dt.
      
    Returns:
      tuple: (ML-FTLE slope, 0.0, 0.0, metadata_dict)
    """
    import builtins as _b

    global EMB_DIM, TAU, HORIZON_MAX, KNN_NEIGH, TEST_RATIO

    # ── Resolve parameters: keyword arg > builtins > module default ──────────
    EMB_DIM     = (m           if m           is not None
                   else getattr(_b, "EMB_DIM",    EMB_DIM))
    TAU         = (lag         if lag         is not None
                   else getattr(_b, "TAU",
                        getattr(_b, "POINCARE_LAG", 5)))
    HORIZON_MAX = (H_ftle      if H_ftle      is not None
                   else getattr(_b, "HORIZON_MAX",
                        getattr(_b, "H_FTLE_MAX",   HORIZON_MAX)))
    KNN_NEIGH   = (k_neighbors if k_neighbors is not None
                   else getattr(_b, "KNN_NEIGH",
                        getattr(_b, "K_NEIGH",      KNN_NEIGH)))
    TEST_RATIO  = getattr(_b, "TEST_RATIO", 0.3)

    x   = np.asarray(x, float)
    _m  = int(EMB_DIM)
    _t  = int(TAU)        # <-- TAU is now defined before any use
    _H  = int(HORIZON_MAX)
    _k  = int(KNN_NEIGH)

    # ── Guard: Window too short -> implies stability (0.0), avoids NaN ───────
    if len(x) < 50:
        return 0.0, 0.0, 0.0, _base_meta(_m, _t, _H, dt, reason="too_short")

    # ── Pre-scale (Guarantees amplitude-independent GMAE calculations) ───────
    x_sc = robust_scale(x)

    # ── Guard: Flat/fixed-point dynamics -> implies stability (0.0) ──────────
    if np.std(x_sc) < 1e-6:
        return 0.0, 0.0, 0.0, _base_meta(_m, _t, _H, dt, reason="flat_signal")

    # ── Evaluate GMAE trajectory across all prediction horizons ──────────────
    horizons:    np.ndarray  = np.arange(1, _H + 1)
    gmae_values: List[float] = []

    for h in horizons:
        X, y = embed_for_ml(x_sc, _m, _t, int(h))
        if X is None or len(X) < 20:
            gmae_values.append(np.nan)
            continue
        
        # Chronological split to preserve structural history and avoid data leakage    
        n_train        = int(len(X) * (1.0 - TEST_RATIO))
        X_tr, X_te     = X[:n_train], X[n_train:]
        y_tr, y_te     = y[:n_train], y[n_train:]

        if len(X_te) < 2:
            gmae_values.append(np.nan)
            continue
        
        # Inverse-distance weighted k-NN prioritizes structurally proximate states   
        k_eff = min(_k, len(X_tr))
        model = KNeighborsRegressor(n_neighbors=k_eff, weights="distance")
        model.fit(X_tr, y_tr)

        abs_err = np.abs(y_te - model.predict(X_te))
        # Aggregate out-of-sample errors via GMAE.
        # No winsorisation is applied: maximal forecast errors remain strictly 
        # untruncated to retain the full dynamical spectrum of chaotic separation.
        gmae_values.append(float(gmean(np.maximum(abs_err, 1e-12))))

    # ── Restrict regression to valid horizons exhibiting positive divergence ──
    valid_idx = [
        i for i, v in enumerate(gmae_values)
        if np.isfinite(v) and v > 0
    ]
    # Handle degenerate windows lacking sufficient valid horizons
    if len(valid_idx) < 3:
        return 0.0, 0.0, 0.0, _base_meta(
            _m, _t, _H, dt,
            n_valid=len(valid_idx),
            reason="too_few_valid_horizons",
        )
    # ── Extract Lyapunov Proxy via Ordinary Least Squares (OLS) ──────────────
    # Fits: ln[GMAE(h)] ~ lambda_ML * (h * dt)
    x_fit = horizons[valid_idx].astype(float) * dt
    y_fit = np.log(np.array(gmae_values)[valid_idx])

    slope, _, r_val, _, _ = linregress(x_fit, y_fit)
    # Return the slope as the local instability proxy, storing R^2 as a reliability metric
    return float(slope), 0.0, 0.0, _base_meta(
        _m, _t, _H, dt,
        n_valid=len(valid_idx),
        r2=float(r_val ** 2),
        reason="",
    )