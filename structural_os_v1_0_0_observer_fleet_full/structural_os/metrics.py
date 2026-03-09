from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def make_returns(px: pd.DataFrame, use_log: bool = True) -> pd.DataFrame:
    if use_log:
        out = np.log(px).diff()
    else:
        out = px.pct_change()
    return out.dropna(how="all")

def corr_from_cov(S: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    d = np.sqrt(np.maximum(np.diag(S), eps))
    Dinv = np.diag(1.0 / d)
    C = Dinv @ S @ Dinv
    C = 0.5 * (C + C.T)
    np.fill_diagonal(C, 1.0)
    return C

def estimate_corr(X: np.ndarray, estimator: str = "lw") -> np.ndarray:
    if estimator == "lw":
        cov = LedoitWolf().fit(X).covariance_
        return corr_from_cov(cov)
    S = np.cov(X, rowvar=False, ddof=1)
    return corr_from_cov(S)

def eig_decomp(C: np.ndarray):
    C = 0.5 * (C + C.T)
    w, V = np.linalg.eigh(C)
    w = np.maximum(w, 0.0)
    idx = np.argsort(w)[::-1]
    return w[idx], V[:, idx]

def spectrum_metrics(w: np.ndarray):
    if np.sum(w) <= 0:
        return dict(s1=np.nan, erank=np.nan, pr=np.nan, gap=np.nan)
    p = w / np.sum(w)
    p = np.clip(p, 1e-18, 1.0)
    s1 = float(w[0] / np.sum(w))
    erank = float(np.exp(-np.sum(p * np.log(p))))
    pr = float(1.0 / np.sum(p ** 2))
    gap = float(w[0] - w[1]) if len(w) > 1 else np.nan
    return dict(s1=s1, erank=erank, pr=pr, gap=gap)

def _logm_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * np.log(w)) @ V.T

def _invsqrt_spd(A: np.ndarray, eps: float = 1e-12) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, eps)
    return (V * (1.0 / np.sqrt(w))) @ V.T

def affine_invariant_distance(C0: np.ndarray, C1: np.ndarray, eps: float = 1e-12) -> float:
    iS = _invsqrt_spd(C0, eps=eps)
    M = iS @ C1 @ iS
    L = _logm_spd(M, eps=eps)
    return float(np.linalg.norm(L, ord="fro"))

def theta_vec(v_prev: np.ndarray, v: np.ndarray) -> float:
    c = abs(float(np.dot(v_prev, v)))
    c = min(1.0, max(-1.0, c))
    return float(np.arccos(c))

def axis_composition(v: np.ndarray, tickers: list[str], t2g: dict[str, str], groups_order: list[str]) -> dict[str, float]:
    absw = np.abs(v)
    denom = float(absw.sum()) if absw.sum() > 0 else np.nan
    comp = {g: 0.0 for g in groups_order}
    for t, w in zip(tickers, absw):
        g = t2g.get(t, "Other")
        if g not in comp:
            comp[g] = 0.0
        comp[g] += float(w)
    if denom and denom > 0:
        for g in comp:
            comp[g] = comp[g] / denom
    else:
        for g in comp:
            comp[g] = np.nan
    return comp

def rolling_slope(series: pd.Series, window: int) -> pd.Series:
    x = np.arange(window, dtype=float)
    x_mean = x.mean()
    x_var = ((x - x_mean) ** 2).sum()
    def slope(y):
        y = np.asarray(y, dtype=float)
        if np.isnan(y).any():
            return np.nan
        y_mean = y.mean()
        cov = ((x - x_mean) * (y - y_mean)).sum()
        return cov / x_var
    return series.rolling(window).apply(slope, raw=False)

def bootstrap_v1_stability(X: np.ndarray, estimator: str = "lw", n_boot: int = 12, block: int = 10, rng_seed: int = 7) -> float:
    # returns median abs dot with reference v1
    n = X.shape[0]
    if n < max(24, block * 2):
        return np.nan
    rng = np.random.default_rng(rng_seed)
    C = estimate_corr(X, estimator=estimator)
    _, V = eig_decomp(C)
    ref = V[:, 0]
    sims = []
    for _ in range(n_boot):
        starts = rng.integers(0, max(1, n - block + 1), size=max(1, n // block))
        idx = []
        for s in starts:
            idx.extend(range(int(s), min(n, int(s + block))))
            if len(idx) >= n:
                break
        idx = np.array(idx[:n], dtype=int)
        Xb = X[idx]
        try:
            Cb = estimate_corr(Xb, estimator=estimator)
            _, Vb = eig_decomp(Cb)
            sims.append(abs(float(np.dot(ref, Vb[:, 0]))))
        except Exception:
            continue
    if not sims:
        return np.nan
    return float(np.median(sims))
