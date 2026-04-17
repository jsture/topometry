# Standalone analysis functions extracted from TopOGraph.
#
# Each function accepts explicit arguments (operators, scaffolds, eigenvalues)
# rather than relying on a TopOGraph instance's internal state.

import numpy as np
import scipy.sparse as sp
from typing import Optional


# ---------------------------------------------------------------------------
# Spectral selectivity helpers (module-private)
# ---------------------------------------------------------------------------


def _std_cols(A, eps=1e-12):
    A = np.asarray(A, float)
    A = A - np.nanmean(A, axis=0, keepdims=True)
    sd = np.nanstd(A, axis=0, keepdims=True)
    return A / (sd + eps)


def _spectral_weights(
    evals=None, m=None, mode="lambda_over_one_minus_lambda", eps=1e-12
):
    if evals is None:
        return np.ones(m or 1, float)
    ev = np.asarray(evals, float)
    if mode == "lambda_over_one_minus_lambda":
        return ev / (1.0 - ev + eps)
    elif mode == "lambda":
        return ev
    return np.ones_like(ev)


def _compute_eas(Zs, w=None, eps=1e-12):
    """Entropy-based Axis Selectivity."""
    n, m = Zs.shape
    if w is None:
        w = np.ones(m, float)
    E = (Zs**2) * w[None, :]
    S = np.sum(E, axis=1, keepdims=True) + eps
    P = E / S
    H = -np.sum(P * np.log(P + eps), axis=1)
    Hmax = np.log(m)
    EAS = 1.0 - (H / (Hmax + eps))
    kstar = np.argmax(E, axis=1)
    sign_kstar = np.sign(Zs[np.arange(n), kstar])
    radius = np.sqrt(np.square(Zs).sum(1))
    return EAS, kstar, sign_kstar, radius


def _compute_radiality(Zs, k=30, metric="euclidean", eps=1e-12):
    from sklearn.neighbors import NearestNeighbors

    n = Zs.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k, n - 1), metric=metric).fit(Zs)
    _, idx = nn.kneighbors(Zs, return_distance=True)
    nbr = idx[:, 1:] if idx.shape[1] > 1 else idx
    r = np.linalg.norm(Zs, axis=1)
    r_med = np.median(r[nbr], axis=1)
    q75 = np.percentile(r[nbr], 75, axis=1)
    q25 = np.percentile(r[nbr], 25, axis=1)
    iqr = q75 - q25
    z = (r - r_med) / (iqr + eps)
    return z, r


def _compute_lac(Zs, k=30, metric="euclidean", eps=1e-12):
    """Local Axial Coherence = EVR1 of local PCA."""
    from sklearn.neighbors import NearestNeighbors
    from numpy.linalg import svd

    n = Zs.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k, n - 1), metric=metric).fit(Zs)
    _, idx = nn.kneighbors(Zs, return_distance=True)
    nbr = idx[:, 1:] if idx.shape[1] > 1 else idx
    out = np.zeros(n, float)
    for i in range(n):
        Znb = Zs[nbr[i]]
        Znb = Znb - Znb.mean(0, keepdims=True)
        _, s, _ = svd(Znb, full_matrices=False)
        num = s[0] ** 2
        den = (s**2).sum() + eps
        out[i] = num / den
    return out


# ---------------------------------------------------------------------------
# Public analysis functions
# ---------------------------------------------------------------------------


def spectral_selectivity(
    Z: np.ndarray,
    evals: Optional[np.ndarray] = None,
    *,
    weight_mode: str = "lambda_over_one_minus_lambda",
    standardize: bool = True,
    k_neighbors: int = 30,
    metric: str = "euclidean",
    P: Optional[sp.spmatrix] = None,
    smooth_t: int = 0,
) -> dict:
    """
    Per-sample spectral selectivity diagnostics on a scaffold *Z*.

    Parameters
    ----------
    Z : ndarray, shape (n, m)
        Scaffold coordinates.
    evals : ndarray or None
        Eigenvalues for axis weighting.  If None, uniform weights are used.
    weight_mode : {'lambda_over_one_minus_lambda', 'lambda', 'none'}
        How eigenvalues are converted to weights.
    standardize : bool
        Column-standardize Z before computing metrics.
    k_neighbors : int
        Neighborhood size for radiality / LAC.
    metric : str
        Metric for neighborhood search.
    P : sparse matrix or None
        Diffusion operator for optional smoothing of scalar fields.
    smooth_t : int
        Number of diffusion steps (requires *P*).

    Returns
    -------
    dict
        Keys: ``EAS``, ``RayScore``, ``LAC``, ``axis``, ``axis_sign``, ``radius``.
    """
    Zs = _std_cols(Z) if standardize else np.asarray(Z, float)
    w = _spectral_weights(evals, m=Z.shape[1], mode=weight_mode)

    EAS, kstar, sign_k, r = _compute_eas(Zs, w=w)
    z_rad, _ = _compute_radiality(Zs, k=k_neighbors, metric=metric)
    LAC = _compute_lac(Zs, k=k_neighbors, metric=metric)
    RayScore = (1.0 / (1.0 + np.exp(-z_rad))) * EAS

    if P is not None and int(smooth_t) > 0:

        def _smooth(v):
            u = np.asarray(v, float).copy()
            for _ in range(int(smooth_t)):
                u = P @ u
            return np.asarray(u).ravel()

        EAS = _smooth(EAS)
        RayScore = _smooth(RayScore)
        LAC = _smooth(LAC)

    return dict(
        EAS=EAS,
        RayScore=RayScore,
        LAC=LAC,
        axis=kstar.astype(int),
        axis_sign=(sign_k > 0).astype(int),
        radius=r,
    )


def filter_signal(signal, P, t: int = 8) -> np.ndarray:
    """
    Diffusion-filter a 1-D signal by applying ``P^t``.

    Parameters
    ----------
    signal : array-like, shape (n,)
        Scalar per-sample values to smooth.
    P : sparse matrix
        Row-stochastic diffusion operator.
    t : int
        Number of diffusion steps.

    Returns
    -------
    ndarray, shape (n,)
    """
    y = np.asarray(signal, float).copy().ravel()
    for _ in range(int(t)):
        y = P @ y
    return np.asarray(y).ravel()


def impute(X, P, t: int = 8, output: str = "auto", dtype=np.float64):
    """
    Diffusion-based imputation using ``P^t``.

    Parameters
    ----------
    X : array-like or sparse, shape (n, d)
        Data matrix.
    P : sparse matrix
        Row-stochastic diffusion operator.
    t : int
        Diffusion steps.
    output : {'auto', 'sparse', 'dense'}
        Output format.  ``'auto'`` preserves input sparsity.
    dtype : numpy dtype
        Computation dtype.

    Returns
    -------
    sparse or ndarray
    """
    if sp.issparse(X):
        Xc = X.tocsr(copy=True).astype(dtype)
        for _ in range(int(t)):
            Xc = P @ Xc
        if output in ("auto", "sparse"):
            return Xc
        return Xc.toarray()
    else:
        Xd = np.asarray(X, dtype=dtype)
        for _ in range(int(t)):
            Xd = P @ Xd
        if output in ("auto", "dense"):
            return Xd
        return sp.csr_matrix(Xd)


def riemann_diagnostics(
    Y: np.ndarray,
    L,
    *,
    center: str = "median",
    diffusion_t: int = 0,
    diffusion_op=None,
    normalize: str = "symmetric",
    clip_percentile: float = 2.0,
    return_limits: bool = True,
    compute_metric: bool = True,
    compute_scalars: bool = True,
) -> dict:
    """
    Riemann metric in 2-D and derived scalars (anisotropy, log det G, deformation).

    Parameters
    ----------
    Y : ndarray, shape (n, 2)
        2-D embedding.
    L : array-like
        Graph Laplacian.
    center : {'median', 'mean'}
    diffusion_t : int
        Diffusion smoothing steps for deformation maps.
    diffusion_op : sparse matrix or None
        Operator for smoothing (if *diffusion_t* > 0).
    normalize : str
    clip_percentile : float
    return_limits : bool
    compute_metric : bool
    compute_scalars : bool

    Returns
    -------
    dict
        Keys: ``G``, ``anisotropy``, ``logdetG``, ``deformation``, ``limits``.
    """
    from topo.eval.rmetric import RiemannMetric, calculate_deformation

    out = {}
    G = None
    if compute_metric:
        G = RiemannMetric(Y, L).get_rmetric()
        out["G"] = G

    if compute_scalars:
        if G is None:
            G = RiemannMetric(Y, L).get_rmetric()
            out["G"] = G
        lam = np.linalg.eigvalsh(G)  # type: ignore
        lam = np.clip(lam, 1e-12, None)
        out["anisotropy"] = np.log(lam[:, -1] / lam[:, 0])
        out["logdetG"] = np.sum(np.log(lam), axis=1)

    deform_vals, limits = calculate_deformation(
        Y,
        L,
        center=center,
        diffusion_t=int(max(0, diffusion_t)),
        diffusion_op=diffusion_op,
        normalize=normalize,
        clip_percentile=float(clip_percentile),
        return_limits=True,
    )
    out["deformation"] = deform_vals
    if return_limits:
        out["limits"] = limits

    return out
