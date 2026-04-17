# Union-of-Manifolds (UoM) logic extracted from TopOGraph.
#
# Provides standalone helper functions and a ``UoMMixin`` class that
# TopOGraph inherits from to keep the main orchestrator slim.

import copy
import warnings
from typing import Any, Optional

import numpy as np
import scipy.sparse as sp

from topo.base.ann import kNN
from topo.spectral.eigen import EigenDecomposition
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing


# ---------------------------------------------------------------------------
# Standalone helpers
# ---------------------------------------------------------------------------


def _to_float32_csr(A):
    if not sp.isspmatrix_csr(A):
        A = A.tocsr()
    if A.dtype != np.float32:
        A = A.astype(np.float32, copy=False)
    return A


def _symmetrize_geometric(P):
    """S_ij = sqrt(P_ij * P_ji) on overlapping support (CSR float32)."""
    P = _to_float32_csr(P)
    S = P.multiply(P.T)
    if S.nnz == 0:
        return S
    S.data = np.sqrt(S.data.astype(np.float64)).astype(np.float32, copy=False)
    S.eliminate_zeros()
    return S


def _normalized_laplacian(A):
    A = _to_float32_csr(A)
    n = A.shape[0]
    d = np.asarray(A.sum(axis=1)).ravel().astype(np.float64).clip(min=1e-12)
    Dmh = sp.diags((1.0 / np.sqrt(d)).astype(np.float32))
    return sp.eye(n, dtype=np.float32, format="csr") - (Dmh @ A @ Dmh)


def _eigengap_k(vals, k_max, k_min=2):
    vals = np.asarray(vals, dtype=float)
    if vals.size <= 2:
        return max(k_min, min(k_max, 2))
    gaps = np.diff(vals)
    if gaps.size <= 1:
        return max(k_min, min(k_max, 2))
    j = int(np.argmax(gaps[1:])) + 1
    return int(max(k_min, min(k_max, j + 1)))


def mbkm(X, n_clusters, random_state=0):
    """MiniBatch KMeans clustering (thin wrapper)."""
    from sklearn.cluster import MiniBatchKMeans

    n_use = int(max(2, n_clusters))
    batch = int(min(2048, max(256, 8 * n_use * n_use)))
    km = MiniBatchKMeans(
        n_clusters=n_use,
        batch_size=batch,
        n_init=10,
        max_no_improvement=30,
        reassignment_ratio=0.01,
        random_state=random_state,
        verbose=0,
    )
    return km.fit_predict(X.astype(np.float32, copy=False))


def consolidate_macros(W, labels, max_iters=100):
    """Merge fragile macro-components using conductance outlier detection."""
    W = _to_float32_csr(W)
    labels = labels.copy()
    for _ in range(max_iters):
        uniq = np.unique(labels)
        if uniq.size <= 2:
            break
        deg = np.asarray(W.sum(axis=1)).ravel().astype(np.float64)

        # Build per-component index lists, volumes, and conductances
        idx_list, phi, vols = [], [], []
        for g in uniq:
            idx = np.where(labels == g)[0]
            idx_list.append(idx)
            vols.append(float(W[idx, :].sum()))
            comp = W[np.ix_(idx, idx)]
            internal = float(comp.sum() - comp.diagonal().sum())
            ext = float(deg[idx].sum() - 2.0 * internal)
            phi.append(ext / (ext + 2.0 * internal + 1e-12))

        phi = np.array(phi, dtype=float)
        merged = False

        # Phase 1: merge tiny-volume components into their heaviest neighbour
        med_vol = float(np.median(vols))
        for i, (idx_t, vol) in enumerate(zip(idx_list, vols)):
            if vol < 0.6 * med_vol and len(idx_t) > 0:
                best_neighbor, best_w = None, -1.0
                for j, idx_other in enumerate(idx_list):
                    if j == i or len(idx_other) == 0:
                        continue
                    w = float(W[np.ix_(idx_t, idx_other)].sum())
                    if w > best_w:
                        best_w, best_neighbor = w, j
                if best_neighbor is not None:
                    labels[idx_t] = uniq[best_neighbor]
                    merged = True

        # Phase 2: conductance-based outlier merge
        if phi.size >= 3:
            q1, q3 = np.percentile(phi, [25, 75])
            thr = q3 + 1.0 * (q3 - q1)
            mask = phi > thr
            if mask.any():
                worst_pos = int(np.argmax(phi * mask))
                idx_worst = idx_list[worst_pos]
                best_neighbor, best_w = None, -1.0
                for pos, idx_other in enumerate(idx_list):
                    if pos == worst_pos:
                        continue
                    w = float(W[np.ix_(idx_worst, idx_other)].sum())
                    if w > best_w:
                        best_w, best_neighbor = w, uniq[pos]
                if best_neighbor is not None:
                    labels[idx_worst] = best_neighbor
                    merged = True

        if not merged:
            break
        _, labels = np.unique(labels, return_inverse=True)

    _, labels_new = np.unique(labels, return_inverse=True)
    return labels_new


def louvain_micro(S, random_state=0, max_passes=100, gamma=0.85):
    """
    Greedy Louvain modularity clustering on a weighted undirected graph *S*.
    No external dependencies beyond NumPy / SciPy.
    """
    rng = np.random.RandomState(int(random_state))
    S = _to_float32_csr(S)
    n = S.shape[0]
    if n <= 2 or S.nnz == 0:
        return np.zeros(n, dtype=int)

    w = float(S.sum())
    if w <= 0:
        return np.zeros(n, dtype=int)
    m2 = w
    ki = np.asarray(S.sum(axis=1)).ravel().astype(np.float64)

    labels = np.arange(n, dtype=int)
    com_deg = ki.copy()

    S = S.tocsr()
    indptr, indices, data = S.indptr, S.indices, S.data

    improved = True
    passes = 0
    while improved and passes < max_passes:
        improved = False
        passes += 1
        order = np.arange(n)
        rng.shuffle(order)

        for v in order:
            v_lab = labels[v]
            k_v = ki[v]

            start, end = indptr[v], indptr[v + 1]
            nbrs = indices[start:end]
            wts = data[start:end]

            com_w = {}
            for u, wvu in zip(nbrs, wts):
                cu = labels[u]
                com_w[cu] = com_w.get(cu, 0.0) + float(wvu)

            best_c, best_dq = v_lab, 0.0
            com_deg_v_removed = com_deg[v_lab] - k_v

            for c, k_v_in in com_w.items():
                if c == v_lab and com_deg_v_removed <= 0:
                    continue
                dq = k_v_in - gamma * (k_v * com_deg[c] / m2)
                if c != v_lab and dq > best_dq:
                    best_dq, best_c = dq, c
            if best_c != v_lab and best_dq > 1e-12:
                labels[v] = best_c
                com_deg[v_lab] -= k_v
                com_deg[best_c] += k_v
                improved = True

        _, labels = np.unique(labels, return_inverse=True)

    return labels


def find_components(P, random_state=0, consolidate=True, max_passes=100, gamma=0.85):
    """
    Discover macro-components under the Union-of-Manifolds hypothesis.

    Returns ``(n_comp, labels)``.
    """
    from scipy.sparse.csgraph import connected_components
    from scipy.sparse.linalg import eigsh

    S = _symmetrize_geometric(P)
    n = S.shape[0]
    if S.nnz == 0:
        return 1, np.zeros(n, dtype=int)

    n_cc, cc_labels = connected_components(S, directed=False, return_labels=True)
    if n_cc > 3:
        return n_cc, cc_labels

    micro = louvain_micro(
        S, random_state=random_state, max_passes=max_passes, gamma=gamma
    )
    _, micro_labels = np.unique(micro, return_inverse=True)
    k = micro_labels.max() + 1

    rows, cols = S.nonzero()
    vals = np.asarray(S[rows, cols]).ravel()
    mr, mc = micro_labels[rows], micro_labels[cols]
    u = mr < mc
    if not np.any(u):
        return 1, np.zeros(n, dtype=int)

    r, c, w = mr[u], mc[u], vals[u]
    idx = r * k + c
    acc = (
        np.bincount(idx, weights=w, minlength=k * k)
        .astype(np.float32, copy=False)
        .reshape(k, k)
    )
    W = sp.csr_matrix(acc + acc.T, dtype=np.float32)

    if W.nnz == 0 or k <= 2:
        return (
            (1, np.zeros(n, dtype=int))
            if k <= 1
            else (np.unique(micro_labels).size, micro_labels)
        )

    Lw = _normalized_laplacian(W)
    k_max = int(min(8, max(3, np.floor(np.sqrt(k) + 1))))
    nev = int(min(k_max + 1, max(2, k - 1)))
    vals_w, vecs_w = eigsh(Lw, k=nev, which="SM")
    order = np.argsort(vals_w)
    vals_w, vecs_w = vals_w[order], vecs_w[:, order]
    k_macro = _eigengap_k(vals_w[:nev], k_max=k_max, k_min=2)
    Uw = vecs_w[:, :k_macro]
    Uw /= np.linalg.norm(Uw, axis=1, keepdims=True) + 1e-12
    macro = mbkm(Uw, n_clusters=k_macro, random_state=random_state)

    if consolidate and np.unique(macro).size > 2:
        macro = consolidate_macros(W, macro)

    labels = macro[micro_labels]
    n_comp = int(np.unique(labels).size)
    return n_comp, labels


# ---------------------------------------------------------------------------
# Lightweight proxy kernel (used for tiny/aggregated blocks)
# ---------------------------------------------------------------------------


class _ProxyKernel:
    __slots__ = ("P", "K")

    def __init__(self, P):
        self.P = P
        self.K = P


# ---------------------------------------------------------------------------
# UoMMixin — mixed into TopOGraph
# ---------------------------------------------------------------------------


class UoMMixin:
    """
    Mixin providing Union-of-Manifolds (UoM) state initialization and the
    per-component fit pipeline used by ``TopOGraph.fit()``.
    """

    # ------------------------------------------------------------------
    # Interface contract — attributes supplied by the host class (TopOGraph).
    # Declaring them here as class-level annotations tells static analysers
    # (Pylance, mypy) that they exist on ``self`` without changing runtime
    # behaviour.
    # ------------------------------------------------------------------

    # Core geometry
    n: int
    verbosity: int
    random_state: Any

    # kNN / kernel settings
    backend: str
    base_knn: int
    base_metric: str
    base_kernel_version: str
    base_kernel: Any
    graph_knn: int
    graph_metric: str
    graph_kernel_version: str

    # Eigendecomposition settings
    n_eigs: int
    eigensolver: str
    eigen_tol: float
    diff_t: int

    # Intrinsic-dimensionality estimation settings
    id_method: str
    id_ks: Any
    id_metric: str
    id_quantile: float
    id_min_components: int
    id_max_components: int
    id_headroom: float

    # Memory / caching
    low_memory: bool
    BaseKernelDict: dict
    GraphKernelDict: dict

    # Projection
    projection_methods: Optional[list]

    # Computed state (written by TopOGraph.fit before _fit_uom is called)
    current_eigenbasis: str
    eigenbasis: Any
    n_jobs: int
    _knn_Z: Any
    _knn_msZ: Any
    _kernel_Z: Any
    _kernel_msZ: Any

    # Methods provided by the host class
    def _build_kernel(self, *args: Any, **kwargs: Any) -> Any: ...
    def spectral_layout(self, *args: Any, **kwargs: Any) -> Any: ...
    def project(self, *args: Any, **kwargs: Any) -> Any: ...

    def _init_uom_state(self):
        """Initialize all UoM-specific attributes (called from TopOGraph.__init__)."""
        self.uom = getattr(self, "uom", False)
        self.uom_enabled = bool(self.uom)
        self.uom_comp_labels_: Optional[np.ndarray] = None
        self.uom_components_: Optional[list] = None

        self.uom_knn_X_list = None
        self.knn_X_uom = None
        self.P_of_X_uom = None
        self.uom_BaseKernel_list = None
        self.uom_DMEig_list = None
        self.uom_msDMEig_list = None
        self.uom_eigenvalues_dm_list = None
        self.uom_eigenvalues_ms_list = None
        self._uom_active_mode = "msDM"
        self.uom_Z_list = None
        self.uom_msZ_list = None
        self.uom_knn_Z_list = None
        self.uom_knn_msZ_list = None
        self.uom_Kernel_Z_list = None
        self.uom_Kernel_msZ_list = None

        self.Z_uom = None
        self.msZ_uom = None
        self.knn_Z_uom = None
        self.knn_msZ_uom = None
        self.P_of_Z_uom = None
        self.P_of_msZ_uom = None
        self._uom_axis_slices = None

        self.verbosity = getattr(self, "verbosity", 0)

    # -----------------------------------------------------------------
    # Public component-finding method (delegates to module function)
    # -----------------------------------------------------------------
    def uom_find_components(
        self, P, random_state=0, consolidate=True, max_passes=100, gamma=0.85
    ):
        """
        Discover disconnected macro-components under the UoM hypothesis.

        See :func:`find_components` for full docstring.
        """
        n_comp, labels = find_components(
            P,
            random_state=random_state,
            consolidate=consolidate,
            max_passes=max_passes,
            gamma=gamma,
        )
        self.uom_comp_labels_ = labels
        self.uom_components_ = [np.where(labels == c)[0] for c in np.unique(labels)]
        return n_comp, labels

    # -----------------------------------------------------------------
    # Per-component fit pipeline
    # -----------------------------------------------------------------
    def _fit_uom(self, X, **kwargs):
        """
        UoM branch of ``fit()``: detect components and build per-component
        scaffolds, refined graphs, and projections.  Aggregates results into
        block-diagonal operators.
        """
        if self.verbosity >= 1:
            print(
                "UoM: detecting disconnected components in P(X) and building per-component scaffolds/graphs..."
            )

        if (self.uom_comp_labels_ is not None) and (
            self.uom_comp_labels_.shape[0] == self.n
        ):
            labels = self.uom_comp_labels_
            n_comp = int(np.unique(labels).size)
            if self.verbosity >= 1:
                print(f"UoM: using precomputed component labels (n={n_comp}).")
        else:
            n_comp, labels = self.uom_find_components(P=self.base_kernel.P)
            if self.verbosity >= 1:
                print(f"UoM: computed component labels on refined graph (n={n_comp}).")

        self.uom_comp_labels_ = labels
        self.uom_components_ = [np.where(labels == c)[0] for c in np.unique(labels)]

        (
            self.uom_knn_X_list,
            self.uom_BaseKernel_list,
            self.uom_DMEig_list,
            self.uom_msDMEig_list,
        ) = [], [], [], []
        self.uom_Z_list, self.uom_msZ_list = [], []
        self.uom_knn_Z_list, self.uom_knn_msZ_list = [], []
        self.uom_Kernel_Z_list, self.uom_Kernel_msZ_list = [], []
        self.uom_eigenvalues_dm_list, self.uom_eigenvalues_ms_list = [], []

        def _local_size(Xi, n_max):
            n_i = Xi.shape[0]
            cap = min(int(self.id_max_components), max(2, n_i - 2))
            k_auto, _det = automated_scaffold_sizing(
                Xi,
                method=self.id_method,
                ks=self.id_ks,
                backend=self.backend,
                metric=self.id_metric,
                n_jobs=self.n_jobs if self.n_jobs != -1 else None,
                quantile=self.id_quantile,
                min_components=int(min(self.id_min_components, cap)),
                max_components=int(min(cap, n_max)),
                headroom=float(self.id_headroom),
                random_state=self.random_state,
                return_details=True,
            )
            return int(max(2, min(k_auto, cap)))

        for idx in self.uom_components_:
            n_i = int(idx.size)
            if n_i < 3:
                self._fit_uom_tiny_component(idx, n_i)
                continue

            # Per-component kNN
            k_neighbors_i = min(self.base_knn, max(1, n_i - 1))
            Xi = self._get_component_data(X, idx)

            knn_i = kNN(
                Xi,
                n_neighbors=k_neighbors_i,
                metric=self.base_metric,
                n_jobs=self.n_jobs,
                backend=self.backend,
                return_instance=False,
                verbose=False,
                **kwargs,
            )
            self.uom_knn_X_list.append(knn_i)

            Ki, _ = self._build_kernel(
                knn_i,
                min(self.base_knn, max(1, n_i - 1)),
                self.base_kernel_version,
                {} if self.low_memory else self.BaseKernelDict,
                suffix=f"_uom_X[{n_i}]",
                low_memory=self.low_memory,
                data_for_expansion=Xi,
                base=True,
            )
            self.uom_BaseKernel_list.append(Ki)

            k_i = _local_size(Xi if Xi is not None else knn_i, n_max=n_i - 2)
            Ki_mat = getattr(Ki, "K", None) or getattr(Ki, "P", None)
            N_i = int(Ki_mat.shape[0])
            if N_i <= 2:
                self._fit_uom_tiny_component(idx, N_i)
                continue

            k_req = int(min(max(k_i, 2), N_i - 1, self.n_eigs))
            k_req = max(1, k_req)

            eig_dm_i = EigenDecomposition(
                n_components=k_req,
                method="DM",
                eigensolver=self.eigensolver,
                eigen_tol=self.eigen_tol,
                drop_first=True,
                weight=True,
                t=self.diff_t,
                random_state=self.random_state,
                verbose=False,
            ).fit(Ki)

            eig_ms_i = copy.deepcopy(eig_dm_i)
            eig_ms_i.method = "msDM"

            self.uom_eigenvalues_dm_list.append(
                np.array(eig_dm_i.eigenvalues, copy=True)
            )
            self.uom_eigenvalues_ms_list.append(
                np.array(eig_ms_i.eigenvalues, copy=True)
            )

            k_avail = eig_dm_i.eigenvalues.shape[0]
            k_use = min(k_i, k_avail)
            Zi = eig_dm_i.transform()[:, :k_use]
            msZi = eig_ms_i.transform()[:, :k_use]

            self.uom_DMEig_list.append(eig_dm_i)
            self.uom_msDMEig_list.append(eig_ms_i)
            self.uom_Z_list.append(Zi)
            self.uom_msZ_list.append(msZi)

            k_graph_i = min(self.graph_knn, max(1, n_i - 1))
            knn_Z_i = kNN(
                Zi,
                n_neighbors=k_graph_i,
                metric=self.graph_metric,
                n_jobs=self.n_jobs,
                backend=self.backend,
                return_instance=False,
                verbose=False,
                **kwargs,
            )
            knn_msZ_i = kNN(
                msZi,
                n_neighbors=k_graph_i,
                metric=self.graph_metric,
                n_jobs=self.n_jobs,
                backend=self.backend,
                return_instance=False,
                verbose=False,
                **kwargs,
            )
            self.uom_knn_Z_list.append(knn_Z_i)
            self.uom_knn_msZ_list.append(knn_msZ_i)

            KZ_i, _ = self._build_kernel(
                knn_Z_i,
                k_graph_i,
                self.graph_kernel_version,
                {} if self.low_memory else self.GraphKernelDict,
                suffix=f"_uom_Z[{n_i}]",
                low_memory=self.low_memory,
                data_for_expansion=Zi,
                base=False,
            )
            KmsZ_i, _ = self._build_kernel(
                knn_msZ_i,
                k_graph_i,
                self.graph_kernel_version,
                {} if self.low_memory else self.GraphKernelDict,
                suffix=f"_uom_msZ[{n_i}]",
                low_memory=self.low_memory,
                data_for_expansion=msZi,
                base=False,
            )
            self.uom_Kernel_Z_list.append(KZ_i)
            self.uom_Kernel_msZ_list.append(KmsZ_i)

        # ----- Aggregate block-diagonal products -----
        self._aggregate_uom_blocks()

        # Spectral layout + projections
        _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)
        if self.projection_methods is not None:
            for proj in self.projection_methods:
                for ms in (True, False):
                    try:
                        self.project(projection_method=proj, multiscale=ms)
                    except Exception as e:
                        tag = "msZ" if ms else "Z/DM"
                        warnings.warn(
                            f"Projection '{proj}' on {tag} (UoM) failed: {e}",
                            RuntimeWarning,
                        )

        if self.low_memory:
            self.uom_BaseKernel_list = None
            self.uom_DMEig_list = None
            self.uom_msDMEig_list = None
            self.uom_Kernel_Z_list = None
            self.uom_Kernel_msZ_list = None

        return self

    # -----------------------------------------------------------------
    # Internal helpers
    # -----------------------------------------------------------------

    def _fit_uom_tiny_component(self, idx, n_i):
        """Fallback for components with < 3 samples."""
        Zi = np.zeros((n_i, 1), dtype=np.float32)
        self.uom_Z_list.append(Zi)
        self.uom_msZ_list.append(Zi.copy())

        P_block = sp.eye(n_i, format="csr", dtype=np.float32)
        self.uom_knn_Z_list.append(P_block.copy())
        self.uom_knn_msZ_list.append(P_block.copy())
        self.uom_Kernel_Z_list.append(_ProxyKernel(P_block))
        self.uom_Kernel_msZ_list.append(_ProxyKernel(P_block))
        self.uom_BaseKernel_list.append(_ProxyKernel(P_block))
        self.uom_knn_X_list.append(P_block.copy())
        self.uom_DMEig_list.append(None)
        self.uom_msDMEig_list.append(None)

    def _get_component_data(self, X, idx):
        """Slice input data for a UoM component."""
        if self.base_metric == "precomputed":
            return (
                X[np.ix_(idx, idx)]
                if X is not None
                else (
                    self.base_kernel.X[np.ix_(idx, idx)]
                    if getattr(self.base_kernel, "X", None) is not None
                    else None
                )
            )
        return (
            X[idx]
            if X is not None
            else (
                self.base_kernel.X[idx]
                if getattr(self.base_kernel, "X", None) is not None
                else None
            )
        )

    def _aggregate_uom_blocks(self):
        """Assemble per-component results into block-diagonal aggregates."""
        n = self.n

        total_cols_Z = int(sum(z.shape[1] for z in self.uom_Z_list))
        total_cols_msZ = int(sum(z.shape[1] for z in self.uom_msZ_list))
        self.Z_uom = np.zeros((n, total_cols_Z), dtype=np.float32)
        self.msZ_uom = np.zeros((n, total_cols_msZ), dtype=np.float32)
        self._uom_axis_slices = []
        c0 = 0
        for idx, Zi in zip(self.uom_components_, self.uom_Z_list):
            c1 = c0 + Zi.shape[1]
            self.Z_uom[idx, c0:c1] = Zi
            self._uom_axis_slices.append((c0, c1))
            c0 = c1
        c0 = 0
        for idx, msZi in zip(self.uom_components_, self.uom_msZ_list):
            c1 = c0 + msZi.shape[1]
            self.msZ_uom[idx, c0:c1] = msZi
            c0 = c1

        def _place_blocks(block_list):
            M = sp.lil_matrix((n, n), dtype=np.float32)
            for idx, B in zip(self.uom_components_, block_list):
                M[np.ix_(idx, idx)] = B
            return M.tocsr()

        self.knn_X_uom = _place_blocks(self.uom_knn_X_list)
        self.P_of_X_uom = _place_blocks([K.P for K in self.uom_BaseKernel_list])
        self.knn_Z_uom = _place_blocks(self.uom_knn_Z_list)
        self.knn_msZ_uom = _place_blocks(self.uom_knn_msZ_list)
        self.P_of_Z_uom = _place_blocks([K.P for K in self.uom_Kernel_Z_list])
        self.P_of_msZ_uom = _place_blocks([K.P for K in self.uom_Kernel_msZ_list])

        self.current_eigenbasis = f"UoM_{self._uom_active_mode}"
        self.eigenbasis = None
        self._knn_Z = self.knn_Z_uom
        self._knn_msZ = self.knn_msZ_uom
        self._kernel_Z = _ProxyKernel(self.P_of_Z_uom)
        self._kernel_msZ = _ProxyKernel(self.P_of_msZ_uom)
