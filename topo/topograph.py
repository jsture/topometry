# TopoMetry high-level API — the TopOGraph class
#
# Author: David S Oliveira <david.oliveira(at)dpag(dot)ox(dot)ac(dot)uk>
#
import copy
import gc
import logging
import time
import warnings
from typing import Optional

import numpy as np
from scipy.sparse import issparse, csr_matrix
from sklearn.base import BaseEstimator, TransformerMixin

from topo.base.ann import kNN
from topo.tpgraph.kernels import Kernel
from topo.spectral.eigen import EigenDecomposition, spectral_layout
from topo.layouts.projector import Projector
from topo.tpgraph.intrinsic_dim import automated_scaffold_sizing
from topo.uom import UoMMixin
from topo import analysis as _analysis

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Valid kernel versions & data-driven dispatch
# ---------------------------------------------------------------------------

VALID_KERNEL_VERSIONS = frozenset(
    [
        "cknn",
        "fuzzy",
        "bw_adaptive",
        "bw_adaptive_alpha_decaying",
        "bw_adaptive_nbr_expansion",
        "bw_adaptive_alpha_decaying_nbr_expansion",
        "gaussian",
    ]
)

# Each entry maps to the Kernel constructor kwargs that *differ* between versions.
# Common kwargs (metric, n_neighbors, backend, …) are merged at call time.
_KERNEL_CONFIGS = {
    "cknn": dict(
        cknn=True,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        sigma=None,
    ),
    "fuzzy": dict(
        cknn=False,
        fuzzy=True,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        sigma=None,
    ),
    "bw_adaptive": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=False,
        sigma=None,
    ),
    "bw_adaptive_alpha_decaying": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=False,
        alpha_decaying=True,
        sigma=None,
    ),
    "bw_adaptive_nbr_expansion": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=True,
        alpha_decaying=False,
        sigma=None,
    ),
    "bw_adaptive_alpha_decaying_nbr_expansion": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=True,
        expand_nbr_search=True,
        alpha_decaying=True,
        sigma=None,
    ),
    "gaussian": dict(
        cknn=False,
        fuzzy=False,
        adaptive_bw=False,
        expand_nbr_search=False,
        alpha_decaying=False,
        # sigma is filled dynamically from self.sigma
    ),
}


# ============================================================================
# TopOGraph
# ============================================================================


class TopOGraph(UoMMixin, BaseEstimator, TransformerMixin):
    """
    Geometry-aware estimator that learns spectral scaffolds, refined operators,
    and 2-D layouts.

    Parameters
    ----------
    base_knn : int, default 30
        k-nearest neighbors for the base graph on input space.
    graph_knn : int, default 30
        k-nearest neighbors for the refined graph built in spectral scaffold space.
    min_eigs : int, default 128
        Minimum number of eigenpairs to compute for the scaffold.
    base_kernel : topo.tpgraph.Kernel or None, default None
        Pre-fitted kernel to reuse; if provided, ``fit`` skips base graph construction.
    laplacian_type : str, default 'normalized'
        Laplacian normalization for spectral computations.
    base_kernel_version : str, default 'bw_adaptive'
        Kernel choice for the base graph.
    graph_kernel_version : str, default 'bw_adaptive'
        Kernel choice for scaffold graphs.
    backend : str, default 'hnswlib'
        Approximate nearest-neighbor backend.
    base_metric : str, default 'cosine'
        Distance metric for the base kNN graph.
    graph_metric : str, default 'euclidean'
        Distance metric for kNN in scaffold space.
    diff_t : int, default 0
        Diffusion time for single-time scaffold.
    sigma : float, default 0.1
        Bandwidth for Gaussian kernels.
    delta : float, default 1.0
        Radius parameter for cKNN kernels.
    n_jobs : int, default -1
        Threads for kNN searches; -1 uses all cores.
    low_memory : bool, default False
        Avoid caching large kernel objects.
    eigen_tol : float, default 1e-8
        Tolerance for the eigensolver.
    eigensolver : str, default 'arpack'
        Solver for eigendecomposition.
    projection_methods : list[str], default ['MAP', 'PaCMAP']
        Layouts to compute during ``fit``.
    cache : bool, default True
        Cache kernel / eigen objects in dictionaries for reuse.
    verbosity : int, default 0
        Logging verbosity (0=silent, 1=major, 2+=layout, 3=debug).
    random_state : int or RandomState, default 42
        Random seed.
    id_method : str, default 'fsa'
        Intrinsic-dimensionality estimator for scaffold sizing.
    id_ks : int or iterable, default 50
        Neighborhood sizes for I.D. estimation.
    id_metric : str, default 'euclidean'
        Metric for I.D. estimation.
    id_quantile : float, default 0.99
    id_min_components : int, default 128
    id_max_components : int, default 1024
    id_headroom : float, default 0.5
    uom : bool, default False
        Enable Union-of-Manifolds (block-diagonal scaffolds).
    """

    def __init__(
        self,
        base_knn: int = 30,
        graph_knn: int = 30,
        min_eigs: int = 128,
        n_jobs: int = -1,
        projection_methods=None,
        base_kernel=None,
        base_kernel_version: str = "bw_adaptive",
        graph_kernel_version: str = "bw_adaptive",
        base_metric: str = "cosine",
        graph_metric: str = "euclidean",
        diff_t: int = 0,
        delta: float = 1.0,
        sigma: float = 0.1,
        low_memory: bool = False,
        eigen_tol: float = 1e-8,
        eigensolver: str = "arpack",
        backend: str = "hnswlib",
        cache: bool = True,
        verbosity: int = 0,
        random_state=42,
        laplacian_type: str = "normalized",
        # Intrinsic-dimensionality sizing
        id_method: str = "fsa",
        id_ks=50,
        id_metric: str = "euclidean",
        id_quantile: float = 0.99,
        id_min_components: int = 128,
        id_max_components: int = 1024,
        id_headroom: float = 0.5,
        # UoM
        uom: bool = False,
    ):
        if projection_methods is None:
            projection_methods = ["MAP", "PaCMAP"]

        # Core config
        self.base_knn = base_knn
        self.graph_knn = graph_knn
        self.min_eigs = min_eigs
        self.n_eigs = min_eigs
        self.n_jobs = n_jobs
        self.projection_methods = projection_methods
        self.base_kernel = base_kernel
        self.base_kernel_version = base_kernel_version
        self.graph_kernel_version = graph_kernel_version
        self.base_metric = base_metric
        self.graph_metric = graph_metric
        self.diff_t = diff_t
        self.delta = delta
        self.sigma = sigma
        self.low_memory = low_memory
        self.eigen_tol = eigen_tol
        self.eigensolver = eigensolver
        self.backend = backend
        self.cache = cache
        self.verbosity = verbosity
        self.random_state = random_state
        self.laplacian_type = laplacian_type

        # ID config
        self.id_method = id_method
        self.id_ks = id_ks
        self.id_metric = id_metric
        self.id_quantile = id_quantile
        self.id_min_components = id_min_components
        self.id_max_components = id_max_components
        self.id_headroom = id_headroom

        # Fitted state
        self.n = None
        self.m = None
        self.base_nbrs_class = None
        self.base_knn_graph = None
        self.eigenbasis = None
        self.current_eigenbasis = None
        self.current_graphkernel = None
        self.graph_kernel = None
        self.SpecLayout = None
        self.global_dimensionality = None
        self.local_dimensionality = None
        self._id_details = {"mle": None, "fsa": None}
        self._scaffold_components_dm = None
        self._scaffold_components_ms = None

        # Dual-scaffold products
        self._knn_msZ = None
        self._knn_Z = None
        self._kernel_msZ = None
        self._kernel_Z = None

        # MAP snapshots
        self.msTopoMAP_snapshots = []
        self.TopoMAP_snapshots = []

        # Verbosity toggles (derived)
        self.bases_graph_verbose = False
        self.layout_verbose = False

        # Legacy / benchmarking dictionaries
        self.BaseKernelDict = {}
        self.EigenbasisDict = {}
        self.GraphKernelDict = {}
        self.ProjectionDict = {}
        self.LocalScoresDict = {}
        self.RiemannMetricDict = {}
        self.runtimes = {}

        # UoM state (from mixin)
        self._init_uom_state()

    # ------------------------------------------------------------------
    # Representation
    # ------------------------------------------------------------------

    def __repr__(self):
        if self.n is None:
            return "TopOGraph (not fitted)"
        parts = [f"TopOGraph with {self.n} samples"]
        if self.m is not None:
            parts[0] += f" × {self.m} features"
        for label, d in [
            ("Base Kernels", self.BaseKernelDict),
            ("Eigenbases", self.EigenbasisDict),
            ("Graph Kernels", self.GraphKernelDict),
            ("Projections", self.ProjectionDict),
        ]:
            if d:
                parts.append(f"  {label}: {', '.join(d.keys())}")
        return "\n".join(parts)

    # ------------------------------------------------------------------
    # Backend / random-state helpers
    # ------------------------------------------------------------------

    def _parse_backend(self):
        for lib in ("hnswlib", "nmslib", "annoy", "faiss"):
            try:
                __import__(lib)
                setattr(self, f"_have_{lib}", True)
            except ImportError:
                setattr(self, f"_have_{lib}", False)
        if self.backend == "hnswlib" and not self._have_hnswlib:
            self.backend = "nmslib" if self._have_nmslib else "sklearn"
        elif self.backend == "nmslib" and not self._have_nmslib:
            self.backend = "hnswlib" if self._have_hnswlib else "sklearn"

    def _parse_random_state(self):
        if self.random_state is None:
            self.random_state = np.random.RandomState()
        elif isinstance(self.random_state, int):
            self.random_state = np.random.RandomState(self.random_state)

    # ------------------------------------------------------------------
    # Data-driven kernel builder (Phase 3)
    # ------------------------------------------------------------------

    def _build_kernel(
        self,
        knn,
        n_neighbors,
        kernel_version,
        results_dict,
        prefix="",
        suffix="",
        low_memory=False,
        base=True,
        data_for_expansion=None,
    ):
        """
        Build a :class:`Kernel` from a kNN graph and a named *kernel_version*.

        Returns ``(kernel, results_dict)``.
        """
        kernel_key = f"{prefix}{kernel_version}{suffix}"
        if kernel_key in results_dict:
            return results_dict[kernel_key], results_dict

        cfg = _KERNEL_CONFIGS[kernel_version].copy()

        # Expansion versions need the original data + correct metric
        if cfg.get("expand_nbr_search"):
            if data_for_expansion is None:
                raise ValueError(
                    f"data_for_expansion required for kernel version '{kernel_version}'."
                )
            metric = self.base_metric if base else self.graph_metric
        else:
            metric = "precomputed"

        # Gaussian uses self.sigma
        if kernel_version == "gaussian":
            cfg["sigma"] = self.sigma

        kernel = Kernel(
            metric=metric,
            n_neighbors=n_neighbors,
            pairwise=False,
            backend=self.backend,
            n_jobs=self.n_jobs,
            laplacian_type=self.laplacian_type,
            semi_aniso=False,
            anisotropy=1.0,
            cache_input=False,
            verbose=self.bases_graph_verbose,
            random_state=self.random_state,
            **cfg,
        ).fit(knn)

        gc.collect()
        if not low_memory:
            results_dict[kernel_key] = kernel
        return kernel, results_dict

    # ------------------------------------------------------------------
    # Intrinsic-dimension sizing
    # ------------------------------------------------------------------

    def _automated_sizing(self, X):
        n = X.shape[0]
        max_cap = min(int(self.id_max_components), max(2, n - 2))

        n_eigs_automated, id_details = automated_scaffold_sizing(
            X,
            method=self.id_method,
            ks=self.id_ks,
            backend=self.backend,
            metric=self.id_metric,
            n_jobs=self.n_jobs if self.n_jobs != -1 else None,
            quantile=self.id_quantile,
            min_components=int(self.id_min_components),
            max_components=int(max_cap),
            headroom=float(self.id_headroom),
            random_state=self.random_state,
            return_details=True,
        )
        self._id_details[self.id_method] = id_details
        k_sel = int(max(2, min(n_eigs_automated, max_cap)))
        self._scaffold_components_ms = k_sel
        self._scaffold_components_dm = k_sel
        self.n_eigs = int(max(self.n_eigs, k_sel))
        self.global_dimensionality = k_sel
        self.local_dimensionality = id_details.get("local_id_mle", None)

    # ------------------------------------------------------------------
    # fit() — decomposed into stages (Phase 4)
    # ------------------------------------------------------------------

    def fit(self, X=None, **kwargs):
        """
        Build base kNN → base kernel P(X) → dual eigenbases (DM + msDM) →
        refined scaffold graphs → 2-D projections.

        When ``uom=True``, detects disconnected components and builds
        per-component scaffolds + block-diagonal operators.
        """
        self._validate_inputs(X)
        self._setup_environment()
        self._build_base_graph(X, **kwargs)
        self._build_base_kernel(X, **kwargs)

        if self.base_metric != "precomputed":
            self._automated_sizing(X if X is not None else self.base_kernel.X)
            if self.verbosity >= 1:
                print(
                    f"Automated sizing → target components: {self._scaffold_components_ms} "
                    f"(n_eigs={self.n_eigs})"
                )

        self.uom_eigenvalues_dm_list, self.uom_eigenvalues_ms_list = [], []

        if self.uom_enabled:
            return self._fit_uom(X, **kwargs)

        return self._fit_global(X, **kwargs)

    # -- Stage helpers --

    def _validate_inputs(self, X):
        if self.base_kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(f"Invalid base_kernel_version: {self.base_kernel_version}")
        if self.graph_kernel_version not in VALID_KERNEL_VERSIONS:
            raise ValueError(
                f"Invalid graph_kernel_version: {self.graph_kernel_version}"
            )
        if X is not None:
            max_eigs = max(2, X.shape[0] - 2)
            if self.n_eigs > max_eigs:
                self.n_eigs = max_eigs
                warnings.warn(f"Clamped n_eigs to {max_eigs} (n_samples={X.shape[0]})")

    def _setup_environment(self):
        self._parse_backend()
        self._parse_random_state()
        if self.n_jobs == -1:
            try:
                from joblib import cpu_count

                self.n_jobs = cpu_count()
            except Exception:
                pass
        self.layout_verbose = self.verbosity >= 2
        self.bases_graph_verbose = self.verbosity >= 3

    def _build_base_graph(self, X, **kwargs):
        if X is None:
            if self.base_kernel is None:
                raise ValueError("X was not passed and no base_kernel provided.")
            if not isinstance(self.base_kernel, Kernel):
                raise ValueError("base_kernel must be a topo.tpgraph.Kernel instance.")
            if self.base_kernel.knn_ is None:
                raise ValueError("base_kernel has not been fitted.")
            self.n = self.base_kernel.knn_.shape[0]
            self.m = self.base_kernel.knn_.shape[1]
        else:
            if self.base_metric == "precomputed":
                self.base_knn_graph = X.copy()
            self.n, self.m = X.shape

        if self.base_knn_graph is None:
            if self.verbosity >= 1:
                print("Computing neighborhood graph (X space)...")
            t0 = time.time()
            self.base_nbrs_class, self.base_knn_graph = kNN(
                X,
                n_neighbors=self.base_knn,
                metric=self.base_metric,
                n_jobs=self.n_jobs,
                backend=self.backend,
                return_instance=True,
                verbose=self.bases_graph_verbose,
                **kwargs,
            )
            self.runtimes["kNN_X"] = time.time() - t0
            if self.verbosity >= 1:
                print(f"  Base kNN computed in {self.runtimes['kNN_X']:.3f}s")

    def _build_base_kernel(self, X, **kwargs):
        if self.base_kernel_version in self.BaseKernelDict:
            self.base_kernel = self.BaseKernelDict[self.base_kernel_version]
        else:
            t0 = time.time()
            self.base_kernel, self.BaseKernelDict = self._build_kernel(
                self.base_knn_graph,
                self.base_knn,
                self.base_kernel_version,
                self.BaseKernelDict,
                low_memory=self.low_memory,
                data_for_expansion=X,
                base=True,
            )
            self.runtimes["Kernel_X"] = time.time() - t0
            if self.verbosity >= 1:
                print(
                    f"  Base kernel ({self.base_kernel_version}) in {self.runtimes['Kernel_X']:.3f}s"
                )

    def _fit_global(self, X, **kwargs):
        """Global (non-UoM) scaffold construction."""
        if self.verbosity >= 1:
            print("Computing eigenbasis → DM/msDM scaffolds...")

        dm_key = f"DM with {self.base_kernel_version}"
        ms_key = f"msDM with {self.base_kernel_version}"

        # Eigendecomposition (shared spectrum, different transforms)
        if dm_key not in self.EigenbasisDict:
            t0 = time.time()
            dm_eig = EigenDecomposition(
                n_components=self.n_eigs,
                method="DM",
                eigensolver=self.eigensolver,
                eigen_tol=self.eigen_tol,
                drop_first=True,
                weight=True,
                t=self.diff_t,
                random_state=self.random_state,
                verbose=self.bases_graph_verbose,
            ).fit(self.base_kernel)
            self.EigenbasisDict[dm_key] = dm_eig
            self.runtimes[dm_key] = time.time() - t0
            if self.verbosity >= 1:
                print(f"  DM/msDM eigenpairs in {self.runtimes[dm_key]:.3f}s")
        else:
            dm_eig = self.EigenbasisDict[dm_key]

        if ms_key not in self.EigenbasisDict:
            ms_eig = copy.deepcopy(dm_eig)
            ms_eig.method = "msDM"
            self.EigenbasisDict[ms_key] = ms_eig
        else:
            ms_eig = self.EigenbasisDict[ms_key]

        self.current_eigenbasis = ms_key
        self.eigenbasis = self.EigenbasisDict[ms_key]

        # Scaffold-space kNN + refined kernels
        self._build_scaffold_graphs(X, dm_eig, ms_eig, dm_key, ms_key, **kwargs)

        self.graph_kernel = self._kernel_msZ
        self.current_graphkernel = f"{self.graph_kernel_version} from {ms_key}"

        # Spectral layout + projections
        _ = self.spectral_layout(graph=self._kernel_msZ.K, n_components=2)
        self._run_projections()
        return self

    def _build_scaffold_graphs(self, X, dm_eig, ms_eig, dm_key, ms_key, **kwargs):
        """Build kNN and refined kernels in both scaffold spaces."""
        # msZ
        if self.verbosity >= 1:
            print("Computing kNN (msZ space)...")
        t0 = time.time()
        ms_target = ms_eig.transform(X)[:, : self._scaffold_components_ms]
        self._knn_msZ = kNN(
            ms_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs,
        )
        self.runtimes["kNN_msZ"] = time.time() - t0

        # Z (DM)
        if self.verbosity >= 1:
            print("Computing kNN (Z/DM space)...")
        t0 = time.time()
        dm_target = dm_eig.transform(X)[:, : self._scaffold_components_dm]
        self._knn_Z = kNN(
            dm_target,
            n_neighbors=self.graph_knn,
            metric=self.graph_metric,
            n_jobs=self.n_jobs,
            backend=self.backend,
            return_instance=False,
            verbose=self.bases_graph_verbose,
            **kwargs,
        )
        self.runtimes["kNN_Z"] = time.time() - t0

        # Refined kernels
        t0 = time.time()
        self._kernel_msZ, self.GraphKernelDict = self._build_kernel(
            self._knn_msZ,
            self.graph_knn,
            self.graph_kernel_version,
            self.GraphKernelDict,
            suffix=f" from {ms_key}",
            low_memory=self.low_memory,
            data_for_expansion=ms_eig.transform(X),
            base=False,
        )
        self.runtimes["Kernel_msZ"] = time.time() - t0

        t0 = time.time()
        self._kernel_Z, self.GraphKernelDict = self._build_kernel(
            self._knn_Z,
            self.graph_knn,
            self.graph_kernel_version,
            self.GraphKernelDict,
            suffix=f" from {dm_key}",
            low_memory=self.low_memory,
            data_for_expansion=dm_eig.transform(X),
            base=False,
        )
        self.runtimes["Kernel_Z"] = time.time() - t0

    def _run_projections(self):
        """Compute requested 2-D projections on both scaffolds."""
        if self.projection_methods is None:
            return
        for proj in self.projection_methods:
            for ms in (True, False):
                try:
                    self.project(projection_method=proj, multiscale=ms)
                except Exception as e:
                    tag = "msZ" if ms else "Z/DM"
                    warnings.warn(
                        f"Projection '{proj}' on {tag} failed: {e}", RuntimeWarning
                    )

    # ------------------------------------------------------------------
    # Spectral scaffold accessor
    # ------------------------------------------------------------------

    def spectral_scaffold(self, multiscale: bool = True):
        """Return scaffold coordinates (n_samples × n_eigs)."""
        if self.uom_enabled:
            arr = self.msZ_uom if multiscale else self.Z_uom
            if arr is None:
                raise AttributeError(
                    "UoM scaffold not available. Call .fit(X, uom=True)."
                )
            return arr
        key = f"{'msDM' if multiscale else 'DM'} with {self.base_kernel_version}"
        if key not in self.EigenbasisDict:
            raise AttributeError("Scaffold not found. Call .fit() first.")
        return self.EigenbasisDict[key].transform(X=None)

    # ------------------------------------------------------------------
    # Properties (public API)
    # ------------------------------------------------------------------

    @property
    def eigenvalues(self):
        """Eigenvalues of the active eigenbasis (dict in UoM mode)."""
        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list:
            mode = getattr(self, "_uom_active_mode", "msDM")
            per_comp = (
                self.uom_eigenvalues_ms_list
                if mode == "msDM"
                else self.uom_eigenvalues_dm_list
            )
            sizes = [int(ix.size) for ix in (self.uom_components_ or [])]
            return {"mode": mode, "per_component": per_comp, "component_sizes": sizes}
        if self.current_eigenbasis is None:
            raise AttributeError("Eigenvalues unavailable. Call .fit() first.")
        return self.EigenbasisDict[self.current_eigenbasis].eigenvalues

    @property
    def knn_msZ(self):
        """kNN graph in msDM scaffold space."""
        if self.uom_enabled and self.knn_msZ_uom is not None:
            return self.knn_msZ_uom
        if self._knn_msZ is None:
            raise AttributeError("knn_msZ unavailable. Call .fit() first.")
        return self._knn_msZ

    @property
    def knn_Z(self):
        """kNN graph in DM scaffold space."""
        if self.uom_enabled and self.knn_Z_uom is not None:
            return self.knn_Z_uom
        if self._knn_Z is None:
            raise AttributeError("knn_Z unavailable. Call .fit() first.")
        return self._knn_Z

    @property
    def P_of_msZ(self):
        """Diffusion operator on the msDM scaffold."""
        if self.uom_enabled and self.P_of_msZ_uom is not None:
            return self.P_of_msZ_uom
        if self._kernel_msZ is None:
            raise AttributeError("P_of_msZ unavailable. Call .fit() first.")
        return self._kernel_msZ.P

    @property
    def P_of_Z(self):
        """Diffusion operator on the DM scaffold."""
        if self.uom_enabled and self.P_of_Z_uom is not None:
            return self.P_of_Z_uom
        if self._kernel_Z is None:
            raise AttributeError("P_of_Z unavailable. Call .fit() first.")
        return self._kernel_Z.P

    @property
    def knn_X(self):
        """Base kNN graph in input space."""
        if self.uom_enabled and self.knn_X_uom is not None:
            return self.knn_X_uom
        if self.base_knn_graph is None:
            raise AttributeError("knn_X unavailable. Call .fit() first.")
        return self.base_knn_graph

    @property
    def P_of_X(self):
        """Diffusion operator on input space."""
        if self.uom_enabled and self.P_of_X_uom is not None:
            return self.P_of_X_uom
        if self.base_kernel is None:
            raise AttributeError("P_of_X unavailable. Call .fit() first.")
        return self.base_kernel.P

    @property
    def global_id(self):
        """Global intrinsic dimensionality estimate."""
        return self.global_dimensionality

    @property
    def intrinsic_dim(self):
        """Structured intrinsic-dimensionality info (global + local)."""
        det = (self._id_details or {}).get(self.id_method)
        return {
            "method": self.id_method,
            "global": self.global_dimensionality,
            "local": self.local_dimensionality,
            "details": det,
        }

    # --- Embedding properties ---

    @property
    def TopoMAP(self):
        """2-D MAP layout on the DM refined graph."""
        return self._get_projection("MAP", multiscale=False)

    @property
    def msTopoMAP(self):
        """2-D MAP layout on the msDM refined graph."""
        return self._get_projection("MAP", multiscale=True)

    @property
    def TopoPaCMAP(self):
        """2-D PaCMAP layout on the DM refined graph."""
        return self._get_projection("PaCMAP", multiscale=False)

    @property
    def msTopoPaCMAP(self):
        """2-D PaCMAP layout on the msDM refined graph."""
        return self._get_projection("PaCMAP", multiscale=True)

    def _get_projection(self, method, multiscale):
        """Look up a projection from ProjectionDict."""
        tag = "msDM" if multiscale else "DM"
        # Standard key
        if method in ("MAP", "Isomap", "IsomorphicMDE", "IsometricMDE"):
            key = f"{method} of {self.graph_kernel_version} from {tag} with {self.base_kernel_version}"
        else:
            key = f"{method} of {tag} with {self.base_kernel_version}"
        if key in self.ProjectionDict:
            return self.ProjectionDict[key]
        # UoM fallback key
        uom_key = f"{method} of UoM {tag} with {self.base_kernel_version}"
        if uom_key in self.ProjectionDict:
            return self.ProjectionDict[uom_key]
        raise AttributeError(
            f"{method} ({tag}) embedding unavailable. Call .fit() first."
        )

    # ------------------------------------------------------------------
    # Spectral layout
    # ------------------------------------------------------------------

    def spectral_layout(self, graph=None, n_components=2):
        """Compute a spectral initialization for layout optimization."""
        if graph is None:
            if self._kernel_msZ is not None:
                graph = self._kernel_msZ.K
            elif self._kernel_Z is not None:
                graph = self._kernel_Z.K
            else:
                raise ValueError("No graph kernel available. Call .fit() first.")
        t0 = time.time()
        try:
            spt = spectral_layout(
                graph,
                n_components,
                self.random_state,
                laplacian_type=self.laplacian_type,
                eigen_tol=self.eigen_tol,
                return_evals=False,
            )
            expansion = 10.0 / np.abs(spt).max()
            spt = (spt * expansion).astype(np.float32) + self.random_state.normal(
                scale=0.0001, size=[graph.shape[0], n_components]
            ).astype(np.float32)
        except Exception:
            spt = EigenDecomposition(n_components=n_components).fit_transform(graph)
        self.runtimes["Spectral"] = time.time() - t0
        self.SpecLayout = spt
        gc.collect()
        return spt

    # ------------------------------------------------------------------
    # project()
    # ------------------------------------------------------------------

    def project(
        self,
        n_components: int = 2,
        init=None,
        projection_method: Optional[str] = None,
        landmarks=None,
        landmark_method: str = "kmeans",
        n_neighbors: Optional[int] = None,
        num_iters: int = 300,
        multiscale: bool = False,
        save_every=None,
        save_limit=None,
        save_callback=None,
        include_init_snapshot: bool = True,
        **kwargs,
    ):
        """Compute a 2-D projection and store it in ``ProjectionDict``."""
        if n_neighbors is None:
            n_neighbors = self.graph_knn
        if projection_method is None:
            projection_method = self.projection_methods[0]

        # Choose refined graph / scaffold
        tag = "msDM" if multiscale else "DM"
        if projection_method in ("MAP", "IsomorphicMDE", "IsometricMDE", "Isomap"):
            metric = "precomputed"
            input_mat = self.P_of_msZ if multiscale else self.P_of_Z
            key = f"{self.graph_kernel_version} from {tag} with {self.base_kernel_version}"
        else:
            metric = self.graph_metric
            if self.uom_enabled:
                input_mat = self.msZ_uom if multiscale else self.Z_uom
            else:
                eig_key = f"{tag} with {self.base_kernel_version}"
                input_mat = self.EigenbasisDict[eig_key].transform(X=None)
            key = f"{tag} with {self.base_kernel_version}"

        # Initialization
        if init is not None:
            if isinstance(init, np.ndarray):
                init_Y = init
            elif isinstance(init, str) and init in self.ProjectionDict:
                init_Y = self.ProjectionDict[init]
            else:
                raise ValueError(f"Invalid init: {init}")
        else:
            g = (
                self._kernel_msZ.K
                if (multiscale and self._kernel_msZ is not None)
                else (self._kernel_Z.K if self._kernel_Z is not None else None)
            )
            if g is None:
                raise ValueError("No refined kernel for spectral initialization.")
            self.SpecLayout = self.spectral_layout(graph=g, n_components=n_components)
            init_Y = self.SpecLayout

        projection_key = f"{projection_method} of {key}"
        t0 = time.time()

        proj = Projector(
            n_components=n_components,
            projection_method=projection_method,
            metric=metric,
            n_neighbors=self.graph_knn,
            n_jobs=self.n_jobs,
            landmarks=landmarks,
            landmark_method=landmark_method,
            num_iters=num_iters,
            init=init_Y,
            nbrs_backend=self.backend,
            keep_estimator=False,
            random_state=self.random_state,
            verbose=self.layout_verbose,
            save_every=save_every,
            save_limit=save_limit,
            save_callback=save_callback,
            include_init_snapshot=include_init_snapshot,
        )

        result = proj.fit_transform(input_mat, **kwargs)
        if isinstance(result, tuple) and len(result) == 2:
            Y, Y_aux = result
        else:
            Y, Y_aux = result, None

        self.runtimes[projection_key] = time.time() - t0
        if self.verbosity >= 1:
            uom_tag = " [UoM]" if self.uom_enabled else ""
            print(
                f"  {projection_method} ({'msZ' if multiscale else 'Z/DM'}{uom_tag}) in {self.runtimes[projection_key]:.3f}s"
            )

        self.ProjectionDict[projection_key] = Y

        if projection_method == "MAP" and Y_aux and isinstance(Y_aux, dict):
            checkpoints = Y_aux.get("checkpoints")
            if checkpoints:
                if multiscale:
                    self.msTopoMAP_snapshots = checkpoints
                else:
                    self.TopoMAP_snapshots = checkpoints
        return Y

    # ------------------------------------------------------------------
    # Eigenspectrum plot
    # ------------------------------------------------------------------

    def eigenspectrum(self, eigenbasis_key=None, **kwargs):
        """Scree plot (calls ``topo.plot.decay_plot``)."""
        try:
            import matplotlib.pyplot as plt  # noqa: F401
        except ImportError:
            print("Matplotlib not found.")
            return
        from topo.plot import decay_plot

        if getattr(self, "uom_enabled", False) and self.uom_eigenvalues_ms_list:
            mode = getattr(self, "_uom_active_mode", "msDM")
            ev_lists = (
                self.uom_eigenvalues_ms_list
                if mode == "msDM"
                else self.uom_eigenvalues_dm_list
            )
            sizes = [int(ix.size) for ix in (self.uom_components_ or [])]
            figs = []
            for j, ev in enumerate(ev_lists):
                figs.append(
                    decay_plot(
                        evals=ev,
                        title=f"Component {j} (n={sizes[j]}) · {mode}",
                        **kwargs,
                    )
                )
            return figs

        eb = (
            self.EigenbasisDict.get(eigenbasis_key)
            if eigenbasis_key
            else self.eigenbasis
        )
        if eb is None:
            raise AttributeError("No eigenbasis available.")
        return decay_plot(evals=eb.eigenvalues, title=eigenbasis_key, **kwargs)

    # ------------------------------------------------------------------
    # find_ideal_projection (grid-search MAP hyperparameters)
    # ------------------------------------------------------------------

    def find_ideal_projection(
        self,
        min_dist_grid=None,
        spread_grid=None,
        initial_alpha_grid=None,
        *,
        multiscale: bool = True,
        num_iters: int = 600,
        save_every: int = 10,
        metric: str = "euclidean",
        n_neighbors: int = 30,
        backend: str = "hnswlib",
        n_jobs: int = -1,
        times=(1, 2, 4),
        r: int = 32,
        k_for_pf1=None,
        symmetric_hint: bool = True,
        verbosity: int = 1,
    ):
        """Grid-search MAP hyperparameters and select the best 2-D projection."""
        from topo.eval.topo_metrics import topo_preserve_score, get_P

        if min_dist_grid is None:
            min_dist_grid = [0.2, 0.6, 1.0]
        if spread_grid is None:
            spread_grid = [0.8, 1.2, 1.6]
        if initial_alpha_grid is None:
            initial_alpha_grid = [0.4, 1.0, 1.6]

        PX_ref = self.base_kernel.P
        if not issparse(PX_ref):
            PX_ref = csr_matrix(PX_ref)

        combos = [
            (md, sp_, ia)
            for md in min_dist_grid
            for sp_ in spread_grid
            for ia in initial_alpha_grid
        ]

        best = {"score": -np.inf, "params": None, "snapshots": None, "Y": None}
        all_scores = []
        snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"

        for md, sp_, ia in combos:
            if verbosity >= 1:
                print(f"[Grid] MAP: min_dist={md}, spread={sp_}, initial_alpha={ia}")

            Y = self.project(
                projection_method="MAP",
                multiscale=bool(multiscale),
                num_iters=int(num_iters),
                save_every=int(save_every),
                include_init_snapshot=True,
                min_dist=float(md),
                spread=float(sp_),
                initial_alpha=float(ia),
            )

            snapshots = getattr(self, snap_attr, None) or []
            scores_this = []
            for snap in snapshots:
                Ysnap = snap["embedding"]
                PY = get_P(
                    Ysnap,
                    metric=metric,
                    n_neighbors=n_neighbors,
                    backend=backend,
                    n_jobs=n_jobs,
                )
                if not issparse(PY):
                    PY = csr_matrix(PY)
                score, parts = topo_preserve_score(
                    PX_ref,
                    PY,
                    times=times,
                    r=r,
                    symmetric_hint=symmetric_hint,
                    k_for_pf1=k_for_pf1,  # type: ignore
                )
                snap["metrics"] = {
                    k: float(v)
                    for k, v in [
                        ("TP", score),
                        ("PF1", parts.get("PF1", np.nan)),
                        ("PJS", parts.get("PJS", np.nan)),
                        ("SP", parts.get("SP", np.nan)),
                    ]
                }
                snap["hyperparams"] = {
                    "min_dist": md,
                    "spread": sp_,
                    "initial_alpha": ia,
                }
                scores_this.append(float(score))

            final_score = scores_this[-1] if scores_this else float("-inf")
            all_scores.append(
                {
                    "min_dist": md,
                    "spread": sp_,
                    "initial_alpha": ia,
                    "final_score": final_score,
                }
            )

            if final_score > best["score"]:
                best = {
                    "score": final_score,
                    "params": {"min_dist": md, "spread": sp_, "initial_alpha": ia},
                    "snapshots": [dict(s) for s in snapshots],
                    "Y": np.array(Y, copy=True),
                }

        if best["snapshots"] is not None:
            setattr(self, snap_attr, best["snapshots"])

        if best["params"] is not None:
            bp = best["params"]
            self.project(
                projection_method="MAP",
                multiscale=bool(multiscale),
                num_iters=int(num_iters),
                save_every=int(save_every),
                include_init_snapshot=True,
                **bp,
            )
            setattr(self, snap_attr, best["snapshots"])

        return {
            "best_params": best["params"],
            "best_score": best["score"],
            "scores": all_scores,
            "best_snapshots": best["snapshots"],
        }

    # ------------------------------------------------------------------
    # Visualization (delegates to topo.plot)
    # ------------------------------------------------------------------

    def visualize_optimization(
        self,
        num_iters: int = 600,
        save_every: int = 10,
        dpi: int = 120,
        color=None,
        *,
        multiscale: bool = True,
        filename: str | None = None,
        point_size: float = 3.0,
        fps: int = 20,
        include_init_snapshot: bool = True,
        overlay_metrics: bool = False,
    ):
        """Produce an animated GIF of MAP training snapshots."""
        from topo.plot import visualize_optimization as _viz

        if multiscale is None:
            multiscale = bool(self.msTopoMAP_snapshots) or not bool(
                self.TopoMAP_snapshots
            )

        snap_attr = "msTopoMAP_snapshots" if multiscale else "TopoMAP_snapshots"
        snapshots = getattr(self, snap_attr, None)

        if not snapshots or len(snapshots) < 2:
            self.project(
                projection_method="MAP",
                num_iters=max(int(num_iters), int(save_every)),
                save_every=int(save_every),
                include_init_snapshot=bool(include_init_snapshot),
                multiscale=bool(multiscale),
            )
            snapshots = getattr(self, snap_attr, None)

        if not snapshots:
            raise RuntimeError("No snapshots available.")

        tag = "msTopoMAP" if multiscale else "TopoMAP"
        path = _viz(
            snapshots,
            dpi=dpi,
            color=color,
            filename=filename,
            point_size=point_size,
            fps=fps,
            tag=tag,
            overlay_metrics=overlay_metrics,
        )
        if self.verbosity >= 1:
            print(f"Wrote {path} with {len(snapshots)} frames.")
        return path

    # ------------------------------------------------------------------
    # Analysis convenience wrappers (delegate to topo.analysis)
    # ------------------------------------------------------------------

    def _select_P_operator(self, which: str = "msZ"):
        """Resolve a diffusion operator by name."""
        which = str(which).lower()
        if which == "x":
            return self.P_of_X
        elif which == "z":
            return self.P_of_Z
        elif which == "msz":
            return self.P_of_msZ
        raise ValueError("`which` must be one of {'X', 'Z', 'msZ'}.")

    def spectral_selectivity(
        self,
        Z=None,
        evals=None,
        multiscale=True,
        use_scaffold_components=True,
        smooth_P=None,
        smooth_t=0,
        out_prefix="spectral",
        return_dict=True,
        **kwargs,
    ):
        """Per-sample spectral selectivity (delegates to ``topo.analysis``)."""
        if Z is None:
            Z = self.spectral_scaffold(multiscale=multiscale)
        if use_scaffold_components and self._scaffold_components_ms is not None:
            Z = Z[:, : int(self._scaffold_components_ms)]
        if evals is None:
            key = f"{'msDM' if multiscale else 'DM'} with {self.base_kernel_version}"
            ev = self.EigenbasisDict[key].eigenvalues
            evals = (
                ev[1 : Z.shape[1] + 1]
                if ev.shape[0] >= Z.shape[1] + 1
                else ev[: Z.shape[1]]
            )

        P = self._select_P_operator(smooth_P) if smooth_P else None
        result = _analysis.spectral_selectivity(
            Z, evals, P=P, smooth_t=smooth_t, **kwargs
        )

        for k, v in result.items():
            self.LocalScoresDict[f"{out_prefix}_{k}"] = v
        return result if return_dict else None

    def filter_signal(self, signal, t: int = 8, which: str = "msZ"):
        """Diffusion-filter a 1-D signal."""
        return _analysis.filter_signal(signal, self._select_P_operator(which), t)

    def impute(self, X, t: int = 8, which: str = "msZ", **kwargs):
        """Diffusion-based imputation."""
        return _analysis.impute(X, self._select_P_operator(which), t, **kwargs)

    def riemann_diagnostics(self, Y=None, L=None, diffusion_op=None, **kwargs):
        """Riemann metric + deformation scalars."""
        if Y is None:
            for prop in ("TopoMAP", "msTopoMAP", "TopoPaCMAP", "msTopoPaCMAP"):
                try:
                    Y = getattr(self, prop)
                    break
                except AttributeError:
                    continue
            if Y is None:
                Y = self.project(projection_method="MAP", multiscale=False)
        if L is None:
            L = self.base_kernel.L
        P = self._select_P_operator(diffusion_op) if diffusion_op else None
        result = _analysis.riemann_diagnostics(Y, L, diffusion_op=P, **kwargs)
        self.RiemannMetricDict["last"] = result
        return result

    # ------------------------------------------------------------------
    # I/O
    # ------------------------------------------------------------------

    def save(self, filename: str = "topograph.pkl", remove_base_class: bool = True):
        """Save this TopOGraph to a pickle file."""
        save_topograph(self, filename, remove_base_class)


# =========================================================================
# Module-level I/O helpers
# =========================================================================


def save_topograph(
    tg: TopOGraph, filename: str = "topograph.pkl", remove_base_class: bool = True
):
    """Save a TopOGraph object to a pickle file."""
    import pickle

    if not isinstance(tg, TopOGraph):
        raise TypeError("`tg` must be a TopOGraph instance.")
    if tg.base_nbrs_class is not None and remove_base_class:
        tg.base_nbrs_class = None
    with open(filename, "wb") as f:
        pickle.dump(tg, f, pickle.HIGHEST_PROTOCOL)
    print(f"TopOGraph saved at {filename}")


def load_topograph(filename: str) -> TopOGraph:
    """Load a TopOGraph from a pickle file."""
    import pickle

    with open(filename, "rb") as f:
        obj = pickle.load(f)
    if not isinstance(obj, TopOGraph):
        warnings.warn("Loaded object is not a TopOGraph.", RuntimeWarning)
    return obj
