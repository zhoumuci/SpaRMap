from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, diags, issparse

try:
    import torch
    import torch.nn.functional as F

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None
    F = None
    TORCH_AVAILABLE = False

from sparmap._logging import get_logger

logger = get_logger(__name__)


def _scanpy():
    try:
        import scanpy as sc
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        logger.error("[rep_utils]: scanpy is required for REP preprocessing.")
        raise ImportError("[rep_utils]: scanpy is required for REP preprocessing.") from exc
    return sc


def _pca_cls():
    try:
        from sklearn.decomposition import PCA
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        logger.error("[rep_utils]: scikit-learn is required for PCA-based REP preprocessing.")
        raise ImportError("[rep_utils]: scikit-learn is required for PCA-based REP preprocessing.") from exc
    return PCA


def _nearest_neighbors_cls():
    try:
        from sklearn.neighbors import NearestNeighbors
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        logger.error("[rep_utils]: scikit-learn is required for neighbor search.")
        raise ImportError("[rep_utils]: scikit-learn is required for neighbor search.") from exc
    return NearestNeighbors


def _kneighbors_graph(*args, **kwargs):
    try:
        from sklearn.neighbors import kneighbors_graph
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        logger.error("[rep_utils]: scikit-learn is required for neighbor graph construction.")
        raise ImportError("[rep_utils]: scikit-learn is required for neighbor graph construction.") from exc
    return kneighbors_graph(*args, **kwargs)


def _radius_neighbors_graph(*args, **kwargs):
    try:
        from sklearn.neighbors import radius_neighbors_graph
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        logger.error("[rep_utils]: scikit-learn is required for radius graph construction.")
        raise ImportError("[rep_utils]: scikit-learn is required for radius graph construction.") from exc
    return radius_neighbors_graph(*args, **kwargs)


def _to_dense_float32(x) -> np.ndarray:
    if issparse(x):
        return x.toarray().astype(np.float32, copy=False)
    return np.asarray(x, dtype=np.float32)


def _csr_float32(x) -> csr_matrix:
    if issparse(x):
        out = x.tocsr().astype(np.float32)
    else:
        out = csr_matrix(np.asarray(x, dtype=np.float32))
    out.eliminate_zeros()
    return out


def _summary_stats(values: np.ndarray) -> dict[str, float | int | None]:
    x = np.asarray(values, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return {"count": 0, "mean": None, "std": None, "min": None, "max": None, "median": None}
    return {
        "count": int(x.size),
        "mean": float(np.mean(x)),
        "std": float(np.std(x)),
        "min": float(np.min(x)),
        "max": float(np.max(x)),
        "median": float(np.median(x)),
    }


def _records_to_columnar(records: list[dict[str, Any]]) -> dict[str, list[Any]]:
    if len(records) == 0:
        return {}
    keys: list[str] = []
    for record in records:
        for key in record.keys():
            key_str = str(key)
            if key_str not in keys:
                keys.append(key_str)
    return {key: [record.get(key, None) for record in records] for key in keys}


def _normalize_sample_ids(sample_ids: pd.Series | np.ndarray | list[str], n_obs: int) -> np.ndarray:
    sid = pd.Series(np.asarray(sample_ids).reshape(-1)).astype("string").str.strip()
    if sid.shape[0] != int(n_obs):
        raise ValueError(f"[microdomain_rep]: sample_id length mismatch, got {sid.shape[0]}, expected {n_obs}.")
    if sid.isna().any() or (sid == "").any():
        raise ValueError("[microdomain_rep]: sample_id contains missing or empty values.")
    return sid.astype(str).to_numpy(copy=False)


def _set_seed(seed: int):
    np.random.seed(seed)
    if TORCH_AVAILABLE:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.use_deterministic_algorithms(False)


def _resolve_device(device_override: str | None):
    if not TORCH_AVAILABLE:
        return "cpu"
    if device_override is None:
        return torch.device("cpu")
    value = str(device_override).strip().lower()
    if value in {"", "cpu"}:
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device(f"cuda:{device_override}")
    logger.warning("[microdomain_rep]: CUDA requested but not available, fallback to CPU.")
    return torch.device("cpu")


def _count_nonzero_per_axis(x, axis: int) -> np.ndarray:
    if issparse(x):
        return np.asarray((x > 0).sum(axis=axis)).reshape(-1)
    return np.sum(np.asarray(x) > 0, axis=axis)


def _sample_safe_filter_masks(
    x_count,
    sample_ids: np.ndarray,
    *,
    min_spots_per_gene: int,
    min_genes_per_spot: int,
    filter_spots: bool,
    filter_genes: bool,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    n_obs, n_vars = x_count.shape
    spot_mask = np.ones(n_obs, dtype=bool)
    gene_mask = np.ones(n_vars, dtype=bool)

    if filter_spots:
        gene_per_spot = _count_nonzero_per_axis(x_count, axis=1)
        spot_mask = gene_per_spot >= int(min_genes_per_spot)
    else:
        gene_per_spot = _count_nonzero_per_axis(x_count, axis=1)

    if filter_genes:
        gene_mask = np.zeros(n_vars, dtype=bool)
        for sid in pd.Index(sample_ids).unique():
            idx = np.where(sample_ids == sid)[0]
            if idx.size == 0:
                continue
            sub = x_count[idx]
            gene_counts = _count_nonzero_per_axis(sub, axis=0)
            gene_mask |= gene_counts >= int(min_spots_per_gene)
    else:
        gene_mask = np.ones(n_vars, dtype=bool)

    summary = {
        "n_obs_before": int(n_obs),
        "n_vars_before": int(n_vars),
        "n_obs_after": int(np.sum(spot_mask)),
        "n_vars_after": int(np.sum(gene_mask)),
        "filter_spots": bool(filter_spots),
        "filter_genes": bool(filter_genes),
        "min_genes_per_spot": int(min_genes_per_spot),
        "min_spots_per_gene": int(min_spots_per_gene),
        "spot_nonzero_gene_stats": _summary_stats(gene_per_spot.astype(np.float32)),
    }
    return spot_mask, gene_mask, summary


def _run_hvg(
    tmp: AnnData,
    *,
    use_hvg: bool,
    n_top_genes: int,
    hvg_flavor: str,
) -> np.ndarray:
    if not use_hvg:
        return np.ones(tmp.n_vars, dtype=bool)
    n_top_eff = min(int(n_top_genes), int(tmp.n_vars))
    if n_top_eff <= 0:
        raise RuntimeError(
            "[microdomain_rep]: Scanpy HVG requires at least one input gene and a positive n_top_genes."
        )
    flavor = str(hvg_flavor).strip()
    hvg_kwargs: dict[str, Any] = {}
    if flavor == "seurat_v3":
        if "counts" not in tmp.layers:
            raise RuntimeError(
                "[microdomain_rep]: Scanpy seurat_v3 HVG requires tmp.layers['counts']."
            )
        hvg_kwargs["layer"] = "counts"
    try:
        _scanpy().pp.highly_variable_genes(
            tmp,
            n_top_genes=n_top_eff,
            flavor=flavor,
            subset=False,
            **hvg_kwargs,
        )
    except Exception as exc:
        raise RuntimeError(
            "[microdomain_rep]: Scanpy HVG failed "
            f"(flavor={flavor!r}, n_top_genes={n_top_eff})."
        ) from exc
    if "highly_variable" not in tmp.var.columns:
        raise RuntimeError("[microdomain_rep]: Scanpy HVG did not create var['highly_variable'].")
    mask = tmp.var["highly_variable"].to_numpy().astype(bool)
    if mask.shape != (int(tmp.n_vars),):
        raise RuntimeError(
            "[microdomain_rep]: Scanpy HVG mask length mismatch "
            f"(observed={mask.shape[0]}, expected={tmp.n_vars})."
        )
    if int(mask.sum()) <= 0:
        raise RuntimeError("[microdomain_rep]: Scanpy HVG selected zero genes.")
    return mask


def _preselect_genes_sample_safe(
    x_count,
    sample_ids: np.ndarray,
    *,
    normalize_target_sum: float,
    use_hvg: bool,
    n_top_genes: int,
    hvg_flavor: str,
) -> tuple[np.ndarray, dict[str, Any]]:
    n_vars = int(x_count.shape[1])
    if not use_hvg:
        return np.ones(n_vars, dtype=bool), {
            "use_hvg": False,
            "n_genes_selected": n_vars,
            "selection_rule": "all_genes_after_filter",
            "requested_hvg_flavor": str(hvg_flavor),
            "actual_hvg_method": "disabled_by_user",
            "requested_n_top_genes": int(n_top_genes),
            "hvg_fallback_used": False,
        }

    votes = np.zeros(n_vars, dtype=np.int32)
    per_sample: list[dict[str, Any]] = []
    for sid in pd.Index(sample_ids).unique():
        idx = np.where(sample_ids == sid)[0]
        if idx.size == 0:
            continue
        tmp = AnnData(X=x_count[idx].copy())
        if issparse(tmp.X):
            tmp.layers["counts"] = tmp.X.tocsr().copy()
        else:
            tmp.layers["counts"] = np.asarray(tmp.X, dtype=np.float32).copy()
        _scanpy().pp.normalize_total(tmp, target_sum=float(normalize_target_sum))
        _scanpy().pp.log1p(tmp)
        hvg_mask = _run_hvg(
            tmp,
            use_hvg=use_hvg,
            n_top_genes=n_top_genes,
            hvg_flavor=hvg_flavor,
        )
        votes += hvg_mask.astype(np.int32)
        per_sample.append(
            {
                "sample_id": str(sid),
                "n_spots": int(idx.size),
                "n_hvg": int(np.sum(hvg_mask)),
            }
        )

    gene_mask = votes > 0
    if int(np.sum(gene_mask)) == 0:
        raise RuntimeError("[microdomain_rep]: sample-safe Scanpy HVG union is empty.")
    return gene_mask, {
        "use_hvg": True,
        "n_genes_selected": int(np.sum(gene_mask)),
        "selection_rule": "sample_safe_hvg_union",
        "requested_hvg_flavor": str(hvg_flavor),
        "actual_hvg_method": f"scanpy.pp.highly_variable_genes:{str(hvg_flavor).strip()}",
        "requested_n_top_genes": int(n_top_genes),
        "hvg_fallback_used": False,
        "per_sample": _records_to_columnar(per_sample),
        "vote_stats": _summary_stats(votes.astype(np.float32)),
    }


@dataclass
class SampleProjection:
    sample_id: str
    kind: str
    used_dim: int
    pca_components: np.ndarray
    pca_mean: np.ndarray
    gene_mean: np.ndarray
    gene_std: np.ndarray
    explained_variance_ratio_sum: float


def _samplewise_log_scale_pca(
    x_count,
    sample_ids: np.ndarray,
    *,
    normalize_target_sum: float,
    pca_dim: int,
    scale_max_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, SampleProjection], dict[str, Any]]:
    n_obs, n_vars = x_count.shape
    x_log = np.zeros((n_obs, n_vars), dtype=np.float32)
    x_pca = np.zeros((n_obs, int(pca_dim)), dtype=np.float32)
    library_size = np.zeros(n_obs, dtype=np.float32)
    projections: dict[str, SampleProjection] = {}
    per_sample: list[dict[str, Any]] = []

    for sample_offset, sid in enumerate(pd.Index(sample_ids).unique().tolist()):
        idx = np.where(sample_ids == sid)[0]
        sub = x_count[idx]
        tmp = AnnData(X=sub.copy())
        if issparse(tmp.X):
            tmp.layers["counts"] = tmp.X.tocsr().copy()
        else:
            tmp.layers["counts"] = np.asarray(tmp.X, dtype=np.float32).copy()
        library_size[idx] = np.asarray(tmp.layers["counts"].sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
        library_size[idx][library_size[idx] <= 0] = 1.0

        _scanpy().pp.normalize_total(tmp, target_sum=float(normalize_target_sum))
        _scanpy().pp.log1p(tmp)
        sub_log = _to_dense_float32(tmp.X)
        x_log[idx] = sub_log

        gene_mean = np.mean(sub_log, axis=0).astype(np.float32, copy=False)
        gene_std = np.std(sub_log, axis=0).astype(np.float32, copy=False)
        gene_std = np.where(gene_std > 1e-6, gene_std, 1.0).astype(np.float32, copy=False)

        _scanpy().pp.scale(tmp, zero_center=True, max_value=float(scale_max_value))
        sub_scaled = _to_dense_float32(tmp.X)

        if sub_scaled.shape[0] <= 1 or sub_scaled.shape[1] <= 1:
            used_dim = int(min(int(pca_dim), sub_scaled.shape[1]))
            if used_dim > 0:
                x_pca[idx, :used_dim] = sub_scaled[:, :used_dim]
                components = np.zeros((used_dim, n_vars), dtype=np.float32)
                components[:, :used_dim] = np.eye(used_dim, dtype=np.float32)
            else:
                components = np.zeros((0, n_vars), dtype=np.float32)
            projections[str(sid)] = SampleProjection(
                sample_id=str(sid),
                kind="identity",
                used_dim=int(used_dim),
                pca_components=components,
                pca_mean=np.zeros((n_vars,), dtype=np.float32),
                gene_mean=gene_mean,
                gene_std=gene_std,
                explained_variance_ratio_sum=0.0,
            )
            per_sample.append(
                {
                    "sample_id": str(sid),
                    "n_spots": int(idx.size),
                    "used_pca_dim": int(used_dim),
                    "projection_kind": "identity",
                    "explained_variance_ratio_sum": 0.0,
                }
            )
            continue

        used_dim = int(min(int(pca_dim), sub_scaled.shape[0] - 1, sub_scaled.shape[1]))
        if used_dim <= 0:
            used_dim = int(min(int(pca_dim), sub_scaled.shape[1]))
            x_pca[idx, :used_dim] = sub_scaled[:, :used_dim]
            components = np.zeros((used_dim, n_vars), dtype=np.float32)
            components[:, :used_dim] = np.eye(used_dim, dtype=np.float32)
            projections[str(sid)] = SampleProjection(
                sample_id=str(sid),
                kind="identity",
                used_dim=int(used_dim),
                pca_components=components,
                pca_mean=np.zeros((n_vars,), dtype=np.float32),
                gene_mean=gene_mean,
                gene_std=gene_std,
                explained_variance_ratio_sum=0.0,
            )
            per_sample.append(
                {
                    "sample_id": str(sid),
                    "n_spots": int(idx.size),
                    "used_pca_dim": int(used_dim),
                    "projection_kind": "identity",
                    "explained_variance_ratio_sum": 0.0,
                }
            )
            continue

        pca = _pca_cls()(n_components=used_dim, random_state=int(seed + sample_offset), svd_solver="randomized")
        sub_pca = pca.fit_transform(sub_scaled).astype(np.float32, copy=False)
        x_pca[idx, :used_dim] = sub_pca
        projections[str(sid)] = SampleProjection(
            sample_id=str(sid),
            kind="pca",
            used_dim=int(used_dim),
            pca_components=np.asarray(pca.components_, dtype=np.float32),
            pca_mean=np.asarray(pca.mean_, dtype=np.float32),
            gene_mean=gene_mean,
            gene_std=gene_std,
            explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
        )
        per_sample.append(
            {
                "sample_id": str(sid),
                "n_spots": int(idx.size),
                "used_pca_dim": int(used_dim),
                "projection_kind": "pca",
                "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
            }
        )

    return x_log, x_pca, library_size, projections, {"per_sample": _records_to_columnar(per_sample)}


def _multisample_log_scale_pca_harmony(
    x_count,
    graph_sample_ids: np.ndarray,
    *,
    harmony_batch_ids: np.ndarray,
    normalize_target_sum: float,
    pca_dim: int,
    scale_max_value: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, SampleProjection], dict[str, Any]]:
    n_obs, n_vars = x_count.shape
    normalized_graph_sample_ids = _normalize_sample_ids(graph_sample_ids, n_obs)
    normalized_harmony_batch_ids = _normalize_sample_ids(harmony_batch_ids, n_obs)
    unique_graph_sample_ids = pd.Index(normalized_graph_sample_ids).unique().tolist()
    unique_harmony_batch_ids = pd.Index(normalized_harmony_batch_ids).unique().tolist()
    if len(unique_graph_sample_ids) < 2:
        raise ValueError("[rep_utils]: multi-sample Harmony preprocessing requires at least two samples.")
    if len(unique_harmony_batch_ids) < 2:
        raise ValueError("[rep_utils]: multi-sample Harmony preprocessing requires at least two Harmony batches.")
    if int(pca_dim) < 1:
        raise ValueError("[rep_utils]: pca_dim must be at least 1 for multi-sample preprocessing.")

    tmp = AnnData(X=x_count.copy())
    if issparse(tmp.X):
        tmp.layers["counts"] = tmp.X.tocsr().copy()
    else:
        tmp.layers["counts"] = np.asarray(tmp.X, dtype=np.float32).copy()
    library_size = np.asarray(tmp.layers["counts"].sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    library_size[library_size <= 0] = 1.0

    _scanpy().pp.normalize_total(tmp, target_sum=float(normalize_target_sum))
    _scanpy().pp.log1p(tmp)
    x_log = _to_dense_float32(tmp.X)
    gene_mean = np.mean(x_log, axis=0).astype(np.float32, copy=False)
    gene_std = np.std(x_log, axis=0).astype(np.float32, copy=False)
    gene_std = np.where(gene_std > 1e-6, gene_std, 1.0).astype(np.float32, copy=False)

    _scanpy().pp.scale(tmp, zero_center=True, max_value=float(scale_max_value))
    x_scaled = _to_dense_float32(tmp.X)
    used_pca_dim = int(min(int(pca_dim), int(n_obs) - 1, int(n_vars)))
    if used_pca_dim <= 0:
        raise RuntimeError(
            "[rep_utils]: shared PCA requires at least two retained spots and one retained gene."
        )
    pca = _pca_cls()(
        n_components=used_pca_dim,
        random_state=int(seed),
        svd_solver="randomized",
    )
    x_pca_uncorrected = pca.fit_transform(x_scaled).astype(np.float32, copy=False)
    if x_pca_uncorrected.shape != (int(n_obs), used_pca_dim) or not np.all(np.isfinite(x_pca_uncorrected)):
        raise RuntimeError("[rep_utils]: shared PCA returned an invalid or non-finite matrix.")

    try:
        import harmonypy
    except ImportError as exc:  # pragma: no cover - depends on runtime environment
        raise ImportError(
            "[rep_utils]: harmonypy is required for multi-sample REP preprocessing."
        ) from exc

    harmony_batch_key = "harmony_batch"
    harmony_metadata = pd.DataFrame({harmony_batch_key: normalized_harmony_batch_ids})
    harmony_kwargs: dict[str, Any] = {}
    try:
        import inspect

        if "random_state" in inspect.signature(harmonypy.run_harmony).parameters:
            harmony_kwargs["random_state"] = int(seed)
    except (TypeError, ValueError):
        pass
    harmony_result = harmonypy.run_harmony(
        x_pca_uncorrected,
        harmony_metadata,
        harmony_batch_key,
        **harmony_kwargs,
    )
    harmony_raw = np.asarray(harmony_result.Z_corr, dtype=np.float32)
    expected_shape = (int(n_obs), used_pca_dim)
    transposed_shape = (used_pca_dim, int(n_obs))
    if harmony_raw.shape == expected_shape == transposed_shape:
        harmony_original = np.asarray(getattr(harmony_result, "Z_orig", None), dtype=np.float32)
        if harmony_original.shape != expected_shape:
            raise RuntimeError("[rep_utils]: Harmony output orientation is ambiguous for square PCA input.")
        direct_error = float(np.linalg.norm(harmony_original - x_pca_uncorrected))
        transposed_error = float(np.linalg.norm(harmony_original.T - x_pca_uncorrected))
        if direct_error < transposed_error:
            x_pca = harmony_raw
        elif transposed_error < direct_error:
            x_pca = harmony_raw.T
        else:
            raise RuntimeError("[rep_utils]: Harmony output orientation is ambiguous for square PCA input.")
    elif harmony_raw.shape == transposed_shape:
        x_pca = harmony_raw.T
    elif harmony_raw.shape == expected_shape:
        x_pca = harmony_raw
    else:
        raise RuntimeError(
            "[rep_utils]: Harmony output shape is incompatible with retained REP spots and shared PCA dimension; "
            f"observed={harmony_raw.shape}, expected={expected_shape}."
        )
    x_pca = np.asarray(x_pca, dtype=np.float32)
    if x_pca.shape != expected_shape or not np.all(np.isfinite(x_pca)):
        raise RuntimeError("[rep_utils]: Harmony-corrected shared PCA matrix is invalid or non-finite.")

    shared_projection = SampleProjection(
        sample_id="multi_sample_shared",
        kind="pca",
        used_dim=used_pca_dim,
        pca_components=np.asarray(pca.components_, dtype=np.float32),
        pca_mean=np.asarray(pca.mean_, dtype=np.float32),
        gene_mean=gene_mean,
        gene_std=gene_std,
        explained_variance_ratio_sum=float(np.sum(pca.explained_variance_ratio_)),
    )
    projections = {str(sample_id): shared_projection for sample_id in unique_graph_sample_ids}
    pca_summary = {
        "preprocessing_scope": "multi_sample_shared",
        "pca_basis": "shared",
        "integration_method": "Harmony",
        "integration_batch_key": harmony_batch_key,
        "harmony_applied": True,
        "harmony_seed_applied": "random_state" in harmony_kwargs,
        "n_samples": int(len(unique_graph_sample_ids)),
        "n_harmony_batches": int(len(unique_harmony_batch_ids)),
        "n_obs": int(n_obs),
        "n_vars": int(n_vars),
        "requested_pca_dim": int(pca_dim),
        "used_pca_dim": used_pca_dim,
        "projection_kind": "shared_pca",
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
    }
    return x_log, x_pca, library_size, projections, pca_summary


def _build_knn_graph(
    x: np.ndarray,
    *,
    k: int,
    radius: float | None,
    include_self: bool,
) -> csr_matrix:
    n_obs = int(x.shape[0])
    if n_obs <= 1:
        return csr_matrix((n_obs, n_obs), dtype=np.float32)
    if radius is not None and radius > 0:
        adj = _radius_neighbors_graph(
            x,
            radius=float(radius),
            mode="connectivity",
            metric="euclidean",
            include_self=include_self,
        )
    else:
        k_eff = min(max(int(k), 1), n_obs - (1 if not include_self else 0))
        if k_eff <= 0:
            return csr_matrix((n_obs, n_obs), dtype=np.float32)
        adj = _kneighbors_graph(
            x,
            n_neighbors=int(k_eff),
            mode="connectivity",
            metric="euclidean",
            include_self=include_self,
        )
    out = adj.tocsr().astype(np.float32)
    out.eliminate_zeros()
    return out


def build_samplewise_spatial_graph(
    coord: np.ndarray,
    sample_ids: np.ndarray,
    *,
    k: int,
    radius: float | None,
    include_self: bool = False,
    sym: bool = True,
) -> tuple[csr_matrix, dict[str, Any]]:
    n_obs = int(coord.shape[0])
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []
    for sid in pd.Index(sample_ids).unique().tolist():
        idx = np.where(sample_ids == sid)[0]
        sub_adj = _build_knn_graph(
            coord[idx],
            k=k,
            radius=radius,
            include_self=include_self,
        ).tocoo()
        if sub_adj.nnz > 0:
            rows.append(idx[sub_adj.row].astype(np.int64, copy=False))
            cols.append(idx[sub_adj.col].astype(np.int64, copy=False))
            vals.append(sub_adj.data.astype(np.float32, copy=False))
        per_sample.append(
            {
                "sample_id": str(sid),
                "n_spots": int(idx.size),
                "n_edges_directed": int(sub_adj.nnz),
            }
        )
    if len(rows) == 0:
        adj = csr_matrix((n_obs, n_obs), dtype=np.float32)
    else:
        adj = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_obs, n_obs),
            dtype=np.float32,
        )
    if sym:
        adj = adj.maximum(adj.T).tocsr().astype(np.float32)
    if not include_self:
        adj.setdiag(0.0)
        adj.eliminate_zeros()
    cross_stats = count_cross_sample_edges(adj, sample_ids)
    summary = {
        "graph_type": "samplewise_spatial_knn" if radius is None else "samplewise_spatial_radius",
        "k": int(k),
        "radius": None if radius is None else float(radius),
        "nnz": int(adj.nnz),
        "per_sample": _records_to_columnar(per_sample),
        "sample_safety": cross_stats,
    }
    return adj, summary


def count_cross_sample_edges(adj: csr_matrix, sample_ids: np.ndarray) -> dict[str, Any]:
    coo = adj.tocoo()
    if coo.nnz == 0:
        return {
            "edge_total": 0,
            "cross_sample_edges": 0,
            "cross_sample_ratio": 0.0,
            "sample_safety_checked": True,
        }
    cross = int(np.sum(sample_ids[coo.row] != sample_ids[coo.col]))
    return {
        "edge_total": int(coo.nnz),
        "cross_sample_edges": cross,
        "cross_sample_ratio": float(cross / max(int(coo.nnz), 1)),
        "sample_safety_checked": True,
    }


def _row_normalize_csr(adj: csr_matrix, eps: float = 1e-8) -> csr_matrix:
    row_sum = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    keep = row_sum > eps
    inv[keep] = 1.0 / row_sum[keep]
    return (diags(inv) @ adj).tocsr().astype(np.float32)


def _csr_to_torch_sparse(adj: csr_matrix, device: torch.device) -> torch.Tensor:
    coo = adj.tocoo()
    indices = torch.tensor(np.vstack([coo.row, coo.col]), dtype=torch.long, device=device)
    values = torch.tensor(coo.data, dtype=torch.float32, device=device)
    return torch.sparse_coo_tensor(indices, values, size=adj.shape, device=device).coalesce()


def _edge_arrays(adj: csr_matrix) -> tuple[np.ndarray, np.ndarray]:
    coo = adj.tocoo()
    return coo.row.astype(np.int64, copy=False), coo.col.astype(np.int64, copy=False)


def _cosine_similarity_rows(a: np.ndarray, b: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    a_norm = np.linalg.norm(a, axis=1, keepdims=True)
    b_norm = np.linalg.norm(b, axis=1, keepdims=True)
    a_norm[a_norm < eps] = 1.0
    b_norm[b_norm < eps] = 1.0
    sim = np.sum((a / a_norm) * (b / b_norm), axis=1)
    return np.clip(sim, -1.0, 1.0).astype(np.float32, copy=False)


def _sample_expression_edge_weight(x_pca: np.ndarray, row: np.ndarray, col: np.ndarray) -> np.ndarray:
    if row.size == 0:
        return np.zeros((0,), dtype=np.float32)
    sim = _cosine_similarity_rows(x_pca[row], x_pca[col])
    return np.clip((sim + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)


def _sample_negative_edges(
    spatial_adj: csr_matrix,
    sample_ids: np.ndarray,
    *,
    negatives_per_node: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    adj = spatial_adj.tocsr()
    rows: list[int] = []
    cols: list[int] = []
    for sid in pd.Index(sample_ids).unique().tolist():
        idx = np.where(sample_ids == sid)[0]
        if idx.size <= 1:
            continue
        local_index = {int(node): pos for pos, node in enumerate(idx.tolist())}
        for node in idx.tolist():
            start, end = adj.indptr[node], adj.indptr[node + 1]
            neigh = set(int(v) for v in adj.indices[start:end].tolist())
            neigh.add(int(node))
            candidates = [int(v) for v in idx.tolist() if int(v) not in neigh]
            if len(candidates) == 0:
                continue
            n_take = min(int(max(1, negatives_per_node)), len(candidates))
            choice = rng.choice(np.asarray(candidates, dtype=np.int64), size=n_take, replace=False)
            rows.extend([int(node)] * n_take)
            cols.extend(choice.astype(np.int64, copy=False).tolist())
    if len(rows) == 0:
        return np.zeros((0,), dtype=np.int64), np.zeros((0,), dtype=np.int64)
    return np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int64)


# Torch loss helpers shared by REP implementations.
def _variance_regularization(z: torch.Tensor, target_std: float = 1.0, eps: float = 1e-4) -> torch.Tensor:
    if z.numel() == 0:
        return z.new_tensor(0.0)
    centered = z - z.mean(dim=0, keepdim=True)
    std = torch.sqrt(torch.var(centered, dim=0, unbiased=False) + eps)
    return torch.mean(F.relu(float(target_std) - std))


def _edge_cosine_logits(z: torch.Tensor, row_idx: torch.Tensor, col_idx: torch.Tensor, scale: float = 5.0) -> torch.Tensor:
    if row_idx.numel() == 0:
        return z.new_zeros((0,))
    left = F.normalize(z[row_idx], dim=1)
    right = F.normalize(z[col_idx], dim=1)
    return torch.sum(left * right, dim=1) * float(scale)


def _graph_loss(
    z: torch.Tensor,
    pos_row: torch.Tensor,
    pos_col: torch.Tensor,
    neg_row: torch.Tensor,
    neg_col: torch.Tensor,
    *,
    scale: float,
) -> torch.Tensor:
    pos_logits = _edge_cosine_logits(z, pos_row, pos_col, scale=scale)
    neg_logits = _edge_cosine_logits(z, neg_row, neg_col, scale=scale)
    loss = z.new_tensor(0.0)
    if pos_logits.numel() > 0:
        loss = loss + F.binary_cross_entropy_with_logits(pos_logits, torch.ones_like(pos_logits))
    if neg_logits.numel() > 0:
        loss = loss + F.binary_cross_entropy_with_logits(neg_logits, torch.zeros_like(neg_logits))
    return loss


def _local_consistency_loss(
    z: torch.Tensor,
    row_idx: torch.Tensor,
    col_idx: torch.Tensor,
    edge_weight: torch.Tensor,
) -> torch.Tensor:
    if row_idx.numel() == 0:
        return z.new_tensor(0.0)
    left = F.normalize(z[row_idx], dim=1)
    right = F.normalize(z[col_idx], dim=1)
    cosine = torch.sum(left * right, dim=1)
    return torch.mean(edge_weight * (1.0 - cosine))


def _invert_sample_projection(proj: SampleProjection, x_pca_pred: np.ndarray) -> np.ndarray:
    used_dim = int(proj.used_dim)
    if used_dim <= 0:
        scaled = np.zeros((x_pca_pred.shape[0], proj.gene_mean.shape[0]), dtype=np.float32)
    elif proj.kind == "pca":
        scaled = (
            np.asarray(x_pca_pred[:, :used_dim], dtype=np.float32) @ np.asarray(proj.pca_components, dtype=np.float32)
            + np.asarray(proj.pca_mean, dtype=np.float32)
        )
    else:
        scaled = np.zeros((x_pca_pred.shape[0], proj.gene_mean.shape[0]), dtype=np.float32)
        scaled[:, :used_dim] = np.asarray(x_pca_pred[:, :used_dim], dtype=np.float32)
    x_log = scaled * np.asarray(proj.gene_std, dtype=np.float32) + np.asarray(proj.gene_mean, dtype=np.float32)
    x_log = np.clip(x_log, a_min=0.0, a_max=None)
    return np.expm1(x_log).astype(np.float32, copy=False)


def _build_microdomain_outputs(
    z_microdomain: np.ndarray,
    coord: np.ndarray,
    sample_ids: np.ndarray,
    *,
    microdomain_k: int,
    temperature: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, csr_matrix, dict[str, Any]]:
    del seed
    n_obs = int(z_microdomain.shape[0])
    d = int(max(1, microdomain_k))
    idx_out = np.full((n_obs, d), -1, dtype=np.int64)
    w_out = np.zeros((n_obs, d), dtype=np.float32)
    confidence = np.zeros(n_obs, dtype=np.float32)
    boundary = np.zeros(n_obs, dtype=np.float32)
    spatial_radius = np.zeros(n_obs, dtype=np.float32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []

    z_norm = np.asarray(z_microdomain, dtype=np.float32)
    denom = np.linalg.norm(z_norm, axis=1, keepdims=True)
    denom[denom < 1e-8] = 1.0
    z_norm = z_norm / denom

    for sid in pd.Index(sample_ids).unique().tolist():
        idx = np.where(sample_ids == sid)[0]
        if idx.size <= 1:
            per_sample.append(
                {
                    "sample_id": str(sid),
                    "n_spots": int(idx.size),
                    "mean_confidence": None,
                    "mean_radius": None,
                }
            )
            continue
        n_neighbors = min(int(d + 1), int(idx.size))
        nbrs = _nearest_neighbors_cls()(metric="cosine", n_neighbors=n_neighbors)
        nbrs.fit(z_norm[idx])
        dist, local_ind = nbrs.kneighbors(z_norm[idx], return_distance=True)
        sample_conf: list[float] = []
        sample_radius: list[float] = []
        sample_boundary: list[float] = []
        for local_row, global_row in enumerate(idx.tolist()):
            local_neighbors = local_ind[local_row].astype(np.int64, copy=False)
            local_dist = dist[local_row].astype(np.float32, copy=False)
            keep = local_neighbors != int(local_row)
            local_neighbors = local_neighbors[keep][:d]
            local_dist = local_dist[keep][:d]
            if local_neighbors.size == 0:
                continue
            global_neighbors = idx[local_neighbors].astype(np.int64, copy=False)
            raw_sim = np.clip(1.0 - local_dist, a_min=0.0, a_max=1.0).astype(np.float32, copy=False)
            logits = raw_sim / max(float(temperature), 1e-6)
            logits = logits - float(np.max(logits))
            weight = np.exp(logits).astype(np.float32, copy=False)
            weight_sum = float(np.sum(weight))
            if weight_sum <= 0:
                weight[:] = 1.0 / float(weight.shape[0])
            else:
                weight /= weight_sum

            k_eff = int(global_neighbors.shape[0])
            idx_out[global_row, :k_eff] = global_neighbors
            w_out[global_row, :k_eff] = weight
            confidence_value = float(np.sum(weight * raw_sim))
            confidence[global_row] = confidence_value
            entropy = float(-np.sum(weight * np.log(np.clip(weight, 1e-8, None))) / np.log(max(k_eff, 2)))
            boundary[global_row] = float(np.clip(0.5 * entropy + 0.5 * (1.0 - confidence_value), 0.0, 1.0))
            radius_value = float(
                np.sum(
                    weight
                    * np.linalg.norm(
                        np.asarray(coord[global_neighbors], dtype=np.float32) - np.asarray(coord[global_row], dtype=np.float32),
                        axis=1,
                    )
                )
            )
            spatial_radius[global_row] = radius_value
            sample_conf.append(confidence_value)
            sample_radius.append(radius_value)
            sample_boundary.append(boundary[global_row])
            rows.append(np.full(k_eff, int(global_row), dtype=np.int64))
            cols.append(global_neighbors)
            vals.append(weight.astype(np.float32, copy=False))
        per_sample.append(
            {
                "sample_id": str(sid),
                "n_spots": int(idx.size),
                "mean_confidence": float(np.mean(sample_conf)) if len(sample_conf) > 0 else None,
                "mean_radius": float(np.mean(sample_radius)) if len(sample_radius) > 0 else None,
                "mean_boundary_uncertainty": float(np.mean(sample_boundary)) if len(sample_boundary) > 0 else None,
            }
        )

    if len(rows) == 0:
        graph = csr_matrix((n_obs, n_obs), dtype=np.float32)
    else:
        graph = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_obs, n_obs),
            dtype=np.float32,
        )
    graph.eliminate_zeros()
    return idx_out, w_out, confidence, boundary, spatial_radius, graph, {"per_sample": _records_to_columnar(per_sample)}


def _build_spatial_constrained_microdomain_outputs(
    z_microdomain: np.ndarray,
    coord: np.ndarray,
    sample_ids: np.ndarray,
    *,
    spatial_candidate_k: int,
    microdomain_k: int,
    temperature: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, csr_matrix, dict[str, Any]]:
    n_obs = int(z_microdomain.shape[0])
    d = int(max(1, microdomain_k))
    candidate_k = int(max(d, spatial_candidate_k))
    idx_out = np.full((n_obs, d), -1, dtype=np.int64)
    w_out = np.zeros((n_obs, d), dtype=np.float32)
    confidence = np.zeros(n_obs, dtype=np.float32)
    boundary = np.zeros(n_obs, dtype=np.float32)
    spatial_radius = np.zeros(n_obs, dtype=np.float32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    per_sample: list[dict[str, Any]] = []

    z_norm = np.asarray(z_microdomain, dtype=np.float32)
    z_norm_denom = np.linalg.norm(z_norm, axis=1, keepdims=True)
    z_norm_denom[z_norm_denom < 1e-8] = 1.0
    z_norm = z_norm / z_norm_denom

    for sid in pd.Index(sample_ids).unique().tolist():
        idx = np.where(sample_ids == sid)[0]
        if idx.size <= 1:
            per_sample.append(
                {
                    "sample_id": str(sid),
                    "n_spots": int(idx.size),
                    "mean_confidence": None,
                    "mean_radius": None,
                    "mean_boundary_uncertainty": None,
                }
            )
            continue

        n_candidates = min(int(candidate_k + 1), int(idx.size))
        spatial_nbrs = _nearest_neighbors_cls()(metric="euclidean", n_neighbors=n_candidates)
        spatial_nbrs.fit(np.asarray(coord[idx], dtype=np.float32))
        spatial_dist, spatial_ind = spatial_nbrs.kneighbors(np.asarray(coord[idx], dtype=np.float32), return_distance=True)

        sample_conf: list[float] = []
        sample_radius: list[float] = []
        sample_boundary: list[float] = []
        for local_row, global_row in enumerate(idx.tolist()):
            candidate_local = spatial_ind[local_row].astype(np.int64, copy=False)
            candidate_spatial_dist = spatial_dist[local_row].astype(np.float32, copy=False)
            keep = candidate_local != int(local_row)
            candidate_local = candidate_local[keep][:candidate_k]
            candidate_spatial_dist = candidate_spatial_dist[keep][:candidate_k]
            if candidate_local.size == 0:
                continue

            focal = z_norm[global_row : global_row + 1]
            candidate_global = idx[candidate_local].astype(np.int64, copy=False)
            sim = _cosine_similarity_rows(
                np.repeat(focal, repeats=candidate_global.shape[0], axis=0),
                z_norm[candidate_global],
            )
            order = np.argsort(-sim)[:d]
            global_neighbors = candidate_global[order]
            raw_sim = np.clip(sim[order], a_min=0.0, a_max=1.0).astype(np.float32, copy=False)
            if global_neighbors.size == 0:
                continue

            logits = raw_sim / max(float(temperature), 1e-6)
            logits = logits - float(np.max(logits))
            weight = np.exp(logits).astype(np.float32, copy=False)
            weight_sum = float(np.sum(weight))
            if weight_sum <= 0:
                weight[:] = 1.0 / float(weight.shape[0])
            else:
                weight /= weight_sum

            k_eff = int(global_neighbors.shape[0])
            idx_out[global_row, :k_eff] = global_neighbors
            w_out[global_row, :k_eff] = weight
            confidence_value = float(np.sum(weight * raw_sim))
            confidence[global_row] = confidence_value
            entropy = float(-np.sum(weight * np.log(np.clip(weight, 1e-8, None))) / np.log(max(k_eff, 2)))
            boundary[global_row] = float(np.clip(0.5 * entropy + 0.5 * (1.0 - confidence_value), 0.0, 1.0))
            radius_value = float(
                np.sum(
                    weight
                    * np.linalg.norm(
                        np.asarray(coord[global_neighbors], dtype=np.float32) - np.asarray(coord[global_row], dtype=np.float32),
                        axis=1,
                    )
                )
            )
            spatial_radius[global_row] = radius_value
            sample_conf.append(confidence_value)
            sample_radius.append(radius_value)
            sample_boundary.append(boundary[global_row])
            rows.append(np.full(k_eff, int(global_row), dtype=np.int64))
            cols.append(global_neighbors)
            vals.append(weight.astype(np.float32, copy=False))

        per_sample.append(
            {
                "sample_id": str(sid),
                "n_spots": int(idx.size),
                "mean_confidence": float(np.mean(sample_conf)) if len(sample_conf) > 0 else None,
                "mean_radius": float(np.mean(sample_radius)) if len(sample_radius) > 0 else None,
                "mean_boundary_uncertainty": float(np.mean(sample_boundary)) if len(sample_boundary) > 0 else None,
            }
        )

    if len(rows) == 0:
        graph = csr_matrix((n_obs, n_obs), dtype=np.float32)
    else:
        graph = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_obs, n_obs),
            dtype=np.float32,
        )
    graph.eliminate_zeros()
    return idx_out, w_out, confidence, boundary, spatial_radius, graph, {"per_sample": _records_to_columnar(per_sample)}


def _materialize_denoised_expression(
    x_pca_recon: np.ndarray,
    sample_ids: np.ndarray,
    projections: dict[str, SampleProjection],
) -> np.ndarray:
    n_obs = int(x_pca_recon.shape[0])
    n_vars = int(next(iter(projections.values())).gene_mean.shape[0]) if len(projections) > 0 else 0
    x_hat = np.zeros((n_obs, n_vars), dtype=np.float32)
    for sid in pd.Index(sample_ids).unique().tolist():
        idx = np.where(sample_ids == sid)[0]
        proj = projections[str(sid)]
        x_hat[idx] = _invert_sample_projection(proj, x_pca_recon[idx])
    return x_hat
