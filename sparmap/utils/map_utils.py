from __future__ import annotations

import logging
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, diags, issparse
from scipy.stats import rankdata

from sparmap.utils.utils import (
    bh_qvalues as _bh_qvalues,
    summary_stats as _summary_stats,
)

from sparmap._logging import get_logger

logger = get_logger(__name__)


def restrict_graph_by_sample(adj, sample_ids: np.ndarray) -> csr_matrix:
    sid = pd.Index(np.asarray(sample_ids).reshape(-1)).astype(str).to_numpy(copy=False)
    try:
        if not issparse(adj):
            adj = csr_matrix(np.asarray(adj, dtype=np.float32))
        adj = adj.tocsr().astype(np.float32)
    except Exception as exc:
        raise RuntimeError("[map_utils]: formal microdomain graph cannot be converted to CSR float32.") from exc
    if adj.shape != (int(sid.shape[0]), int(sid.shape[0])):
        raise RuntimeError(
            "[map_utils]: sample_ids length does not match the formal graph shape "
            f"(n_sample_ids={sid.shape[0]}, graph_shape={adj.shape})."
        )
    coo = adj.tocoo()
    cross_sample = sid[coo.row] != sid[coo.col]
    if bool(np.any(cross_sample)):
        raise RuntimeError(
            "[map_utils]: formal microdomain graph contains cross-sample edges "
            f"(n_cross_sample_edges={int(np.sum(cross_sample))})."
        )
    out = adj.tocsr().astype(np.float32)
    out.eliminate_zeros()
    return out

def _add_warning(warnings: list[str], message: str):
    warnings.append(str(message))
    logger.warning(f"[map_utils.py]: {message}")

def _prepare_st_log_expression(
    adata: AnnData,
    common_pos: np.ndarray,
    *,
    count_layer: str | None,
    target_sum: float,
) -> tuple[np.ndarray, str, np.ndarray, np.ndarray, np.ndarray]:
    if count_layer is not None and count_layer in adata.layers:
        x_source = adata.layers[count_layer]
        source_name = f"layers[{count_layer}]"
    elif "counts" in adata.layers:
        x_source = adata.layers["counts"]
        source_name = "layers[counts]"
    else:
        x_source = adata.X
        source_name = "X"

    x_sub = x_source[:, common_pos]
    if issparse(x_sub):
        x_raw = x_sub.tocsr().astype(np.float32)
        lib = np.asarray(x_raw.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
        lib[lib <= 0] = 1.0
        scale = (float(target_sum) / lib).astype(np.float32, copy=False)
        x_norm = x_raw.multiply(scale[:, None])
        x_log = np.log1p(x_norm.toarray()).astype(np.float32, copy=False)
        detect_rate = np.asarray((x_raw > 0).mean(axis=0)).reshape(-1).astype(np.float32, copy=False)
    else:
        x_arr = np.asarray(x_sub, dtype=np.float32)
        lib = np.sum(x_arr, axis=1, dtype=np.float32)
        lib[lib <= 0] = 1.0
        scale = (float(target_sum) / lib).astype(np.float32, copy=False)
        x_log = np.log1p(np.clip(x_arr * scale[:, None], 0.0, None)).astype(np.float32, copy=False)
        detect_rate = np.mean(x_arr > 0, axis=0).astype(np.float32, copy=False)
    x_log[~np.isfinite(x_log)] = 0.0
    mean_expr = np.mean(x_log, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    var_expr = np.var(x_log, axis=0, dtype=np.float64).astype(np.float32, copy=False)
    return x_log, source_name, mean_expr, detect_rate, var_expr


def _prepare_st_log_expression_large(
    adata: AnnData,
    row_indices: np.ndarray,
    common_pos: np.ndarray,
    *,
    count_layer: str | None,
    target_sum: float,
    gene_chunk_size: int,
    row_chunk_size: int,
) -> tuple[
    str,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    if count_layer is not None and count_layer in adata.layers:
        x_source = adata.layers[count_layer]
        source_name = f"layers[{count_layer}]"
    elif "counts" in adata.layers:
        x_source = adata.layers["counts"]
        source_name = "layers[counts]"
    else:
        x_source = adata.X
        source_name = "X"

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    positions = np.asarray(common_pos, dtype=np.int64).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= int(adata.n_obs)):
        raise IndexError("[map_utils]: large expression row index is out of bounds.")
    if np.any(positions < 0) or np.any(positions >= int(adata.n_vars)):
        raise IndexError("[map_utils]: large expression gene index is out of bounds.")

    n_rows = int(rows.size)
    n_genes = int(positions.size)
    library_size = np.zeros((n_rows,), dtype=np.float32)
    row_chunk = max(int(row_chunk_size), 1)
    for start in range(0, n_rows, row_chunk):
        end = min(start + row_chunk, n_rows)
        chunk_rows = rows[start:end]
        if issparse(x_source):
            x_raw = x_source[chunk_rows, :][:, positions].tocsr().astype(np.float32)
            library_size[start:end] = np.asarray(x_raw.sum(axis=1)).reshape(-1).astype(
                np.float32,
                copy=False,
            )
        else:
            x_arr = np.asarray(x_source[np.ix_(chunk_rows, positions)], dtype=np.float32)
            library_size[start:end] = np.sum(x_arr, axis=1, dtype=np.float32)
    library_size[library_size <= 0] = 1.0
    scale = (float(target_sum) / library_size).astype(np.float32, copy=False)

    mean_expr = np.zeros((n_genes,), dtype=np.float32)
    detect_rate = np.zeros((n_genes,), dtype=np.float32)
    var_expr = np.zeros((n_genes,), dtype=np.float32)
    gene_chunk = max(int(gene_chunk_size), 1)
    max_expression_chunk_shape = (0, 0)
    for start in range(0, n_genes, gene_chunk):
        end = min(start + gene_chunk, n_genes)
        chunk_positions = positions[start:end]
        if issparse(x_source):
            x_raw = x_source[:, chunk_positions][rows, :].tocsr().astype(np.float32)
            x_norm = x_raw.multiply(scale[:, None])
            x_log = np.log1p(x_norm.toarray()).astype(np.float32, copy=False)
            detect_chunk = np.asarray((x_raw > 0).mean(axis=0)).reshape(-1).astype(
                np.float32,
                copy=False,
            )
        else:
            x_arr = np.asarray(x_source[np.ix_(rows, chunk_positions)], dtype=np.float32)
            x_log = np.log1p(np.clip(x_arr * scale[:, None], 0.0, None)).astype(
                np.float32,
                copy=False,
            )
            detect_chunk = np.mean(x_arr > 0, axis=0).astype(np.float32, copy=False)
        x_log[~np.isfinite(x_log)] = 0.0
        mean_expr[start:end] = np.mean(x_log, axis=0, dtype=np.float64).astype(
            np.float32,
            copy=False,
        )
        detect_rate[start:end] = detect_chunk
        var_expr[start:end] = np.var(x_log, axis=0, dtype=np.float64).astype(
            np.float32,
            copy=False,
        )
        if int(x_log.shape[0]) * int(x_log.shape[1]) > (
            int(max_expression_chunk_shape[0]) * int(max_expression_chunk_shape[1])
        ):
            max_expression_chunk_shape = (int(x_log.shape[0]), int(x_log.shape[1]))
        del x_log

    logger.info(
        "[map_utils.py]: large expression stats n_spots=%d n_full_genes=%d "
        "gene_chunk_size=%d row_chunk_size=%d max_expression_chunk_shape=%s "
        "full_x_log_materialized=False",
        n_rows,
        n_genes,
        gene_chunk,
        row_chunk,
        max_expression_chunk_shape,
    )
    return source_name, library_size, scale, mean_expr, detect_rate, var_expr


def _resolve_sample_ids(adata: AnnData, cfg, warnings: list[str]) -> tuple[np.ndarray, str]:
    key = str(getattr(cfg, "spot_sample_id_col", "") or "").strip()
    if key == "":
        raise ValueError("[map_utils]: cfg.spot_sample_id_col must be non-empty.")
    if key not in adata.obs.columns:
        raise KeyError(f"[map_utils]: required sample column obs['{key}'] is missing.")
    values = pd.Series(adata.obs[key]).astype("string").str.strip()
    if values.shape[0] != int(adata.n_obs):
        raise ValueError(
            f"[map_utils]: obs['{key}'] length mismatch "
            f"(observed={values.shape[0]}, expected={adata.n_obs})."
        )
    if values.isna().any():
        raise ValueError(f"[map_utils]: obs['{key}'] contains missing sample IDs.")
    if (values == "").any():
        raise ValueError(f"[map_utils]: obs['{key}'] contains empty sample IDs.")
    return values.astype(str).to_numpy(copy=False), key

def _row_normalize_graph(adj, n_obs: int, warnings: list[str]) -> csr_matrix:
    try:
        if not issparse(adj):
            adj = csr_matrix(np.asarray(adj, dtype=np.float32))
        adj = adj.tocsr().astype(np.float32)
    except Exception as exc:
        raise RuntimeError("[map_utils]: formal microdomain graph cannot be converted to CSR float32.") from exc
    if adj.shape != (int(n_obs), int(n_obs)):
        raise RuntimeError(
            f"[map_utils]: formal microdomain graph shape {adj.shape} does not match n_obs={n_obs}."
        )
    adj.eliminate_zeros()
    if adj.nnz == 0:
        raise RuntimeError("[map_utils]: formal microdomain graph has no edges.")
    if not np.all(np.isfinite(adj.data)):
        raise RuntimeError("[map_utils]: formal microdomain graph contains non-finite weights.")
    if np.any(adj.data < 0.0):
        raise RuntimeError("[map_utils]: formal microdomain graph contains negative weights.")
    row_sum = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    zero_rows = np.where(row_sum <= 1e-8)[0]
    if zero_rows.size > 0:
        adj = adj + csr_matrix(
            (np.ones(zero_rows.size, dtype=np.float32), (zero_rows, zero_rows)),
            shape=adj.shape,
            dtype=np.float32,
        )
        row_sum = np.asarray(adj.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    inv = np.zeros_like(row_sum, dtype=np.float32)
    valid = row_sum > 1e-8
    inv[valid] = 1.0 / row_sum[valid]
    return (diags(inv) @ adj).tocsr().astype(np.float32)

def _resolve_microdomain_weights(
    adata: AnnData,
    cfg,
    sample_ids: np.ndarray | None,
    warnings: list[str],
) -> tuple[csr_matrix | None, str | None]:
    key = str(getattr(cfg, "rep_microdomain_graph_key", "") or "").strip()
    if key == "":
        raise ValueError("[map_utils]: cfg.rep_microdomain_graph_key must be non-empty.")
    if key not in adata.obsp:
        raise KeyError(f"[map_utils]: required formal microdomain graph obsp['{key}'] is missing.")
    if sample_ids is None:
        raise ValueError("[map_utils]: sample IDs are required to validate the formal microdomain graph.")
    adj = restrict_graph_by_sample(adata.obsp[key], sample_ids)
    return _row_normalize_graph(adj, adata.n_obs, warnings), key


def _resolve_microdomain_weights_large(
    adata: AnnData,
    cfg,
    sample_ids: np.ndarray | None,
    row_indices: np.ndarray,
    warnings: list[str],
) -> tuple[csr_matrix | None, str | None]:
    key = str(getattr(cfg, "rep_microdomain_graph_key", "") or "").strip()
    if key == "":
        raise ValueError("[map_utils]: cfg.rep_microdomain_graph_key must be non-empty.")
    if key not in adata.obsp:
        raise KeyError(f"[map_utils]: required formal microdomain graph obsp['{key}'] is missing.")
    if sample_ids is None:
        raise ValueError("[map_utils]: sample IDs are required to validate the formal microdomain graph.")

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= int(adata.n_obs)):
        raise IndexError("[map_utils]: large graph row index is out of bounds.")
    graph = adata.obsp[key]
    if graph.shape != (int(adata.n_obs), int(adata.n_obs)):
        raise RuntimeError(
            f"[map_utils]: formal microdomain graph shape {graph.shape} does not match n_obs={adata.n_obs}."
        )
    adj = graph[rows, :][:, rows]
    adj = restrict_graph_by_sample(adj, sample_ids)
    return _row_normalize_graph(adj, int(rows.size), warnings), key


def _compute_mlre_matrix(
    x_log: np.ndarray,
    sample_ids: np.ndarray | None,
    micro_weights: csr_matrix | None,
    *,
    chunk_size: int,
) -> np.ndarray:
    """Compute nonnegative microdomain local rank enrichment for all common genes."""
    x = np.asarray(x_log, dtype=np.float32)
    if x.ndim != 2:
        raise ValueError("[map_utils]: x_log must be a 2D spot-by-gene matrix.")
    n_spots, n_genes = int(x.shape[0]), int(x.shape[1])
    weights = None
    if micro_weights is not None:
        weights = micro_weights.tocsr().astype(np.float32)
        if weights.shape != (n_spots, n_spots):
            raise ValueError(
                f"[map_utils]: micro_weights shape {weights.shape} does not match n_spots={n_spots}."
            )
    out = np.zeros((n_spots, n_genes), dtype=np.float32)
    chunk = max(int(chunk_size), 1)

    if sample_ids is None or np.asarray(sample_ids).shape[0] != n_spots:
        sample_indices = [np.arange(n_spots, dtype=np.int64)]
    else:
        sid = pd.Index(np.asarray(sample_ids, dtype=object).astype(str))
        sid_values = sid.to_numpy(copy=False)
        sample_indices = [np.where(sid_values == sample)[0].astype(np.int64, copy=False) for sample in sid.unique().tolist()]

    for idx in sample_indices:
        if idx.size == 0:
            continue
        n_sample = int(idx.size)
        sample_weights = weights[idx, :][:, idx] if weights is not None else None
        for start in range(0, n_genes, chunk):
            end = min(start + chunk, n_genes)
            ranks = rankdata(x[idx, start:end], axis=0, method="average").astype(np.float32, copy=False)
            ranks = (ranks / float(max(n_sample, 1))).astype(np.float32, copy=False)
            mean_rank = np.mean(ranks, axis=0, dtype=np.float64).astype(np.float32, copy=False)
            mean_rank = np.maximum(mean_rank, np.float32(1e-12))
            if sample_weights is None:
                micro_rank = ranks
            else:
                micro_rank = np.asarray(sample_weights @ ranks, dtype=np.float32)
            local = micro_rank.astype(np.float64) / mean_rank.astype(np.float64)[None, :] - 1.0
            local[~np.isfinite(local)] = 0.0
            local = np.maximum(local, 0.0)
            out[idx, start:end] = local.astype(np.float32, copy=False)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)


def _compute_mlre_matrix_large(
    adata: AnnData,
    row_indices: np.ndarray,
    gene_positions: np.ndarray,
    scale: np.ndarray,
    sample_ids: np.ndarray | None,
    micro_weights: csr_matrix | None,
    *,
    count_layer: str | None,
    chunk_size: int,
) -> np.ndarray:
    if count_layer is not None and count_layer in adata.layers:
        x_source = adata.layers[count_layer]
    elif "counts" in adata.layers:
        x_source = adata.layers["counts"]
    else:
        x_source = adata.X

    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    positions = np.asarray(gene_positions, dtype=np.int64).reshape(-1)
    scale_values = np.asarray(scale, dtype=np.float32).reshape(-1)
    if np.any(rows < 0) or np.any(rows >= int(adata.n_obs)):
        raise IndexError("[map_utils]: large MLRE row index is out of bounds.")
    if np.any(positions < 0) or np.any(positions >= int(adata.n_vars)):
        raise IndexError("[map_utils]: large MLRE gene index is out of bounds.")
    if scale_values.shape[0] != rows.shape[0]:
        raise ValueError(
            "[map_utils]: large MLRE scale length does not match the selected spot count "
            f"(scale={scale_values.shape[0]}, spots={rows.shape[0]})."
        )

    n_spots, n_genes = int(rows.size), int(positions.size)
    weights = None
    if micro_weights is not None:
        weights = micro_weights.tocsr().astype(np.float32)
        if weights.shape != (n_spots, n_spots):
            raise ValueError(
                f"[map_utils]: micro_weights shape {weights.shape} does not match n_spots={n_spots}."
            )
    out = np.zeros((n_spots, n_genes), dtype=np.float32)
    chunk = max(int(chunk_size), 1)

    if sample_ids is None or np.asarray(sample_ids).shape[0] != n_spots:
        sample_indices = [np.arange(n_spots, dtype=np.int64)]
    else:
        sid = pd.Index(np.asarray(sample_ids, dtype=object).astype(str))
        sid_values = sid.to_numpy(copy=False)
        sample_indices = [
            np.where(sid_values == sample)[0].astype(np.int64, copy=False)
            for sample in sid.unique().tolist()
        ]

    max_expression_chunk_shape = (0, 0)
    for idx in sample_indices:
        if idx.size == 0:
            continue
        n_sample = int(idx.size)
        sample_weights = weights[idx, :][:, idx] if weights is not None else None
        sample_rows = rows[idx]
        sample_scale = scale_values[idx]
        for start in range(0, n_genes, chunk):
            end = min(start + chunk, n_genes)
            chunk_positions = positions[start:end]
            if issparse(x_source):
                x_raw = x_source[:, chunk_positions][sample_rows, :].tocsr().astype(np.float32)
                x_norm = x_raw.multiply(sample_scale[:, None])
                x_log = np.log1p(x_norm.toarray()).astype(np.float32, copy=False)
            else:
                x_arr = np.asarray(
                    x_source[np.ix_(sample_rows, chunk_positions)],
                    dtype=np.float32,
                )
                x_log = np.log1p(
                    np.clip(x_arr * sample_scale[:, None], 0.0, None)
                ).astype(np.float32, copy=False)
            x_log[~np.isfinite(x_log)] = 0.0
            ranks = rankdata(x_log, axis=0, method="average").astype(np.float32, copy=False)
            ranks = (ranks / float(max(n_sample, 1))).astype(np.float32, copy=False)
            mean_rank = np.mean(ranks, axis=0, dtype=np.float64).astype(np.float32, copy=False)
            mean_rank = np.maximum(mean_rank, np.float32(1e-12))
            if sample_weights is None:
                micro_rank = ranks
            else:
                micro_rank = np.asarray(sample_weights @ ranks, dtype=np.float32)
            local = micro_rank.astype(np.float64) / mean_rank.astype(np.float64)[None, :] - 1.0
            local[~np.isfinite(local)] = 0.0
            local = np.maximum(local, 0.0)
            out[idx, start:end] = local.astype(np.float32, copy=False)
            if int(x_log.shape[0]) * int(x_log.shape[1]) > (
                int(max_expression_chunk_shape[0]) * int(max_expression_chunk_shape[1])
            ):
                max_expression_chunk_shape = (int(x_log.shape[0]), int(x_log.shape[1]))
            del x_log
    out[~np.isfinite(out)] = 0.0
    logger.info(
        "[map_utils.py]: large MLRE n_spots=%d n_requested_genes=%d chunk_size=%d "
        "max_expression_chunk_shape=%s output_shape=%s full_mlre_materialized=%s",
        n_spots,
        n_genes,
        chunk,
        max_expression_chunk_shape,
        (n_spots, n_genes),
        bool(n_genes == int(adata.n_vars)),
    )
    return out.astype(np.float32, copy=False)


def _weighted_mlre_score(
    mlre: np.ndarray,
    weights: np.ndarray,
    *,
    chunk_size: int | None = None,
    eps: float = 1e-12,
) -> np.ndarray:
    local = np.asarray(mlre, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    if local.ndim != 2 or local.shape[1] != w.shape[0]:
        raise ValueError("[map_utils]: mlre and weights have incompatible shapes.")
    w[~np.isfinite(w)] = 0.0
    denom = float(np.sum(w))
    out = np.zeros((local.shape[0],), dtype=np.float64)
    if denom <= float(eps):
        return out.astype(np.float32, copy=False)
    # Restrict the matrix product to genes with positive weights.
    active = np.where(w > 0.0)[0].astype(np.int64, copy=False)
    if active.size == 0:
        return out.astype(np.float32, copy=False)
    w_active = w[active]
    if chunk_size is None:
        out = np.asarray(local[:, active], dtype=np.float64) @ w_active
    else:
        chunk = max(int(chunk_size), 1)
        for start in range(0, local.shape[0], chunk):
            end = min(start + chunk, local.shape[0])
            out[start:end] = np.asarray(local[start:end, :][:, active], dtype=np.float64) @ w_active
    out = out / (denom + float(eps))
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32, copy=False)

def _quantile_bins(values: np.ndarray, n_bins: int) -> np.ndarray:
    s = pd.Series(np.asarray(values, dtype=np.float64))
    if s.nunique(dropna=True) <= 1:
        return np.zeros((len(s),), dtype=np.int32)
    bins = pd.qcut(
        s.rank(method="average"),
        q=min(int(max(n_bins, 1)), int(s.nunique(dropna=True))),
        labels=False,
        duplicates="drop",
    )
    return np.asarray(pd.Series(bins).fillna(0), dtype=np.int32)

def _make_gene_matching_bins(mean_expr: np.ndarray, detect_rate: np.ndarray, n_bins: int) -> tuple[np.ndarray, dict[str, Any]]:
    mean_bins = _quantile_bins(mean_expr, n_bins)
    detect_bins = _quantile_bins(detect_rate, n_bins)
    pair_keys = pd.Series([f"{a}_{b}" for a, b in zip(mean_bins.tolist(), detect_bins.tolist())], dtype="string")
    codes, uniques = pd.factorize(pair_keys, sort=True)
    counts = pd.Series(codes).value_counts(dropna=False)
    return codes.astype(np.int32, copy=False), {
        "n_bins_requested": int(n_bins),
        "n_bins_realized": int(len(uniques)),
        "matching_scheme": "quantile_bins(mean_st_expression,st_detection_rate)",
        "binning_features": ["mean_st_expression", "st_detection_rate"],
        "n_singleton_bins": int(np.sum(counts.to_numpy(dtype=np.int64) == 1)),
        "min_bin_size": int(counts.min()) if not counts.empty else 0,
        "max_bin_size": int(counts.max()) if not counts.empty else 0,
    }

def _matched_weight_permutation_pvalues(
    mlre: np.ndarray,
    weights: np.ndarray,
    observed: np.ndarray,
    control_bins: np.ndarray,
    *,
    n_perm: int,
    seed: int,
    chunk_size: int,
    eps: float = 1e-12,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    local = np.asarray(mlre, dtype=np.float32)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    bins = np.asarray(control_bins, dtype=np.int32).reshape(-1)
    if local.ndim != 2 or local.shape[1] != w.shape[0] or bins.shape[0] != w.shape[0]:
        raise ValueError("[map_utils]: mlre, weights, and control_bins have incompatible shapes.")
    if obs.shape[0] != local.shape[0]:
        raise ValueError("[map_utils]: observed score length does not match mlre rows.")
    w[~np.isfinite(w)] = 0.0
    weight_sum = float(np.sum(w))
    n_perm_eff = max(int(n_perm), 1)
    rng = np.random.default_rng(int(seed))
    unique_bins = pd.Index(bins).unique().tolist()
    bin_indices = [np.where(bins == b)[0].astype(np.int64, copy=False) for b in unique_bins]

    ge_count = np.ones((local.shape[0],), dtype=np.int64)
    null_sum = np.zeros((local.shape[0],), dtype=np.float64)
    null_sq_sum = np.zeros((local.shape[0],), dtype=np.float64)
    perm_chunk = max(int(chunk_size), 1)
    for start in range(0, n_perm_eff, perm_chunk):
        this_chunk = min(perm_chunk, n_perm_eff - start)
        permuted_weights = np.empty((w.shape[0], this_chunk), dtype=np.float32)
        for col in range(this_chunk):
            permuted = w.copy()
            for idx in bin_indices:
                if idx.size > 1:
                    permuted[idx] = w[rng.permutation(idx)]
            permuted_weights[:, col] = permuted.astype(np.float32, copy=False)
        if weight_sum <= float(eps):
            null_score = np.zeros((local.shape[0], this_chunk), dtype=np.float32)
        else:
            null_score = np.asarray(local @ permuted_weights, dtype=np.float32)
            null_score = (null_score.astype(np.float64) / (weight_sum + float(eps))).astype(np.float32, copy=False)
        null_score[~np.isfinite(null_score)] = 0.0
        ge_count += np.sum(null_score >= obs[:, None], axis=1).astype(np.int64, copy=False)
        null_sum += np.sum(null_score, axis=1, dtype=np.float64)
        null_sq_sum += np.sum(null_score.astype(np.float64) ** 2, axis=1)

    null_mean = null_sum / float(n_perm_eff)
    null_var = np.maximum(null_sq_sum / float(n_perm_eff) - null_mean * null_mean, 0.0)
    null_std = np.sqrt(null_var)
    p = ge_count.astype(np.float64) / float(n_perm_eff + 1)
    q = _bh_qvalues(p)
    meta = {
        "available": True,
        "null_method": "matched_within_bin_continuous_weight_permutation",
        "n_perm": int(n_perm_eff),
        "n_genes": int(w.shape[0]),
        "n_nonzero_weights": int(np.sum(w > 0.0)),
        "weight_sum": weight_sum,
        "seed": int(seed),
        "chunk_size": int(perm_chunk),
        "eps": float(eps),
        "continuous_weights": True,
        "observed_summary": _summary_stats(obs),
        "null_mean_summary": _summary_stats(null_mean),
        "null_std_summary": _summary_stats(null_std),
    }
    if weight_sum <= float(eps):
        meta["warning"] = "nonpositive_weight_sum"
    return (
        p.astype(np.float32, copy=False),
        q.astype(np.float32, copy=False),
        null_mean.astype(np.float32, copy=False),
        null_std.astype(np.float32, copy=False),
        meta,
    )


def _matched_weight_permutation_pvalues_large(
    adata: AnnData,
    row_indices: np.ndarray,
    weights: np.ndarray,
    observed: np.ndarray,
    control_bins: np.ndarray,
    scale: np.ndarray,
    sample_ids: np.ndarray | None,
    micro_weights: csr_matrix | None,
    *,
    count_layer: str | None,
    mlre_chunk_size: int,
    n_perm: int,
    seed: int,
    chunk_size: int,
    eps: float = 1e-12,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    rows = np.asarray(row_indices, dtype=np.int64).reshape(-1)
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    obs = np.asarray(observed, dtype=np.float64).reshape(-1)
    bins = np.asarray(control_bins, dtype=np.int32).reshape(-1)
    scale_values = np.asarray(scale, dtype=np.float32).reshape(-1)
    if w.shape[0] != int(adata.n_vars) or bins.shape[0] != w.shape[0]:
        raise ValueError("[map_utils]: weights and control_bins must span the full ST gene universe.")
    if obs.shape[0] != rows.shape[0]:
        raise ValueError("[map_utils]: observed score length does not match selected spot rows.")
    if scale_values.shape[0] != rows.shape[0]:
        raise ValueError("[map_utils]: large matched-null scale length does not match selected spot rows.")
    if micro_weights is not None and micro_weights.shape != (int(rows.size), int(rows.size)):
        raise ValueError(
            "[map_utils]: large matched-null micro_weights shape does not match selected spot rows."
        )
    w[~np.isfinite(w)] = 0.0
    weight_sum = float(np.sum(w))
    n_perm_eff = max(int(n_perm), 1)
    rng = np.random.default_rng(int(seed))
    unique_bins = pd.Index(bins).unique().tolist()
    bin_indices = [
        np.where(bins == b)[0].astype(np.int64, copy=False)
        for b in unique_bins
    ]

    ge_count = np.ones((rows.shape[0],), dtype=np.int64)
    null_sum = np.zeros((rows.shape[0],), dtype=np.float64)
    null_sq_sum = np.zeros((rows.shape[0],), dtype=np.float64)
    perm_chunk = max(int(chunk_size), 1)
    for start in range(0, n_perm_eff, perm_chunk):
        this_chunk = min(perm_chunk, n_perm_eff - start)
        permuted_weights = np.empty((w.shape[0], this_chunk), dtype=np.float32)
        for col in range(this_chunk):
            permuted = w.copy()
            for idx in bin_indices:
                if idx.size > 1:
                    permuted[idx] = w[rng.permutation(idx)]
            permuted_weights[:, col] = permuted.astype(np.float32, copy=False)
        control_positions = np.flatnonzero(
            np.any(permuted_weights != 0.0, axis=1)
        ).astype(np.int64, copy=False)
        if weight_sum <= float(eps):
            null_score = np.zeros((rows.shape[0], this_chunk), dtype=np.float32)
            control_mlre_shape = (int(rows.size), 0)
        else:
            control_mlre = _compute_mlre_matrix_large(
                adata=adata,
                row_indices=rows,
                gene_positions=control_positions,
                scale=scale_values,
                sample_ids=sample_ids,
                micro_weights=micro_weights,
                count_layer=count_layer,
                chunk_size=mlre_chunk_size,
            )
            control_mlre_shape = tuple(int(value) for value in control_mlre.shape)
            null_score = np.asarray(
                control_mlre @ permuted_weights[control_positions, :],
                dtype=np.float32,
            )
            null_score = (
                null_score.astype(np.float64) / (weight_sum + float(eps))
            ).astype(np.float32, copy=False)
            del control_mlre
        null_score[~np.isfinite(null_score)] = 0.0
        ge_count += np.sum(null_score >= obs[:, None], axis=1).astype(np.int64, copy=False)
        null_sum += np.sum(null_score, axis=1, dtype=np.float64)
        null_sq_sum += np.sum(null_score.astype(np.float64) ** 2, axis=1)
        logger.info(
            "[map_utils.py]: large matched-null permutation_chunk=%d:%d "
            "n_control_genes=%d control_mlre_shape=%s permuted_weights_shape=%s",
            int(start),
            int(start + this_chunk),
            int(control_positions.size),
            control_mlre_shape,
            tuple(int(value) for value in permuted_weights.shape),
        )
        del null_score
        del control_positions
        del permuted_weights

    null_mean = null_sum / float(n_perm_eff)
    null_var = np.maximum(null_sq_sum / float(n_perm_eff) - null_mean * null_mean, 0.0)
    null_std = np.sqrt(null_var)
    p = ge_count.astype(np.float64) / float(n_perm_eff + 1)
    q = _bh_qvalues(p)
    meta = {
        "available": True,
        "null_method": "matched_within_bin_continuous_weight_permutation",
        "n_perm": int(n_perm_eff),
        "n_genes": int(w.shape[0]),
        "n_nonzero_weights": int(np.sum(w > 0.0)),
        "weight_sum": weight_sum,
        "seed": int(seed),
        "chunk_size": int(perm_chunk),
        "eps": float(eps),
        "continuous_weights": True,
        "observed_summary": _summary_stats(obs),
        "null_mean_summary": _summary_stats(null_mean),
        "null_std_summary": _summary_stats(null_std),
    }
    if weight_sum <= float(eps):
        meta["warning"] = "nonpositive_weight_sum"
    return (
        p.astype(np.float32, copy=False),
        q.astype(np.float32, copy=False),
        null_mean.astype(np.float32, copy=False),
        null_std.astype(np.float32, copy=False),
        meta,
    )


prepare_st_log_expression = _prepare_st_log_expression
resolve_sample_ids = _resolve_sample_ids
resolve_microdomain_weights = _resolve_microdomain_weights
compute_mlre_matrix = _compute_mlre_matrix
weighted_mlre_score = _weighted_mlre_score
make_gene_matching_bins = _make_gene_matching_bins
matched_weight_permutation_pvalues = _matched_weight_permutation_pvalues
prepare_st_log_expression_large = _prepare_st_log_expression_large
resolve_microdomain_weights_large = _resolve_microdomain_weights_large
compute_mlre_matrix_large = _compute_mlre_matrix_large
matched_weight_permutation_pvalues_large = _matched_weight_permutation_pvalues_large
