from __future__ import annotations


import hashlib
import logging
import math
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Union

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.stats import rankdata

from sparmap.utils.map_utils import (
    _compute_mlre_matrix,
    _compute_mlre_matrix_large,
    make_gene_matching_bins as _make_gene_matching_bins,
    matched_weight_permutation_pvalues as _matched_weight_permutation_pvalues,
    _matched_weight_permutation_pvalues_large,
    _prepare_st_log_expression,
    _prepare_st_log_expression_large,
    _resolve_microdomain_weights,
    _resolve_microdomain_weights_large,
    _resolve_sample_ids,
    weighted_mlre_score as _weighted_mlre_score,
)
from sparmap.utils.utils import (
    effective_gene_number as _effective_gene_number,
    finite_float32 as _finite_float32,
    h5ad_safe_uns_value as _h5ad_safe_uns_value,
    normalize_weights as _normalize_weights,
    sample_index_groups as _sample_index_groups,
    save_json as _save_json,
)

from sparmap._logging import get_logger

logger = get_logger(__name__)


METHOD_NAME = "SpaRMap"
MAP_MODE = "active_module_bank_projection"
SIGNED_MODULE_DISCOVERY_MODE = "program_guided_signed_active_module_refinement"
NEGATIVE_MODULE_SCORE_FORMULA = "module_base_score*anti_support_to_positive_bank"
FINAL_SCORE_FORMULA = "active_module_bank_projection"
ANCHOR_SOURCE = "full_universe_matched_null_mlre_anchor"
ANCHOR_UNIVERSE = "full_st_gene_universe"
PROGRAM_ANCHOR_REFERENCE = "calibrated_full_program_anchor"

MAP_POSITIVE_KEY = "map_single_active_module_positive"
MAP_NEGATIVE_KEY = "map_single_active_module_negative"
MAP_CONTRAST_KEY = "map_single_active_module_contrast"
MAP_RANK_KEY = "map_single_active_module_rank"
PROGRAM_ANCHOR_KEY = "map_single_program_anchor"

SCORE_SPEC_FILENAME = "single_map_score_spec.csv"
PROGRAM_SUMMARY_FILENAME = "single_map_program_summary.csv"
SUMMARY_JSON_FILENAME = "single_map_summary.json"
EDGE_TABLE_FILENAME = "single_map_edge_table.csv"
POSITIVE_MODULE_TABLE_FILENAME = "single_map_positive_module_table.csv"
NEGATIVE_MODULE_TABLE_FILENAME = "single_map_negative_module_table.csv"
MODULE_GENE_TABLE_FILENAME = "single_map_module_gene_table.csv"
SAMPLE_SUMMARY_FILENAME = "single_map_sample_summary.csv"

SIGNED_RHO_MIN = 0.0
MODULE_OVERLAP_JACCARD = 0.80
MLRE_CHUNK_SIZE = 512
EPS = 1e-12


EDGE_TABLE_COLUMNS = [
    "program",
    "program_token",
    "sample_id",
    "gene_i",
    "gene_j",
    "local_i",
    "local_j",
    "edge_role",
    "edge_sign",
    "signed_spearman_rho",
    "program_conditioned_weight",
    "topology_weight",
    "program_consistency_i",
    "program_consistency_j",
    "rho_min",
    "used_for_positive_module",
    "used_for_negative_module",
    "used_for_positive_negative_relation",
]

MODULE_TABLE_COLUMNS = [
    "program",
    "program_token",
    "sample_id",
    "module_id",
    "module_type",
    "module_name",
    "module_found",
    "n_genes",
    "n_eff_genes",
    "genes",
    "top_genes",
    "module_program_consistency",
    "module_coactivity",
    "support_factor",
    "module_base_score",
    "module_score",
    "module_weight",
    "anti_support_to_positive_bank",
    "connected",
    "module_stopped_by_safety_guard",
    "seed_edge",
    "seed_gene_i",
    "seed_gene_j",
    "n_expansion_steps",
    "warning",
]

MODULE_GENE_TABLE_COLUMNS = [
    "program",
    "program_token",
    "sample_id",
    "module_id",
    "module_type",
    "module_name",
    "gene",
    "program_consistency_positive",
    "program_consistency_negative",
    "rho_to_program_anchor_reference",
    "intramodular_support",
    "module_gene_weight",
    "module_weight",
    "module_base_score",
    "gene_rank_in_module",
    "top_weighted_gene_flag",
]

SAMPLE_SUMMARY_COLUMNS = [
    "program",
    "program_token",
    "sample_id",
    "n_input_genes",
    "n_finite_mlre_genes",
    "n_positive_edges",
    "n_negative_edges",
    "n_positive_active_modules",
    "n_negative_active_modules",
    "positive_module_weight_entropy",
    "negative_module_weight_entropy",
    "positive_effective_support_total",
    "negative_effective_support_total",
    "positive_top_module_size",
    "positive_top_module_n_eff",
    "positive_top_module_score",
    "negative_top_module_size",
    "negative_top_module_n_eff",
    "negative_top_module_score",
    "negative_antagonism_available",
    "module_stopped_by_safety_guard",
    "positive_projection_available",
    "negative_projection_available",
    "contrast_negative_missing",
    "anchor_source",
    "anchor_universe",
    "n_anchor_universe_genes",
    "n_positive_anchor_genes",
    "anchor_status",
    "anchor_effective_n_genes",
    "matched_null_status",
    "matched_null_available",
    "uses_full_gene_set_anchor",
    "warnings",
]

PROGRAM_SUMMARY_COLUMNS = [
    "program",
    "program_token",
    "method",
    "map_mode",
    "module_discovery_mode",
    "positive_score_key",
    "negative_score_key",
    "contrast_score_key",
    "final_formula",
    "n_input_genes",
    "n_overlap_genes",
    "n_samples",
    "n_samples_positive_active_module_found",
    "n_samples_negative_active_module_found",
    "n_samples_negative_projection_available",
    "n_samples_no_positive_active_module",
    "anchor_source",
    "anchor_universe",
    "n_anchor_universe_genes",
    "n_positive_anchor_genes",
    "anchor_status",
    "anchor_effective_n_genes",
    "matched_null_status",
    "matched_null_available",
    "uses_full_gene_set_anchor",
]


@dataclass
class ActiveModuleCandidate:
    found: bool
    nodes: list[int]
    module_score: float
    module_program_consistency: float
    module_coactivity: float
    support_factor: float
    n_eff_genes: float
    gene_intramodular_support: np.ndarray
    gene_weights: np.ndarray
    stopped_by_safety_guard: bool
    seed_edge: tuple[int, int] | None
    n_steps: int
    warning: str = ""
    module_weight: float = 0.0
    anti_support_to_positive_bank: float = float("nan")
    module_base_score: float = float("nan")


@dataclass
class _SingleMapSampleContext:
    sample_id: str
    sample_token: str
    global_indices: np.ndarray
    sample_ids: np.ndarray | None
    mlre: np.ndarray
    control_bin_codes: np.ndarray
    control_bin_metadata: dict[str, Any]
    expression_source: str
    microdomain_graph_key: str | None
    mean_expr: np.ndarray
    detect_rate: np.ndarray
    var_expr: np.ndarray


@dataclass
class _SingleMapSampleContextLarge:
    sample_id: str
    sample_token: str
    global_indices: np.ndarray
    sample_ids: np.ndarray | None
    control_bin_codes: np.ndarray
    control_bin_metadata: dict[str, Any]
    expression_source: str
    microdomain_graph_key: str | None
    mean_expr: np.ndarray
    detect_rate: np.ndarray
    var_expr: np.ndarray
    scale: np.ndarray
    micro_weights: Any


def _invalid_active_module(
    warning: str,
    *,
    nodes: Optional[list[int]] = None,
    seed_edge: tuple[int, int] | None = None,
    stopped_by_safety_guard: bool = False,
    n_steps: int = 0,
) -> ActiveModuleCandidate:
    return ActiveModuleCandidate(
        found=False,
        nodes=[] if nodes is None else [int(node) for node in nodes],
        module_score=float("nan"),
        module_program_consistency=float("nan"),
        module_coactivity=float("nan"),
        support_factor=float("nan"),
        n_eff_genes=float("nan"),
        gene_intramodular_support=np.zeros((0,), dtype=np.float32),
        gene_weights=np.zeros((0,), dtype=np.float32),
        stopped_by_safety_guard=bool(stopped_by_safety_guard),
        seed_edge=seed_edge,
        n_steps=int(n_steps),
        warning=str(warning or ""),
        module_base_score=float("nan"),
    )


def _add_warning(warnings: list[str], message: str) -> None:
    text = str(message)
    if text not in warnings:
        warnings.append(text)
    logger.warning("[single_map]: %s", text)


def _cfg_int(cfg: Any, name: str, default: int) -> int:
    try:
        value = int(getattr(cfg, name, default))
    except Exception:
        value = int(default)
    return int(value)


def _cfg_float(cfg: Any, name: str, default: float) -> float:
    try:
        value = float(getattr(cfg, name, default))
    except Exception:
        value = float(default)
    return float(value) if np.isfinite(value) else float(default)


def _resolve_output_dir(cfg: Any, output_dir: Optional[Union[str, Path]]) -> Path:
    if output_dir is not None:
        out = Path(output_dir)
    else:
        save_root = Path(getattr(cfg, "save_path", None) or "result")
        out = save_root / "map"
    out.mkdir(parents=True, exist_ok=True)
    return out


def _safe_token(value: Any, max_len: int = 60) -> str:
    token = re.sub(r"[^0-9A-Za-z]+", "_", str(value or "program").strip().lower()).strip("_")
    return (token or "program")[:max_len]


def _sample_context_token(sample_id: str) -> str:
    raw = str(sample_id)
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{_safe_token(raw, max_len=48)}_{digest}"


def _unique_program_tokens(programs: list[str]) -> dict[str, str]:
    used: dict[str, int] = {}
    out: dict[str, str] = {}
    for program in programs:
        base = _safe_token(program)
        count = used.get(base, 0) + 1
        used[base] = count
        out[str(program)] = base if count == 1 else f"{base}_{count}"
    return out


def _read_gene_set_table(path: Union[str, Path]) -> pd.DataFrame:
    p = Path(path)
    suffix = p.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(p)
    if suffix in {".tsv", ".txt"}:
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p, sep=None, engine="python")


def _prepare_gene_set_df(gene_set_df: pd.DataFrame, warnings: list[str]) -> pd.DataFrame:
    if gene_set_df is None or gene_set_df.empty:
        raise ValueError("[single_map]: gene set table is empty.")
    cols = {str(col).strip().lower(): col for col in gene_set_df.columns}
    gene_col = cols.get("gene") or cols.get("genes") or cols.get("symbol") or cols.get("gene_symbol")
    if gene_col is None:
        raise ValueError("[single_map]: gene set table must contain a gene column.")
    program_col = cols.get("program") or cols.get("gene_set") or cols.get("set") or cols.get("pathway") or cols.get("name")
    weight_col = cols.get("weight") or cols.get("score") or cols.get("gene_weight")
    direction_col = cols.get("direction") or cols.get("sign") or cols.get("regulation")

    df = pd.DataFrame()
    df["gene"] = gene_set_df[gene_col].astype("string").str.strip()
    df = df[df["gene"].notna() & (df["gene"] != "")].copy()
    if df.empty:
        raise ValueError("[single_map]: no valid gene rows after parsing gene set table.")
    df["gene_upper"] = df["gene"].astype(str).str.upper()
    if program_col is None:
        df["program"] = "program"
    else:
        df["program"] = gene_set_df.loc[df.index, program_col].astype("string").fillna("program").str.strip()
        df.loc[df["program"] == "", "program"] = "program"
        df["program"] = df["program"].astype(str)
    if weight_col is None:
        df["weight"] = 1.0
    else:
        weight = pd.to_numeric(gene_set_df.loc[df.index, weight_col], errors="coerce").to_numpy(dtype=float)
        weight[~np.isfinite(weight) | (weight <= 0.0)] = 1.0
        df["weight"] = weight
    if direction_col is not None:
        raw_direction = gene_set_df.loc[df.index, direction_col].astype("string").fillna("").str.strip()
        if bool((raw_direction != "").any()):
            _add_warning(warnings, "direction column is diagnostic only; active modules are inferred from MLRE topology.")
    # Canonical ordering prevents input-row-dependent program offsets and local gene indices.
    df = (
        df.sort_values(
            by=["program", "weight", "gene_upper", "gene"],
            ascending=[True, False, True, True],
            kind="mergesort",
        )
        .drop_duplicates(subset=["program", "gene_upper"], keep="first")
        .reset_index(drop=True)
    )
    return df


def _build_var_lookup(adata: AnnData) -> dict[str, tuple[int, str]]:
    out: dict[str, tuple[int, str]] = {}
    for idx, gene in enumerate(pd.Index(adata.var_names).astype(str).tolist()):
        key = str(gene).strip().upper()
        if key and key not in out:
            out[key] = (int(idx), str(gene))
    return out


def _sample_label(sample_ids: np.ndarray | None, idx: np.ndarray, fallback: int) -> str:
    if sample_ids is None or idx.size == 0:
        return "all" if sample_ids is None else str(fallback)
    sid = np.asarray(sample_ids, dtype=object).reshape(-1)
    if sid.shape[0] <= int(idx[0]):
        return str(fallback)
    return str(sid[int(idx[0])])


def _build_per_sample_contexts(
    adata_map: AnnData,
    cfg: Any,
    sample_ids: np.ndarray | None,
    *,
    full_st_pos: np.ndarray,
    mlre_chunk: int,
    control_bins: int,
    warnings: list[str],
) -> list[_SingleMapSampleContext]:
    contexts: list[_SingleMapSampleContext] = []
    sample_groups = _sample_index_groups(int(adata_map.n_obs), sample_ids)
    use_microdomain = bool(getattr(cfg, "map_use_rep_microdomain", True))
    if not use_microdomain:
        _add_warning(warnings, "map_use_rep_microdomain_false; MLRE profiles use spot-level ranks.")

    for sample_ord, idx in enumerate(sample_groups):
        sample_idx = np.asarray(idx, dtype=np.int64)
        sample_id = _sample_label(sample_ids, sample_idx, sample_ord)
        sample_adata = adata_map[sample_idx, :].copy()
        sample_ids_local = (
            None
            if sample_ids is None
            else np.asarray(sample_ids, dtype=object).reshape(-1)[sample_idx]
        )
        x_log, expression_source, mean_expr, detect_rate, var_expr = _prepare_st_log_expression(
            sample_adata,
            full_st_pos,
            count_layer=getattr(cfg, "rep_count_layer", "counts"),
            target_sum=float(getattr(cfg, "rep_normalize_target_sum", 1e4)),
        )
        sample_warnings: list[str] = []
        micro_weights = None
        microdomain_graph_key = None
        if use_microdomain:
            micro_weights, microdomain_graph_key = _resolve_microdomain_weights(
                sample_adata,
                cfg,
                sample_ids_local,
                sample_warnings,
            )
        for warning in sample_warnings:
            _add_warning(warnings, f"sample_id={sample_id}:{warning}")
        mlre = _compute_mlre_matrix(
            x_log,
            sample_ids_local,
            micro_weights,
            chunk_size=mlre_chunk,
        )
        control_bin_codes, control_bin_metadata = _make_gene_matching_bins(
            mean_expr,
            detect_rate,
            control_bins,
        )
        contexts.append(
            _SingleMapSampleContext(
                sample_id=sample_id,
                sample_token=_sample_context_token(sample_id),
                global_indices=sample_idx,
                sample_ids=sample_ids_local,
                mlre=mlre,
                control_bin_codes=control_bin_codes,
                control_bin_metadata=control_bin_metadata,
                expression_source=expression_source,
                microdomain_graph_key=microdomain_graph_key,
                mean_expr=mean_expr,
                detect_rate=detect_rate,
                var_expr=var_expr,
            )
        )
    return contexts


def _build_per_sample_contexts_large(
    adata_map: AnnData,
    cfg: Any,
    sample_ids: np.ndarray | None,
    *,
    full_st_pos: np.ndarray,
    mlre_chunk: int,
    control_bins: int,
    warnings: list[str],
) -> list[_SingleMapSampleContextLarge]:
    contexts: list[_SingleMapSampleContextLarge] = []
    sample_groups = _sample_index_groups(int(adata_map.n_obs), sample_ids)
    use_microdomain = bool(getattr(cfg, "map_use_rep_microdomain", True))
    if not use_microdomain:
        _add_warning(warnings, "map_use_rep_microdomain_false; MLRE profiles use spot-level ranks.")

    for sample_ord, idx in enumerate(sample_groups):
        sample_idx = np.asarray(idx, dtype=np.int64)
        sample_id = _sample_label(sample_ids, sample_idx, sample_ord)
        sample_ids_local = (
            None
            if sample_ids is None
            else np.asarray(sample_ids, dtype=object).reshape(-1)[sample_idx]
        )
        (
            expression_source,
            library_size,
            scale,
            mean_expr,
            detect_rate,
            var_expr,
        ) = _prepare_st_log_expression_large(
            adata_map,
            sample_idx,
            full_st_pos,
            count_layer=getattr(cfg, "rep_count_layer", "counts"),
            target_sum=float(getattr(cfg, "rep_normalize_target_sum", 1e4)),
            gene_chunk_size=mlre_chunk,
            row_chunk_size=mlre_chunk,
        )
        if library_size.shape[0] != sample_idx.shape[0] or scale.shape[0] != sample_idx.shape[0]:
            raise RuntimeError(
                "[single_map]: large expression statistics returned incompatible spot lengths "
                f"(sample_id={sample_id!r}, spots={sample_idx.shape[0]}, "
                f"library_size={library_size.shape[0]}, scale={scale.shape[0]})."
            )
        sample_warnings: list[str] = []
        micro_weights = None
        microdomain_graph_key = None
        if use_microdomain:
            micro_weights, microdomain_graph_key = _resolve_microdomain_weights_large(
                adata_map,
                cfg,
                sample_ids_local,
                sample_idx,
                sample_warnings,
            )
        for warning in sample_warnings:
            _add_warning(warnings, f"sample_id={sample_id}:{warning}")
        control_bin_codes, control_bin_metadata = _make_gene_matching_bins(
            mean_expr,
            detect_rate,
            control_bins,
        )
        logger.info(
            "[single_map]: large context sample_id=%s n_spots=%d n_full_genes=%d "
            "expression_gene_chunk_size=%d full_x_log_materialized=False "
            "full_mlre_materialized=False",
            sample_id,
            int(sample_idx.size),
            int(full_st_pos.size),
            int(mlre_chunk),
        )
        contexts.append(
            _SingleMapSampleContextLarge(
                sample_id=sample_id,
                sample_token=_sample_context_token(sample_id),
                global_indices=sample_idx,
                sample_ids=sample_ids_local,
                control_bin_codes=control_bin_codes,
                control_bin_metadata=control_bin_metadata,
                expression_source=expression_source,
                microdomain_graph_key=microdomain_graph_key,
                mean_expr=mean_expr,
                detect_rate=detect_rate,
                var_expr=var_expr,
                scale=scale,
                micro_weights=micro_weights,
            )
        )
        del library_size
    return contexts


def _samplewise_rank01(values: np.ndarray, sample_ids: np.ndarray | None) -> np.ndarray:
    x = np.asarray(values, dtype=float).reshape(-1)
    out = np.full(x.shape, np.nan, dtype=np.float32)
    for idx in _sample_index_groups(x.size, sample_ids):
        finite = np.isfinite(x[idx])
        if not np.any(finite):
            continue
        local_idx = idx[finite]
        ranks = pd.Series(x[local_idx]).rank(method="average", pct=True).to_numpy(dtype=float)
        out[local_idx] = ranks.astype(np.float32, copy=False)
    return out


def _finite_profile_columns(values: np.ndarray) -> np.ndarray:
    x = np.asarray(values, dtype=float)
    if x.ndim != 2 or x.shape[1] == 0:
        return np.zeros((0,), dtype=bool)
    finite = np.all(np.isfinite(x), axis=0)
    spread = np.nanmax(x, axis=0) - np.nanmin(x, axis=0)
    return finite & np.isfinite(spread) & (spread > EPS)


def _spearman_vector(x: np.ndarray, y: np.ndarray) -> float:
    xv = np.asarray(x, dtype=float).reshape(-1)
    yv = np.asarray(y, dtype=float).reshape(-1)
    mask = np.isfinite(xv) & np.isfinite(yv)
    if int(np.sum(mask)) < 3:
        return float("nan")
    xr = rankdata(xv[mask], method="average").astype(float, copy=False)
    yr = rankdata(yv[mask], method="average").astype(float, copy=False)
    xr -= float(np.mean(xr))
    yr -= float(np.mean(yr))
    denom = math.sqrt(float(np.sum(xr * xr) * np.sum(yr * yr)))
    if not np.isfinite(denom) or denom <= EPS:
        return float("nan")
    return float(np.clip(float(np.sum(xr * yr) / denom), -1.0, 1.0))


def _program_anchor_consistency(
    profiles: np.ndarray,
    program_anchor_reference: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z = np.asarray(profiles, dtype=float)
    ref = np.asarray(program_anchor_reference, dtype=float).reshape(-1)
    n_genes = int(z.shape[1]) if z.ndim == 2 else 0
    rho = np.full((n_genes,), np.nan, dtype=np.float32)
    if z.ndim != 2 or ref.size != int(z.shape[0]):
        return rho, np.zeros((n_genes,), dtype=np.float32), np.zeros((n_genes,), dtype=np.float32)
    for gene_idx in range(n_genes):
        rho[gene_idx] = np.float32(_spearman_vector(z[:, gene_idx], ref))
    a_pos = np.maximum(np.where(np.isfinite(rho), rho, 0.0), 0.0).astype(np.float32, copy=False)
    a_neg = np.maximum(np.where(np.isfinite(rho), -rho, 0.0), 0.0).astype(np.float32, copy=False)
    return rho, a_pos, a_neg


def _signed_spearman_matrix(profiles: np.ndarray) -> np.ndarray:
    x = np.asarray(profiles, dtype=float)
    n_genes = int(x.shape[1]) if x.ndim == 2 else 0
    if x.ndim != 2 or x.shape[0] < 3 or n_genes < 2:
        return np.full((n_genes, n_genes), np.nan, dtype=np.float32)
    ranks = rankdata(x, axis=0, method="average").astype(float, copy=False)
    ranks -= np.mean(ranks, axis=0, keepdims=True)
    denom = np.sqrt(np.sum(ranks * ranks, axis=0))
    valid = np.isfinite(denom) & (denom > EPS)
    corr = np.full((n_genes, n_genes), np.nan, dtype=float)
    if int(np.sum(valid)) >= 2:
        z = ranks[:, valid] / denom[valid][None, :]
        corr_valid = np.asarray(z.T @ z, dtype=float)
        valid_idx = np.flatnonzero(valid)
        corr[np.ix_(valid_idx, valid_idx)] = corr_valid
    np.fill_diagonal(corr, np.nan)
    corr[~np.isfinite(corr)] = np.nan
    return np.clip(corr, -1.0, 1.0).astype(np.float32, copy=False)


def _build_program_conditioned_graph(
    corr: np.ndarray,
    node_activity: np.ndarray,
    *,
    rho_min: float,
) -> tuple[np.ndarray, list[tuple[int, int, float, float]]]:
    r = np.asarray(corr, dtype=float)
    activity = np.asarray(node_activity, dtype=float).reshape(-1)
    n = int(r.shape[0]) if r.ndim == 2 else 0
    adj = np.zeros((n, n), dtype=np.float32)
    if n < 2 or activity.size != n:
        return adj, []
    activity = np.maximum(np.where(np.isfinite(activity), activity, 0.0), 0.0)
    node_factor = np.sqrt(np.outer(activity, activity))
    base = np.where(np.isfinite(r) & (r >= float(rho_min)), r, 0.0)
    scores = np.maximum(base, 0.0) * node_factor
    np.fill_diagonal(scores, 0.0)
    edges: list[tuple[int, int, float, float]] = []
    for i in range(n):
        for j in range(i + 1, n):
            weight = float(scores[i, j])
            rho = float(r[i, j])
            if not np.isfinite(rho) or rho < float(rho_min):
                continue
            if not np.isfinite(weight) or weight <= 0.0:
                continue
            adj[i, j] = adj[j, i] = np.float32(weight)
            edges.append((int(i), int(j), rho, weight))
    return adj, edges


def _build_positive_negative_cross_antagonism(
    corr: np.ndarray,
    a_pos: np.ndarray,
    a_neg: np.ndarray,
    *,
    rho_min: float,
) -> np.ndarray:
    r = np.asarray(corr, dtype=float)
    pos = np.maximum(np.where(np.isfinite(a_pos), np.asarray(a_pos, dtype=float), 0.0), 0.0)
    neg = np.maximum(np.where(np.isfinite(a_neg), np.asarray(a_neg, dtype=float), 0.0), 0.0)
    n = int(r.shape[0]) if r.ndim == 2 else 0
    if n == 0 or pos.size != n or neg.size != n:
        return np.zeros((n, n), dtype=np.float32)
    base = np.where(np.isfinite(r) & (r <= -float(rho_min)), -r, 0.0)
    cross = np.maximum(base, 0.0) * np.sqrt(np.outer(pos, neg))
    np.fill_diagonal(cross, 0.0)
    return cross.astype(np.float32, copy=False)


def _is_connected(adj: np.ndarray, nodes: list[int]) -> bool:
    if len(nodes) <= 1:
        return bool(nodes)
    node_set = {int(node) for node in nodes}
    seen = {int(nodes[0])}
    stack = [int(nodes[0])]
    while stack:
        cur = stack.pop()
        neigh = np.flatnonzero(np.asarray(adj[cur], dtype=float) > 0.0)
        for item in neigh.tolist():
            node = int(item)
            if node in node_set and node not in seen:
                seen.add(node)
                stack.append(node)
    return seen == node_set


def _mean_internal_weight(adj: np.ndarray, nodes: list[int]) -> float:
    if len(nodes) < 2:
        return float("nan")
    sub = np.asarray(adj[np.ix_(nodes, nodes)], dtype=float)
    upper_sum = float(np.sum(np.triu(sub, k=1)))
    denom = len(nodes) * (len(nodes) - 1) / 2.0
    value = upper_sum / denom if denom > 0.0 else float("nan")
    return float(value) if np.isfinite(value) else float("nan")


def _evaluate_active_module(
    adj: np.ndarray,
    node_activity: np.ndarray,
    nodes: list[int],
    *,
    seed_edge: tuple[int, int] | None,
    n_steps: int,
    stopped_by_safety_guard: bool = False,
    warning: str = "",
) -> ActiveModuleCandidate:
    ordered_nodes = list(dict.fromkeys(int(node) for node in nodes))
    if len(ordered_nodes) < 2:
        return _invalid_active_module("active_module_requires_at_least_two_genes", nodes=ordered_nodes, seed_edge=seed_edge)
    if stopped_by_safety_guard:
        return _invalid_active_module(
            warning or "module_stopped_by_safety_guard",
            nodes=ordered_nodes,
            seed_edge=seed_edge,
            stopped_by_safety_guard=True,
            n_steps=n_steps,
        )
    if not _is_connected(adj, ordered_nodes):
        return _invalid_active_module("active_module_not_connected", nodes=ordered_nodes, seed_edge=seed_edge, n_steps=n_steps)

    sub = np.asarray(adj[np.ix_(ordered_nodes, ordered_nodes)], dtype=float)
    kin = np.sum(sub, axis=1)
    kin[~np.isfinite(kin) | (kin < 0.0)] = 0.0
    activity = np.asarray(node_activity, dtype=float).reshape(-1)
    local_activity = activity[ordered_nodes] if activity.size > max(ordered_nodes) else np.zeros((len(ordered_nodes),), dtype=float)
    local_activity = np.maximum(np.where(np.isfinite(local_activity), local_activity, 0.0), 0.0)
    pre_weight = local_activity * kin
    total_pre = float(np.sum(pre_weight))
    if not np.isfinite(total_pre) or total_pre <= EPS:
        return _invalid_active_module(
            "active_module_program_guided_support_zero",
            nodes=ordered_nodes,
            seed_edge=seed_edge,
            n_steps=n_steps,
        )
    gene_weights = pre_weight / total_pre
    concentration = float(np.sum(gene_weights * gene_weights))
    n_eff = float(1.0 / max(concentration, EPS))
    support_factor = float(np.clip(1.0 - concentration, 0.0, 1.0))
    module_program_consistency = float(np.mean(local_activity)) if local_activity.size else float("nan")
    module_coactivity = _mean_internal_weight(adj, ordered_nodes)
    module_base_score = float(module_program_consistency * module_coactivity * support_factor)
    if (
        not np.isfinite(module_base_score)
        or module_base_score <= EPS
        or not np.isfinite(module_program_consistency)
        or not np.isfinite(module_coactivity)
        or not np.isfinite(support_factor)
    ):
        return _invalid_active_module(
            "active_module_score_nonpositive",
            nodes=ordered_nodes,
            seed_edge=seed_edge,
            n_steps=n_steps,
        )
    return ActiveModuleCandidate(
        found=True,
        nodes=ordered_nodes,
        module_score=module_base_score,
        module_program_consistency=module_program_consistency,
        module_coactivity=module_coactivity,
        support_factor=support_factor,
        n_eff_genes=n_eff,
        gene_intramodular_support=kin.astype(np.float32, copy=False),
        gene_weights=gene_weights.astype(np.float32, copy=False),
        stopped_by_safety_guard=False,
        seed_edge=seed_edge,
        n_steps=int(n_steps),
        warning=str(warning or ""),
        module_base_score=module_base_score,
    )


def _nodewise_strongest_seed_edges(adj: np.ndarray) -> list[tuple[int, int, float]]:
    a = np.asarray(adj, dtype=float)
    if a.ndim != 2 or a.shape[0] < 2:
        return []
    selected: dict[tuple[int, int], float] = {}
    for i in range(a.shape[0]):
        neighbors = np.flatnonzero(np.isfinite(a[i]) & (a[i] > 0.0))
        neighbors = neighbors[neighbors != i]
        if neighbors.size == 0:
            continue
        weights = a[i, neighbors]
        max_weight = float(np.max(weights))
        j = int(np.min(neighbors[weights == max_weight]))
        edge = (min(int(i), j), max(int(i), j))
        selected.setdefault(edge, float(a[i, j]))
    edges = [(i, j, weight) for (i, j), weight in selected.items()]
    edges.sort(key=lambda item: (-item[2], item[0], item[1]))
    return edges


def _grow_active_module(
    adj: np.ndarray,
    node_activity: np.ndarray,
    seed_edge: tuple[int, int],
) -> ActiveModuleCandidate:
    n = int(adj.shape[0]) if adj.ndim == 2 else 0
    nodes = [int(seed_edge[0]), int(seed_edge[1])]
    current = _evaluate_active_module(
        adj,
        node_activity,
        nodes,
        seed_edge=seed_edge,
        n_steps=0,
    )
    if not current.found:
        return current

    steps = 0
    max_steps = max(n, 1)
    while True:
        if steps >= max_steps:
            return _evaluate_active_module(
                adj,
                node_activity,
                nodes,
                seed_edge=seed_edge,
                n_steps=steps,
                stopped_by_safety_guard=True,
                warning="module_stopped_by_safety_guard",
            )
        node_set = set(nodes)
        candidates: set[int] = set()
        for node in nodes:
            candidates.update(int(x) for x in np.flatnonzero(np.asarray(adj[int(node)], dtype=float) > 0.0).tolist())
        candidates.difference_update(node_set)
        if not candidates:
            break

        accepted: list[tuple[float, float, int, ActiveModuleCandidate]] = []
        for candidate in sorted(candidates):
            grown = _evaluate_active_module(
                adj,
                node_activity,
                nodes + [int(candidate)],
                seed_edge=seed_edge,
                n_steps=steps + 1,
            )
            if not grown.found:
                continue
            if grown.module_score + EPS < current.module_score:
                continue
            accepted.append((float(grown.module_score), float(grown.n_eff_genes), int(candidate), grown))
        if not accepted:
            break
        accepted.sort(key=lambda item: (-item[0], -item[1], item[2]))
        current = accepted[0][3]
        nodes = list(current.nodes)
        steps += 1
    return current


def _jaccard(nodes_a: list[int], nodes_b: list[int]) -> float:
    a = {int(node) for node in nodes_a}
    b = {int(node) for node in nodes_b}
    if not a and not b:
        return 1.0
    union = a | b
    return float(len(a & b) / len(union)) if union else 0.0


def _normalize_module_weights(modules: list[ActiveModuleCandidate]) -> list[ActiveModuleCandidate]:
    valid_modules = [m for m in modules if m.found]
    total = float(np.sum([m.module_score for m in valid_modules if np.isfinite(m.module_score) and m.module_score > 0.0]))
    if total <= EPS:
        if not valid_modules:
            return []
        uniform = 1.0 / float(len(valid_modules))
        for module in valid_modules:
            module.module_weight = uniform
        return valid_modules
    for module in valid_modules:
        module.module_weight = float(module.module_score / total) if np.isfinite(module.module_score) and module.module_score > 0.0 else 0.0
    return valid_modules


def _discover_active_module_bank(
    adj: np.ndarray,
    node_activity: np.ndarray,
    *,
    overlap_jaccard: float,
) -> tuple[list[ActiveModuleCandidate], dict[str, Any]]:
    seed_edges = _nodewise_strongest_seed_edges(adj)
    diagnostics = {
        "n_seed_edges": int(len(seed_edges)),
        "n_candidate_modules": 0,
        "n_redundant_active_modules": 0,
        "n_modules_stopped_by_safety_guard": 0,
    }
    if not seed_edges:
        return [], diagnostics
    candidates: list[ActiveModuleCandidate] = []
    for i, j, _weight in seed_edges:
        module = _grow_active_module(
            adj,
            node_activity,
            (int(i), int(j)),
        )
        if bool(module.stopped_by_safety_guard):
            diagnostics["n_modules_stopped_by_safety_guard"] = int(diagnostics["n_modules_stopped_by_safety_guard"]) + 1
        if module.found:
            candidates.append(module)
    diagnostics["n_candidate_modules"] = int(len(candidates))
    candidates.sort(key=lambda m: (-float(m.module_score), -float(m.n_eff_genes), len(m.nodes), tuple(m.nodes)))

    selected: list[ActiveModuleCandidate] = []
    redundancy_threshold = float(np.clip(overlap_jaccard, 0.0, 1.0))
    for module in candidates:
        if any(_jaccard(module.nodes, kept.nodes) > redundancy_threshold for kept in selected):
            diagnostics["n_redundant_active_modules"] = int(diagnostics["n_redundant_active_modules"]) + 1
            continue
        selected.append(module)
    selected = _normalize_module_weights(selected)
    return selected, diagnostics


def _bank_gene_weight_vector(n_genes: int, modules: list[ActiveModuleCandidate]) -> np.ndarray:
    weights = np.zeros((int(n_genes),), dtype=float)
    for module in modules:
        if not module.found or not module.nodes:
            continue
        module_weight = float(module.module_weight)
        if not np.isfinite(module_weight) or module_weight <= 0.0:
            continue
        local_w = np.asarray(module.gene_weights, dtype=float).reshape(-1)
        if local_w.size != len(module.nodes):
            continue
        for pos, node in enumerate(module.nodes):
            weights[int(node)] += module_weight * max(float(local_w[pos]), 0.0)
    total = float(np.sum(weights))
    if total > EPS:
        weights /= total
    return weights.astype(np.float32, copy=False)


def _module_cross_antagonism(
    cross: np.ndarray,
    positive_modules: list[ActiveModuleCandidate],
    negative_module: ActiveModuleCandidate,
) -> float:
    n = int(cross.shape[0]) if cross.ndim == 2 else 0
    if n == 0 or not positive_modules or not negative_module.found:
        return float("nan")
    pos_w = _bank_gene_weight_vector(n, positive_modules).astype(float, copy=False)
    neg_w = np.zeros((n,), dtype=float)
    local = np.asarray(negative_module.gene_weights, dtype=float).reshape(-1)
    if local.size != len(negative_module.nodes):
        return float("nan")
    for pos, node in enumerate(negative_module.nodes):
        neg_w[int(node)] += max(float(local[pos]), 0.0)
    neg_total = float(np.sum(neg_w))
    if float(np.sum(pos_w)) <= EPS or neg_total <= EPS:
        return float("nan")
    neg_w /= neg_total
    value = float(np.sum(np.asarray(cross, dtype=float) * np.outer(pos_w, neg_w)))
    return value if np.isfinite(value) else float("nan")


def _finalize_negative_modules(
    negative_modules: list[ActiveModuleCandidate],
    positive_modules: list[ActiveModuleCandidate],
    cross: np.ndarray,
) -> list[ActiveModuleCandidate]:
    scored: list[ActiveModuleCandidate] = []
    for module in negative_modules:
        anti = _module_cross_antagonism(cross, positive_modules, module)
        module.anti_support_to_positive_bank = anti
        if not np.isfinite(anti) or anti <= EPS:
            continue
        module.module_score = float(module.module_base_score * anti)
        if np.isfinite(module.module_score) and module.module_score > EPS:
            scored.append(module)
    scored.sort(key=lambda m: (-float(m.module_score), -float(m.n_eff_genes), len(m.nodes), tuple(m.nodes)))
    return _normalize_module_weights(scored)


def _module_bank_projection(
    profiles: np.ndarray,
    modules: list[ActiveModuleCandidate],
) -> np.ndarray:
    z = np.asarray(profiles, dtype=float)
    if z.ndim != 2 or not modules:
        return np.full(
            (z.shape[0] if z.ndim == 2 else 0,),
            np.nan,
            dtype=np.float32,
        )

    out = np.zeros((z.shape[0],), dtype=float)
    has_module = False

    for module in modules:
        if not module.found or not module.nodes:
            continue

        module_weight = float(module.module_weight)
        gene_weights = np.asarray(module.gene_weights, dtype=float).reshape(-1)

        if (
            gene_weights.size != len(module.nodes)
            or not np.isfinite(module_weight)
            or module_weight <= 0.0
        ):
            continue

        out += module_weight * (
            z[:, module.nodes] @ gene_weights
        )
        has_module = True

    if not has_module:
        return np.full((z.shape[0],), np.nan, dtype=np.float32)

    out[~np.isfinite(out)] = np.nan
    return out.astype(np.float32, copy=False)


def _module_weight_entropy(modules: list[ActiveModuleCandidate]) -> float:
    w = np.asarray([m.module_weight for m in modules if m.found and np.isfinite(m.module_weight) and m.module_weight > 0.0], dtype=float)
    if w.size == 0:
        return float("nan")
    w = w / float(np.sum(w))
    return float(-np.sum(w * np.log(w + EPS)))


def _effective_support_total(modules: list[ActiveModuleCandidate]) -> float:
    values = [
        float(m.module_weight) * float(m.n_eff_genes)
        for m in modules
        if m.found and np.isfinite(m.module_weight) and np.isfinite(m.n_eff_genes)
    ]
    return float(np.sum(values)) if values else float("nan")


def _module_gene_names(genes: list[str], nodes: list[int]) -> list[str]:
    return [str(genes[int(node)]) for node in nodes]


def _module_table_rows(
    *,
    program: str,
    program_token: str,
    sample_id: str,
    module_type: str,
    modules: list[ActiveModuleCandidate],
    genes: list[str],
    warning: str = "",
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not modules:
        rows.append(
            {
                "program": program,
                "program_token": program_token,
                "sample_id": sample_id,
                "module_id": f"{program_token}_{sample_id}_{module_type}_none",
                "module_type": module_type,
                "module_name": f"{program_token}_{sample_id}_{module_type}_active_module_none",
                "module_found": False,
                "n_genes": 0,
                "n_eff_genes": np.nan,
                "genes": "",
                "top_genes": "",
                "module_program_consistency": np.nan,
                "module_coactivity": np.nan,
                "support_factor": np.nan,
                "module_base_score": np.nan,
                "module_score": np.nan,
                "module_weight": np.nan,
                "anti_support_to_positive_bank": np.nan,
                "connected": False,
                "module_stopped_by_safety_guard": False,
                "seed_edge": "",
                "seed_gene_i": "",
                "seed_gene_j": "",
                "n_expansion_steps": 0,
                "warning": warning,
            }
        )
        return rows
    for idx, module in enumerate(modules, start=1):
        names = _module_gene_names(genes, module.nodes)
        local_order = np.argsort(-np.asarray(module.gene_weights, dtype=float), kind="mergesort")
        top_names = [names[int(pos)] for pos in local_order[: min(10, len(names))].tolist()]
        module_id = f"{program_token}_{sample_id}_{module_type}_{idx:03d}"
        seed_i = seed_j = ""
        seed_edge = ""
        if module.seed_edge is not None:
            seed_i = str(genes[int(module.seed_edge[0])])
            seed_j = str(genes[int(module.seed_edge[1])])
            seed_edge = f"{seed_i};{seed_j}"
        rows.append(
            {
                "program": program,
                "program_token": program_token,
                "sample_id": sample_id,
                "module_id": module_id,
                "module_type": module_type,
                "module_name": f"{program_token}_{sample_id}_{module_type}_active_module_{idx:03d}",
                "module_found": bool(module.found),
                "n_genes": int(len(module.nodes)),
                "n_eff_genes": float(module.n_eff_genes),
                "genes": ";".join(names),
                "top_genes": ";".join(top_names),
                "module_program_consistency": float(module.module_program_consistency),
                "module_coactivity": float(module.module_coactivity),
                "support_factor": float(module.support_factor),
                "module_base_score": float(module.module_base_score) if np.isfinite(module.module_base_score) else np.nan,
                "module_score": float(module.module_score),
                "module_weight": float(module.module_weight),
                "anti_support_to_positive_bank": (
                    float(module.anti_support_to_positive_bank)
                    if np.isfinite(module.anti_support_to_positive_bank)
                    else np.nan
                ),
                "connected": True,
                "module_stopped_by_safety_guard": bool(module.stopped_by_safety_guard),
                "seed_edge": seed_edge,
                "seed_gene_i": seed_i,
                "seed_gene_j": seed_j,
                "n_expansion_steps": int(module.n_steps),
                "warning": str(module.warning or ""),
            }
        )
    return rows


def _append_module_gene_rows(
    rows: list[dict[str, Any]],
    *,
    program: str,
    program_token: str,
    sample_id: str,
    module_type: str,
    modules: list[ActiveModuleCandidate],
    genes: list[str],
    a_pos: np.ndarray,
    a_neg: np.ndarray,
    rho_program: np.ndarray,
) -> None:
    for idx, module in enumerate(modules, start=1):
        if not module.found:
            continue
        module_id = f"{program_token}_{sample_id}_{module_type}_{idx:03d}"
        module_name = f"{program_token}_{sample_id}_{module_type}_active_module_{idx:03d}"
        local_weights = np.asarray(module.gene_weights, dtype=float)
        rank_order = np.argsort(-local_weights, kind="mergesort")
        ranks = np.empty((len(module.nodes),), dtype=int)
        for rank_pos, local_pos in enumerate(rank_order.tolist(), start=1):
            ranks[int(local_pos)] = int(rank_pos)
        top_cut = max(1, int(math.ceil(0.20 * max(len(module.nodes), 1))))
        kin = np.asarray(module.gene_intramodular_support, dtype=float)
        for local_pos, node in enumerate(module.nodes):
            node_int = int(node)
            rows.append(
                {
                    "program": program,
                    "program_token": program_token,
                    "sample_id": sample_id,
                    "module_id": module_id,
                    "module_type": module_type,
                    "module_name": module_name,
                    "gene": str(genes[node_int]),
                    "program_consistency_positive": float(a_pos[node_int]) if node_int < int(a_pos.size) else np.nan,
                    "program_consistency_negative": float(a_neg[node_int]) if node_int < int(a_neg.size) else np.nan,
                    "rho_to_program_anchor_reference": (
                        float(rho_program[node_int]) if node_int < int(rho_program.size) else np.nan
                    ),
                    "intramodular_support": float(kin[local_pos]) if local_pos < int(kin.size) else np.nan,
                    "module_gene_weight": (
                        float(local_weights[local_pos]) if local_pos < int(local_weights.size) else np.nan
                    ),
                    "module_weight": float(module.module_weight),
                    "module_base_score": (
                        float(module.module_base_score)
                        if np.isfinite(module.module_base_score)
                        else np.nan
                    ),
                    "gene_rank_in_module": int(ranks[local_pos]),
                    "top_weighted_gene_flag": bool(int(ranks[local_pos]) <= top_cut),
                }
            )


def _edge_records(
    *,
    program: str,
    program_token: str,
    sample_id: str,
    genes: list[str],
    corr: np.ndarray,
    pos_edges: list[tuple[int, int, float, float]],
    neg_edges: list[tuple[int, int, float, float]],
    cross: np.ndarray,
    a_pos: np.ndarray,
    a_neg: np.ndarray,
    rho_min: float,
    positive_modules: list[ActiveModuleCandidate],
    negative_modules: list[ActiveModuleCandidate],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    positive_nodes = {int(node) for module in positive_modules for node in module.nodes}
    negative_nodes = {int(node) for module in negative_modules for node in module.nodes}

    for role, edges, activity in (
        ("positive_internal", pos_edges, a_pos),
        ("negative_internal", neg_edges, a_neg),
    ):
        for i, j, rho, weight in edges:
            i_int, j_int = int(i), int(j)
            rows.append(
                {
                    "program": program,
                    "program_token": program_token,
                    "sample_id": sample_id,
                    "gene_i": genes[i_int],
                    "gene_j": genes[j_int],
                    "local_i": i_int,
                    "local_j": j_int,
                    "edge_role": role,
                    "edge_sign": "positive",
                    "signed_spearman_rho": float(rho),
                    "program_conditioned_weight": float(weight),
                    "topology_weight": float(weight),
                    "program_consistency_i": float(activity[i_int]) if i_int < int(activity.size) else np.nan,
                    "program_consistency_j": float(activity[j_int]) if j_int < int(activity.size) else np.nan,
                    "rho_min": float(rho_min),
                    "used_for_positive_module": bool(role == "positive_internal" and {i_int, j_int}.issubset(positive_nodes)),
                    "used_for_negative_module": bool(role == "negative_internal" and {i_int, j_int}.issubset(negative_nodes)),
                    "used_for_positive_negative_relation": False,
                }
            )

    if positive_nodes and negative_nodes:
        cross_arr = np.asarray(cross, dtype=float)
        corr_arr = np.asarray(corr, dtype=float)
        for i_int in sorted(positive_nodes):
            for j_int in sorted(negative_nodes):
                weight = float(cross_arr[i_int, j_int]) if i_int < cross_arr.shape[0] and j_int < cross_arr.shape[1] else 0.0
                if not np.isfinite(weight) or weight <= 0.0:
                    continue
                rho = float(corr_arr[i_int, j_int]) if np.isfinite(corr_arr[i_int, j_int]) else np.nan
                rows.append(
                    {
                        "program": program,
                        "program_token": program_token,
                        "sample_id": sample_id,
                        "gene_i": genes[i_int],
                        "gene_j": genes[j_int],
                        "local_i": i_int,
                        "local_j": j_int,
                        "edge_role": "positive_negative_relation",
                        "edge_sign": "negative",
                        "signed_spearman_rho": rho,
                        "program_conditioned_weight": weight,
                        "topology_weight": weight,
                        "program_consistency_i": float(a_pos[i_int]) if i_int < int(a_pos.size) else np.nan,
                        "program_consistency_j": float(a_neg[j_int]) if j_int < int(a_neg.size) else np.nan,
                        "rho_min": float(rho_min),
                        "used_for_positive_module": False,
                        "used_for_negative_module": False,
                        "used_for_positive_negative_relation": True,
                    }
                )
    return rows


def _aggregate_program_vectors(program_results: list[dict[str, Any]], key: str, n_obs: int) -> np.ndarray:
    if not program_results:
        return np.full((int(n_obs),), np.nan, dtype=np.float32)
    matrix = []
    weights = []
    for result in program_results:
        values = np.asarray(result.get(key), dtype=float).reshape(-1)
        if values.size != int(n_obs):
            continue
        matrix.append(values)
        weights.append(max(float(result.get("program_weight", 1.0) or 0.0), 0.0))
    if not matrix:
        return np.full((int(n_obs),), np.nan, dtype=np.float32)
    stack = np.column_stack(matrix)
    w = np.asarray(weights, dtype=float)
    w[~np.isfinite(w) | (w < 0.0)] = 0.0
    if float(np.sum(w)) <= EPS:
        w = np.ones_like(w, dtype=float)
    out = np.full((int(n_obs),), np.nan, dtype=float)
    for row in range(stack.shape[0]):
        finite = np.isfinite(stack[row])
        if not np.any(finite):
            continue
        local_w = w[finite]
        denom = float(np.sum(local_w))
        if denom <= EPS:
            local_w = np.ones_like(local_w, dtype=float)
            denom = float(np.sum(local_w))
        out[row] = float(np.sum(stack[row, finite] * local_w) / denom)
    return out.astype(np.float32, copy=False)


def _write_obs_vector(adata: AnnData, key: str, values: np.ndarray, output_obs_keys: list[str]) -> None:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size != int(adata.n_obs):
        raise ValueError(f"[single_map]: obs vector {key} has size {arr.size}, expected {adata.n_obs}.")
    adata.obs[key] = arr
    if key not in output_obs_keys:
        output_obs_keys.append(key)


def _write_score_spec(outdir: Path) -> Path:
    path = outdir / SCORE_SPEC_FILENAME
    pd.DataFrame(
        [
            {
                "method": METHOD_NAME,
                "score_role": "positive",
                "score_key": MAP_POSITIVE_KEY,
                "score_source": "obs",
                "score_direction": 1,
                "is_final": True,
            },
            {
                "method": METHOD_NAME,
                "score_role": "negative",
                "score_key": MAP_NEGATIVE_KEY,
                "score_source": "obs",
                "score_direction": 1,
                "is_final": True,
            },
            {
                "method": METHOD_NAME,
                "score_role": "contrast",
                "score_key": MAP_CONTRAST_KEY,
                "score_source": "obs",
                "score_direction": 1,
                "is_final": True,
            },
        ]
    ).to_csv(path, index=False)
    return path


def _program_gene_positions(
    program_df: pd.DataFrame,
    common_lookup: dict[str, int],
) -> tuple[list[str], list[int], list[float]]:
    genes: list[str] = []
    positions: list[int] = []
    weights: list[float] = []
    seen: set[str] = set()
    for row in program_df.to_dict(orient="records"):
        key = str(row.get("gene_upper", "")).strip().upper()
        if key == "" or key in seen or key not in common_lookup:
            continue
        seen.add(key)
        genes.append(str(row.get("gene", key)))
        positions.append(int(common_lookup[key]))
        try:
            weight = float(row.get("weight", 1.0))
        except Exception:
            weight = 1.0
        weights.append(weight if np.isfinite(weight) and weight > 0.0 else 1.0)
    return genes, positions, weights


def _empty_sample_result(
    *,
    program: str,
    program_token: str,
    sample_id: str,
    n_input: int,
    n_finite: int,
    n_spots: int,
    warning: str,
    calibrated_anchor_values_sample: Optional[np.ndarray] = None,
    anchor_metadata: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    warnings = [warning]
    anchor_values = np.asarray(
        calibrated_anchor_values_sample
        if calibrated_anchor_values_sample is not None
        else np.full((int(n_spots),), np.nan, dtype=np.float32),
        dtype=np.float32,
    ).reshape(-1)
    if anchor_values.size != int(n_spots):
        anchor_values = np.full((int(n_spots),), np.nan, dtype=np.float32)
    anchor_fields = dict(anchor_metadata or {})
    return {
        "positive": np.full((int(n_spots),), np.nan, dtype=np.float32),
        "negative": np.full((int(n_spots),), np.nan, dtype=np.float32),
        "contrast": np.full((int(n_spots),), np.nan, dtype=np.float32),
        "program_anchor": anchor_values.astype(np.float32, copy=False),
        "edge_rows": [],
        "positive_module_rows": _module_table_rows(
            program=program,
            program_token=program_token,
            sample_id=sample_id,
            module_type="positive",
            modules=[],
            genes=[],
            warning=warning,
        ),
        "negative_module_rows": _module_table_rows(
            program=program,
            program_token=program_token,
            sample_id=sample_id,
            module_type="negative",
            modules=[],
            genes=[],
            warning="negative_active_module_not_searched_without_positive_bank",
        ),
        "module_gene_rows": [],
        "summary_row": {
            "program": program,
            "program_token": program_token,
            "sample_id": sample_id,
            "n_input_genes": int(n_input),
            "n_finite_mlre_genes": int(n_finite),
            "n_positive_edges": 0,
            "n_negative_edges": 0,
            "n_positive_active_modules": 0,
            "n_negative_active_modules": 0,
            "n_positive_module_families": 0,
            "positive_module_weight_entropy": np.nan,
            "negative_module_weight_entropy": np.nan,
            "positive_effective_support_total": np.nan,
            "negative_effective_support_total": np.nan,
            "positive_top_module_size": 0,
            "positive_top_module_n_eff": np.nan,
            "positive_top_module_score": np.nan,
            "negative_top_module_size": 0,
            "negative_top_module_n_eff": np.nan,
            "negative_top_module_score": np.nan,
            "negative_antagonism_available": False,
            "positive_top_family_weight": np.nan,
            "module_stopped_by_safety_guard": False,
            "positive_projection_available": False,
            "negative_projection_available": False,
            "contrast_negative_missing": False,
            **anchor_fields,
            "warnings": ";".join(warnings),
        },
    }


def _zero_gene_set_anchor_scores(n_spots: int, status: str = "empty_anchor_gene_set") -> dict[str, Any]:
    unavailable = np.full((max(int(n_spots), 0),), np.nan, dtype=np.float32)
    return {
        "activity": unavailable.copy(),
        "null_z": unavailable.copy(),
        "null_z_unclipped": unavailable.copy(),
        "rank": unavailable.copy(),
        "status": str(status),
        "n_anchor_genes": 0,
        "effective_n_anchor_genes": 0.0,
        "matched_null_meta": {},
    }


def _directional_anchor_weights(
    input_weight_full: np.ndarray,
    input_sign_full: np.ndarray,
    *,
    direction: str,
) -> np.ndarray:
    weights = np.asarray(input_weight_full, dtype=np.float64).reshape(-1)
    signs = np.asarray(input_sign_full, dtype=np.float64).reshape(-1)
    n_genes = int(weights.size)
    out = np.zeros((n_genes,), dtype=np.float32)
    if n_genes <= 0:
        return out
    usable = int(min(weights.size, signs.size))
    clean_weights = np.zeros((usable,), dtype=np.float64)
    if usable > 0:
        clean_weights = np.maximum(np.nan_to_num(weights[:usable], nan=0.0, posinf=0.0, neginf=0.0), 0.0)
    normalized_direction = str(direction).strip().lower()
    if normalized_direction != "positive":
        return out
    mask = clean_weights > 0.0
    if usable > 0 and np.any(mask):
        selected = np.where(mask)[0]
        out[selected] = clean_weights[selected].astype(np.float32, copy=False)
    out[~np.isfinite(out)] = 0.0
    out[out < 0.0] = 0.0
    return out.astype(np.float32, copy=False)


def _compute_gene_set_anchor_scores(
    *,
    mlre: np.ndarray,
    weights_full: np.ndarray,
    control_bin_codes: np.ndarray,
    n_perm: int,
    seed: int,
    perm_chunk_size: int,
    z_max: float,
    sample_ids: Optional[np.ndarray],
) -> dict[str, Any]:
    local = np.asarray(mlre, dtype=np.float32)
    n_spots = int(local.shape[0]) if local.ndim >= 1 else 0
    n_genes = int(local.shape[1]) if local.ndim == 2 else int(np.asarray(weights_full).size)
    clean_weights = np.zeros((max(n_genes, 0),), dtype=np.float32)
    raw = np.asarray(weights_full, dtype=np.float64).reshape(-1)
    usable = int(min(clean_weights.size, raw.size))
    if usable > 0:
        w = np.nan_to_num(raw[:usable], nan=0.0, posinf=0.0, neginf=0.0)
        clean_weights[:usable] = np.maximum(w, 0.0).astype(np.float32, copy=False)
    if clean_weights.size <= 0 or float(np.sum(clean_weights)) <= 0.0:
        return _zero_gene_set_anchor_scores(n_spots, status="empty_anchor_gene_set")

    normalized_weights = _normalize_weights(clean_weights).astype(np.float32, copy=False)
    if float(np.sum(normalized_weights)) <= 0.0:
        return _zero_gene_set_anchor_scores(n_spots, status="empty_anchor_gene_set")
    activity = _finite_float32(_weighted_mlre_score(local, normalized_weights))
    n_anchor_genes = int(np.count_nonzero(clean_weights > 0.0))
    effective_n_anchor_genes = float(_effective_gene_number(normalized_weights))
    try:
        _, _, null_mean, null_std, null_meta = _matched_weight_permutation_pvalues(
            local,
            normalized_weights,
            activity,
            control_bin_codes,
            n_perm=max(int(n_perm), 1),
            seed=int(seed),
            chunk_size=max(int(perm_chunk_size), 1),
        )
        denom = np.asarray(null_std, dtype=np.float64).reshape(-1)
        observed = np.asarray(activity, dtype=np.float64).reshape(-1)
        mean = np.asarray(null_mean, dtype=np.float64).reshape(-1)
        n = int(min(observed.size, mean.size, denom.size))
        null_z_unclipped = np.zeros((observed.size,), dtype=np.float64)
        if n > 0:
            valid_std = np.isfinite(denom[:n]) & (denom[:n] > 1e-8)
            valid_pos = np.where(valid_std)[0]
            null_z_unclipped[valid_pos] = (observed[valid_pos] - mean[valid_pos]) / denom[valid_pos]
        null_z_unclipped[~np.isfinite(null_z_unclipped)] = 0.0
        null_z = np.clip(null_z_unclipped, -float(z_max), float(z_max))
        rank = _samplewise_rank01(null_z_unclipped, sample_ids)
        null_meta = dict(null_meta)
        null_meta["status"] = "computed"
        status = "computed"
    except Exception as exc:
        raise RuntimeError(
            "[single_map]: matched-null anchor computation failed "
            f"(n_spots={n_spots}, n_genes={n_genes}, n_anchor_genes={n_anchor_genes}, "
            f"n_perm={max(int(n_perm), 1)}, seed={int(seed)})."
        ) from exc

    return {
        "activity": activity.astype(np.float32, copy=False),
        "null_z": np.asarray(null_z, dtype=np.float32),
        "null_z_unclipped": np.asarray(null_z_unclipped, dtype=np.float32),
        "rank": np.asarray(rank, dtype=np.float32),
        "status": status,
        "n_anchor_genes": int(n_anchor_genes),
        "effective_n_anchor_genes": float(effective_n_anchor_genes),
        "matched_null_meta": null_meta,
    }


def _compute_gene_set_anchor_scores_large(
    *,
    adata_map: AnnData,
    context: _SingleMapSampleContextLarge,
    program_mlre: np.ndarray,
    program_positions: np.ndarray,
    weights_full: np.ndarray,
    control_bin_codes: np.ndarray,
    n_perm: int,
    seed: int,
    perm_chunk_size: int,
    mlre_chunk_size: int,
    z_max: float,
) -> dict[str, Any]:
    local = np.asarray(program_mlre, dtype=np.float32)
    positions = np.asarray(program_positions, dtype=np.int64).reshape(-1)
    n_spots = int(local.shape[0]) if local.ndim >= 1 else 0
    n_genes = int(np.asarray(weights_full).size)
    if local.ndim != 2 or local.shape[1] != positions.size:
        raise RuntimeError(
            "[single_map]: large program MLRE and program positions have incompatible shapes "
            f"(mlre_shape={local.shape}, n_program_positions={positions.size})."
        )
    if np.any(positions < 0) or np.any(positions >= n_genes):
        raise RuntimeError("[single_map]: large program position is outside the full ST gene universe.")

    clean_weights = np.zeros((max(n_genes, 0),), dtype=np.float32)
    raw = np.asarray(weights_full, dtype=np.float64).reshape(-1)
    usable = int(min(clean_weights.size, raw.size))
    if usable > 0:
        w = np.nan_to_num(raw[:usable], nan=0.0, posinf=0.0, neginf=0.0)
        clean_weights[:usable] = np.maximum(w, 0.0).astype(np.float32, copy=False)
    if clean_weights.size <= 0 or float(np.sum(clean_weights)) <= 0.0:
        return _zero_gene_set_anchor_scores(n_spots, status="empty_anchor_gene_set")

    normalized_weights = _normalize_weights(clean_weights).astype(np.float32, copy=False)
    if float(np.sum(normalized_weights)) <= 0.0:
        return _zero_gene_set_anchor_scores(n_spots, status="empty_anchor_gene_set")
    nonzero_positions = np.flatnonzero(normalized_weights != 0.0).astype(np.int64, copy=False)
    if not np.all(np.isin(nonzero_positions, positions)):
        raise RuntimeError(
            "[single_map]: large program positions do not cover every nonzero normalized anchor weight."
        )
    program_weights = normalized_weights[positions]
    activity = _finite_float32(_weighted_mlre_score(local, program_weights))
    n_anchor_genes = int(np.count_nonzero(clean_weights > 0.0))
    effective_n_anchor_genes = float(_effective_gene_number(normalized_weights))
    try:
        _, _, null_mean, null_std, null_meta = _matched_weight_permutation_pvalues_large(
            adata=adata_map,
            row_indices=context.global_indices,
            weights=normalized_weights,
            observed=activity,
            control_bins=control_bin_codes,
            scale=context.scale,
            sample_ids=context.sample_ids,
            micro_weights=context.micro_weights,
            count_layer=(
                context.expression_source[len("layers[") : -1]
                if context.expression_source.startswith("layers[")
                and context.expression_source.endswith("]")
                else None
            ),
            mlre_chunk_size=max(int(mlre_chunk_size), 1),
            n_perm=max(int(n_perm), 1),
            seed=int(seed),
            chunk_size=max(int(perm_chunk_size), 1),
        )
        denom = np.asarray(null_std, dtype=np.float64).reshape(-1)
        observed = np.asarray(activity, dtype=np.float64).reshape(-1)
        mean = np.asarray(null_mean, dtype=np.float64).reshape(-1)
        n = int(min(observed.size, mean.size, denom.size))
        null_z_unclipped = np.zeros((observed.size,), dtype=np.float64)
        if n > 0:
            valid_std = np.isfinite(denom[:n]) & (denom[:n] > 1e-8)
            valid_pos = np.where(valid_std)[0]
            null_z_unclipped[valid_pos] = (observed[valid_pos] - mean[valid_pos]) / denom[valid_pos]
        null_z_unclipped[~np.isfinite(null_z_unclipped)] = 0.0
        null_z = np.clip(null_z_unclipped, -float(z_max), float(z_max))
        rank = _samplewise_rank01(null_z_unclipped, context.sample_ids)
        null_meta = dict(null_meta)
        null_meta["status"] = "computed"
        status = "computed"
    except Exception as exc:
        raise RuntimeError(
            "[single_map]: matched-null large anchor computation failed "
            f"(n_spots={n_spots}, n_genes={n_genes}, n_anchor_genes={n_anchor_genes}, "
            f"n_perm={max(int(n_perm), 1)}, seed={int(seed)})."
        ) from exc

    return {
        "activity": activity.astype(np.float32, copy=False),
        "null_z": np.asarray(null_z, dtype=np.float32),
        "null_z_unclipped": np.asarray(null_z_unclipped, dtype=np.float32),
        "rank": np.asarray(rank, dtype=np.float32),
        "status": status,
        "n_anchor_genes": int(n_anchor_genes),
        "effective_n_anchor_genes": float(effective_n_anchor_genes),
        "matched_null_meta": null_meta,
    }


def _formal_anchor_metadata(
    anchor_scores: dict[str, Any],
    *,
    n_anchor_universe_genes: int,
    n_positive_anchor_genes: int,
) -> dict[str, Any]:
    matched_null_meta = anchor_scores.get("matched_null_meta", {})
    if not isinstance(matched_null_meta, dict):
        matched_null_meta = {}
    anchor_status = str(anchor_scores.get("status", "") or "")
    matched_null_status = str(
        matched_null_meta.get("status")
        or matched_null_meta.get("warning")
        or anchor_status
        or "not_computed"
    )
    matched_null_available = bool(matched_null_meta.get("available", anchor_status == "computed"))
    return {
        "anchor_source": ANCHOR_SOURCE,
        "anchor_universe": ANCHOR_UNIVERSE,
        "n_anchor_universe_genes": int(n_anchor_universe_genes),
        "n_positive_anchor_genes": int(n_positive_anchor_genes),
        "anchor_status": anchor_status,
        "anchor_effective_n_genes": float(anchor_scores.get("effective_n_anchor_genes", np.nan)),
        "matched_null_status": matched_null_status,
        "matched_null_available": bool(matched_null_available),
        "uses_full_gene_set_anchor": True,
    }


def _aggregate_anchor_metadata(sample_metadata: list[dict[str, Any]]) -> dict[str, Any]:
    anchor_statuses = sorted(
        {
            str(meta.get("anchor_status", ""))
            for meta in sample_metadata
            if str(meta.get("anchor_status", "")).strip()
        }
    )
    matched_null_statuses = sorted(
        {
            str(meta.get("matched_null_status", ""))
            for meta in sample_metadata
            if str(meta.get("matched_null_status", "")).strip()
        }
    )
    effective_values = np.asarray(
        [
            float(meta.get("anchor_effective_n_genes"))
            for meta in sample_metadata
            if meta.get("anchor_effective_n_genes") is not None
            and np.isfinite(float(meta.get("anchor_effective_n_genes")))
        ],
        dtype=float,
    )
    return {
        "anchor_source": ANCHOR_SOURCE,
        "anchor_universe": ANCHOR_UNIVERSE,
        "n_anchor_universe_genes": int(
            max((int(meta.get("n_anchor_universe_genes", 0)) for meta in sample_metadata), default=0)
        ),
        "n_positive_anchor_genes": int(
            max((int(meta.get("n_positive_anchor_genes", 0)) for meta in sample_metadata), default=0)
        ),
        "anchor_status": ";".join(anchor_statuses),
        "anchor_effective_n_genes": (
            float(np.mean(effective_values)) if effective_values.size else np.nan
        ),
        "matched_null_status": ";".join(matched_null_statuses),
        "matched_null_available": bool(
            sample_metadata and all(bool(meta.get("matched_null_available", False)) for meta in sample_metadata)
        ),
        "uses_full_gene_set_anchor": True,
    }


def _run_program_sample(
    *,
    program: str,
    program_token: str,
    sample_id: str,
    sample_profiles: np.ndarray,
    sample_gene_names: list[str],
    calibrated_anchor_values_sample: np.ndarray,
    anchor_metadata: dict[str, Any],
    control_bin_metadata: dict[str, Any],
    matched_null_metadata: dict[str, Any],
    min_finite_genes: int,
    rho_min: float,
    module_overlap_jaccard: float,
) -> dict[str, Any]:
    warnings: list[str] = []
    matched_null_context = {
        "anchor_metadata": dict(anchor_metadata),
        "control_bin_metadata": dict(control_bin_metadata),
        "matched_null_metadata": dict(matched_null_metadata),
    }
    n_input = len(sample_gene_names)
    n_spots = int(sample_profiles.shape[0]) if np.asarray(sample_profiles).ndim == 2 else 0
    anchor_sample = np.asarray(calibrated_anchor_values_sample, dtype=np.float32).reshape(-1)
    if anchor_sample.size != n_spots:
        warnings.append("program_anchor_reference_length_mismatch")
        anchor_sample = np.full((n_spots,), np.nan, dtype=np.float32)
    valid_mask = _finite_profile_columns(sample_profiles)
    finite_positions = np.flatnonzero(valid_mask).astype(int)
    if finite_positions.size < int(min_finite_genes):
        anchor_fields = dict(anchor_metadata)
        result = _empty_sample_result(
            program=program,
            program_token=program_token,
            sample_id=sample_id,
            n_input=n_input,
            n_finite=int(finite_positions.size),
            n_spots=n_spots,
            warning="fewer_than_map_single_min_genes_finite_mlre",
            calibrated_anchor_values_sample=anchor_sample,
            anchor_metadata=anchor_fields,
        )
        result["matched_null_context"] = matched_null_context
        return result

    profiles = np.asarray(sample_profiles[:, finite_positions], dtype=np.float32)
    genes = [sample_gene_names[int(pos)] for pos in finite_positions.tolist()]
    rho_program, a_pos, a_neg = _program_anchor_consistency(profiles, anchor_sample)
    corr = _signed_spearman_matrix(profiles)
    pos_adj, pos_edges = _build_program_conditioned_graph(corr, a_pos, rho_min=rho_min)
    neg_adj, neg_edges = _build_program_conditioned_graph(corr, a_neg, rho_min=rho_min)
    cross = _build_positive_negative_cross_antagonism(corr, a_pos, a_neg, rho_min=rho_min)

    positive_modules, pos_diag = _discover_active_module_bank(
        pos_adj,
        a_pos,
        overlap_jaccard=module_overlap_jaccard,
    )
    if int(pos_diag.get("n_redundant_active_modules", 0)) > 0:
        warnings.append(f"redundant_positive_active_modules_skipped={int(pos_diag['n_redundant_active_modules'])}")
    if int(pos_diag.get("n_modules_stopped_by_safety_guard", 0)) > 0:
        warnings.append("positive_module_growth_stopped_by_safety_guard")

    if not positive_modules:
        warnings.append("no_positive_active_module")
        edge_rows = _edge_records(
            program=program,
            program_token=program_token,
            sample_id=sample_id,
            genes=genes,
            corr=corr,
            pos_edges=pos_edges,
            neg_edges=neg_edges,
            cross=cross,
            a_pos=a_pos,
            a_neg=a_neg,
            rho_min=rho_min,
            positive_modules=[],
            negative_modules=[],
        )
        return {
            "positive": np.full((profiles.shape[0],), np.nan, dtype=np.float32),
            "negative": np.full((profiles.shape[0],), np.nan, dtype=np.float32),
            "contrast": np.full((profiles.shape[0],), np.nan, dtype=np.float32),
            "program_anchor": anchor_sample.astype(np.float32, copy=False),
            "edge_rows": edge_rows,
            "positive_module_rows": _module_table_rows(
                program=program,
                program_token=program_token,
                sample_id=sample_id,
                module_type="positive",
                modules=[],
                genes=genes,
                warning="no_positive_active_module",
            ),
            "negative_module_rows": _module_table_rows(
                program=program,
                program_token=program_token,
                sample_id=sample_id,
                module_type="negative",
                modules=[],
                genes=genes,
                warning="negative_active_module_not_searched_without_positive_bank",
            ),
            "module_gene_rows": [],
            "matched_null_context": matched_null_context,
            "summary_row": {
                "program": program,
                "program_token": program_token,
                "sample_id": sample_id,
                "n_input_genes": int(n_input),
                "n_finite_mlre_genes": int(finite_positions.size),
                "n_positive_edges": int(len(pos_edges)),
                "n_negative_edges": int(len(neg_edges)),
                "n_positive_active_modules": 0,
                "n_negative_active_modules": 0,
                "positive_module_weight_entropy": np.nan,
                "negative_module_weight_entropy": np.nan,
                "positive_effective_support_total": np.nan,
                "negative_effective_support_total": np.nan,
                "positive_top_module_size": 0,
                "positive_top_module_n_eff": np.nan,
                "positive_top_module_score": np.nan,
                "negative_top_module_size": 0,
                "negative_top_module_n_eff": np.nan,
                "negative_top_module_score": np.nan,
                "negative_antagonism_available": False,
                "module_stopped_by_safety_guard": bool(int(pos_diag.get("n_modules_stopped_by_safety_guard", 0)) > 0),
                "positive_projection_available": False,
                "negative_projection_available": False,
                "contrast_negative_missing": False,
                **dict(anchor_metadata),
                "warnings": ";".join(sorted(set(warnings))),
            },
        }

    positive_modules.sort(
        key=lambda module: (
            -(
                float(module.module_score)
                if np.isfinite(module.module_score)
                else -float("inf")
            ),
            -float(module.n_eff_genes) if np.isfinite(module.n_eff_genes) else 0.0,
            tuple(module.nodes),
        )
    )
    positive_projection = _module_bank_projection(
        profiles,
        positive_modules,
    )
    negative_modules_raw, neg_diag = _discover_active_module_bank(
        neg_adj,
        a_neg,
        overlap_jaccard=module_overlap_jaccard,
    )
    if int(neg_diag.get("n_redundant_active_modules", 0)) > 0:
        warnings.append(f"redundant_negative_active_modules_skipped={int(neg_diag['n_redundant_active_modules'])}")
    if int(neg_diag.get("n_modules_stopped_by_safety_guard", 0)) > 0:
        warnings.append("negative_module_growth_stopped_by_safety_guard")
    negative_modules = _finalize_negative_modules(negative_modules_raw, positive_modules, cross)

    if negative_modules:
        negative_modules.sort(
            key=lambda module: (
                -(
                    float(module.module_score)
                    if np.isfinite(module.module_score)
                    else -float("inf")
                ),
                -float(module.n_eff_genes) if np.isfinite(module.n_eff_genes) else 0.0,
                tuple(module.nodes),
            )
        )
    else:
        warnings.append("negative_active_module_not_found")
    negative_projection = _module_bank_projection(
        profiles,
        negative_modules,
    )
    negative_antagonism_available = bool(negative_modules)

    positive_projection_available = bool(np.any(np.isfinite(positive_projection)))
    negative_projection_available = bool(np.any(np.isfinite(negative_projection)))
    if positive_projection_available and negative_projection_available:
        contrast = positive_projection - negative_projection
        contrast_negative_missing = False
    elif positive_projection_available and not negative_projection_available:
        contrast = np.full((profiles.shape[0],), np.nan, dtype=np.float32)
        contrast_negative_missing = True
    else:
        contrast = np.full((profiles.shape[0],), np.nan, dtype=np.float32)
        contrast_negative_missing = False
    module_gene_rows: list[dict[str, Any]] = []
    _append_module_gene_rows(
        module_gene_rows,
        program=program,
        program_token=program_token,
        sample_id=sample_id,
        module_type="positive",
        modules=positive_modules,
        genes=genes,
        a_pos=a_pos,
        a_neg=a_neg,
        rho_program=rho_program,
    )
    _append_module_gene_rows(
        module_gene_rows,
        program=program,
        program_token=program_token,
        sample_id=sample_id,
        module_type="negative",
        modules=negative_modules,
        genes=genes,
        a_pos=a_pos,
        a_neg=a_neg,
        rho_program=rho_program,
    )

    edge_rows = _edge_records(
        program=program,
        program_token=program_token,
        sample_id=sample_id,
        genes=genes,
        corr=corr,
        pos_edges=pos_edges,
        neg_edges=neg_edges,
        cross=cross,
        a_pos=a_pos,
        a_neg=a_neg,
        rho_min=rho_min,
        positive_modules=positive_modules,
        negative_modules=negative_modules,
    )
    pos_top = positive_modules[0] if positive_modules else _invalid_active_module("none")
    neg_top = negative_modules[0] if negative_modules else _invalid_active_module("none")
    module_stopped = bool(
        any(bool(m.stopped_by_safety_guard) for m in [*positive_modules, *negative_modules])
        or int(pos_diag.get("n_modules_stopped_by_safety_guard", 0)) > 0
        or int(neg_diag.get("n_modules_stopped_by_safety_guard", 0)) > 0
    )
    summary_row = {
        "program": program,
        "program_token": program_token,
        "sample_id": sample_id,
        "n_input_genes": int(n_input),
        "n_finite_mlre_genes": int(finite_positions.size),
        "n_positive_edges": int(len(pos_edges)),
        "n_negative_edges": int(len(neg_edges)),
        "n_positive_active_modules": int(len(positive_modules)),
        "n_negative_active_modules": int(len(negative_modules)),
        "positive_module_weight_entropy": _module_weight_entropy(positive_modules),
        "negative_module_weight_entropy": _module_weight_entropy(negative_modules),
        "positive_effective_support_total": _effective_support_total(positive_modules),
        "negative_effective_support_total": _effective_support_total(negative_modules),
        "positive_top_module_size": int(len(pos_top.nodes)) if pos_top.found else 0,
        "positive_top_module_n_eff": float(pos_top.n_eff_genes) if pos_top.found else np.nan,
        "positive_top_module_score": float(pos_top.module_score) if pos_top.found else np.nan,
        "negative_top_module_size": int(len(neg_top.nodes)) if neg_top.found else 0,
        "negative_top_module_n_eff": float(neg_top.n_eff_genes) if neg_top.found else np.nan,
        "negative_top_module_score": float(neg_top.module_score) if neg_top.found else np.nan,
        "negative_antagonism_available": bool(negative_antagonism_available),
        "module_stopped_by_safety_guard": bool(module_stopped),
        "positive_projection_available": bool(positive_projection_available),
        "negative_projection_available": bool(negative_projection_available),
        "contrast_negative_missing": bool(contrast_negative_missing),
        **dict(anchor_metadata),
        "warnings": ";".join(sorted(set(warnings))),
    }
    return {
        "positive": positive_projection,
        "negative": negative_projection,
        "contrast": contrast,
        "program_anchor": anchor_sample.astype(np.float32, copy=False),
        "edge_rows": edge_rows,
        "positive_module_rows": _module_table_rows(
            program=program,
            program_token=program_token,
            sample_id=sample_id,
            module_type="positive",
            modules=positive_modules,
            genes=genes,
        ),
        "negative_module_rows": _module_table_rows(
            program=program,
            program_token=program_token,
            sample_id=sample_id,
            module_type="negative",
            modules=negative_modules,
            genes=genes,
            warning="negative_active_module_not_found" if not negative_modules else "",
        ),
        "module_gene_rows": module_gene_rows,
        "matched_null_context": matched_null_context,
        "summary_row": summary_row,
    }


def _write_tables(
    outdir: Path,
    *,
    edge_rows: list[dict[str, Any]],
    positive_module_rows: list[dict[str, Any]],
    negative_module_rows: list[dict[str, Any]],
    module_gene_rows: list[dict[str, Any]],
    sample_summary_rows: list[dict[str, Any]],
    program_summary_rows: list[dict[str, Any]],
) -> dict[str, str]:
    edge_df = pd.DataFrame(edge_rows).reindex(columns=EDGE_TABLE_COLUMNS)
    positive_module_df = pd.DataFrame(positive_module_rows).reindex(columns=MODULE_TABLE_COLUMNS)
    negative_module_df = pd.DataFrame(negative_module_rows).reindex(columns=MODULE_TABLE_COLUMNS)
    module_gene_df = pd.DataFrame(module_gene_rows).reindex(columns=MODULE_GENE_TABLE_COLUMNS)
    sample_summary_df = pd.DataFrame(sample_summary_rows).reindex(columns=SAMPLE_SUMMARY_COLUMNS)
    program_summary_df = pd.DataFrame(program_summary_rows).reindex(columns=PROGRAM_SUMMARY_COLUMNS)

    edge_path = outdir / EDGE_TABLE_FILENAME
    positive_module_path = outdir / POSITIVE_MODULE_TABLE_FILENAME
    negative_module_path = outdir / NEGATIVE_MODULE_TABLE_FILENAME
    module_gene_path = outdir / MODULE_GENE_TABLE_FILENAME
    sample_summary_path = outdir / SAMPLE_SUMMARY_FILENAME
    program_summary_path = outdir / PROGRAM_SUMMARY_FILENAME

    edge_df.to_csv(edge_path, index=False)
    positive_module_df.to_csv(positive_module_path, index=False)
    negative_module_df.to_csv(negative_module_path, index=False)
    module_gene_df.to_csv(module_gene_path, index=False)
    sample_summary_df.to_csv(sample_summary_path, index=False)
    program_summary_df.to_csv(program_summary_path, index=False)

    score_spec_path = _write_score_spec(outdir)
    return {
        "single_map_edge_table": str(edge_path),
        "single_map_positive_module_table": str(positive_module_path),
        "single_map_negative_module_table": str(negative_module_path),
        "single_map_module_gene_table": str(module_gene_path),
        "single_map_sample_summary": str(sample_summary_path),
        "single_map_program_summary": str(program_summary_path),
        "single_map_score_spec": str(score_spec_path),
    }


def run_single_map(
    adata: AnnData,
    cfg: Any,
    gene_set_df: Optional[pd.DataFrame] = None,
    output_dir=None,
    large_effect: bool = False,
) -> AnnData:
    start = time.perf_counter()
    warnings: list[str] = []
    outdir = _resolve_output_dir(cfg, output_dir)
    gene_set_path = getattr(cfg, "map_single_gene_set_path", None)
    if gene_set_df is None:
        if gene_set_path is None or str(gene_set_path).strip() == "":
            raise ValueError("[single_map]: cfg.map_single_gene_set_path is required when gene_set_df is None.")
        gene_set_df = _read_gene_set_table(gene_set_path)
    normalized_gene_set = _prepare_gene_set_df(gene_set_df, warnings)

    logger.info("[single_map]: MAP started | large_effect=%s", large_effect)
    if large_effect:
        adata_map = adata
    else:
        adata_map = adata.copy()
    var_lookup = _build_var_lookup(adata_map)
    unique_gene_keys = list(dict.fromkeys(normalized_gene_set["gene_upper"].astype(str).tolist()))
    common_keys: list[str] = []
    common_pos: list[int] = []
    for key in unique_gene_keys:
        if key in var_lookup:
            pos, _display = var_lookup[key]
            common_keys.append(str(key))
            common_pos.append(int(pos))
    full_st_pos = np.arange(int(adata_map.n_vars), dtype=np.int64)
    full_st_lookup = {str(key): int(pos_display[0]) for key, pos_display in var_lookup.items()}
    sample_ids, sample_col = _resolve_sample_ids(adata_map, cfg, warnings)
    output_obs_keys: list[str] = []

    min_finite_genes = max(_cfg_int(cfg, "map_single_min_genes", 10), 2)
    rho_min_raw = getattr(cfg, "signed_mcap_rho_min", SIGNED_RHO_MIN)
    try:
        rho_min = float(rho_min_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError("[single_map]: signed_mcap_rho_min must be a finite value in [0, 1].") from exc
    if not np.isfinite(rho_min) or not 0.0 <= rho_min <= 1.0:
        raise ValueError("[single_map]: signed_mcap_rho_min must be a finite value in [0, 1].")
    module_overlap_jaccard = float(
        np.clip(
            _cfg_float(cfg, "signed_mcap_module_overlap_jaccard", MODULE_OVERLAP_JACCARD),
            0.0,
            1.0,
        )
    )
    mlre_chunk = max(_cfg_int(cfg, "signed_mcap_mlre_chunk_size", MLRE_CHUNK_SIZE), 1)
    auxiliary_output_keys = [MAP_RANK_KEY, PROGRAM_ANCHOR_KEY]
    if not common_pos:
        _add_warning(warnings, "no_gene_set_genes_overlap_adata_var_names")

    seed = _cfg_int(cfg, "seed", 0)
    n_perm = max(_cfg_int(cfg, "map_single_n_perm", 50), 1)
    z_max = _cfg_float(cfg, "map_single_z_max", 5.0)
    if not np.isfinite(z_max) or z_max < 0.0:
        z_max = 5.0
    control_bins = max(_cfg_int(cfg, "map_single_control_bins", 20), 1)
    perm_chunk_size = max(_cfg_int(cfg, "map_single_perm_chunk_size", 20), 1)
    if large_effect:
        sample_contexts = _build_per_sample_contexts_large(
            adata_map,
            cfg,
            sample_ids,
            full_st_pos=full_st_pos,
            mlre_chunk=mlre_chunk,
            control_bins=control_bins,
            warnings=warnings,
        )
    else:
        sample_contexts = _build_per_sample_contexts(
            adata_map,
            cfg,
            sample_ids,
            full_st_pos=full_st_pos,
            mlre_chunk=mlre_chunk,
            control_bins=control_bins,
            warnings=warnings,
        )
    sample_context_details = {
        context.sample_token: {
            "sample_id": context.sample_id,
            "n_spots": int(context.global_indices.size),
            "expression_source": context.expression_source,
            "microdomain_graph_key": context.microdomain_graph_key,
            "mean_expr_mean": float(np.nanmean(context.mean_expr)) if context.mean_expr.size else None,
            "detect_rate_mean": float(np.nanmean(context.detect_rate)) if context.detect_rate.size else None,
            "var_expr_mean": float(np.nanmean(context.var_expr)) if context.var_expr.size else None,
            "control_bin_metadata": dict(context.control_bin_metadata),
            "matched_null_by_program": {},
        }
        for context in sample_contexts
    }

    programs = list(pd.Index(normalized_gene_set["program"].astype(str)).unique())
    token_by_program = _unique_program_tokens(programs)
    program_results: list[dict[str, Any]] = []
    all_edge_rows: list[dict[str, Any]] = []
    all_positive_module_rows: list[dict[str, Any]] = []
    all_negative_module_rows: list[dict[str, Any]] = []
    all_module_gene_rows: list[dict[str, Any]] = []
    all_sample_summary_rows: list[dict[str, Any]] = []
    program_summary_rows: list[dict[str, Any]] = []

    for program_offset, program in enumerate(programs):
        program_token = token_by_program[str(program)]
        program_df = normalized_gene_set[normalized_gene_set["program"].astype(str) == str(program)].copy()
        program_genes, program_positions, program_weights = _program_gene_positions(program_df, full_st_lookup)
        positive_vec = np.full((adata_map.n_obs,), np.nan, dtype=np.float32)
        negative_vec = np.full((adata_map.n_obs,), np.nan, dtype=np.float32)
        contrast_vec = np.full((adata_map.n_obs,), np.nan, dtype=np.float32)
        program_anchor_vec = np.full((adata_map.n_obs,), np.nan, dtype=np.float32)
        if len(program_positions) < min_finite_genes:
            _add_warning(
                warnings,
                f"program={program}: fewer than map_single_min_genes overlapping genes; active-module outputs are NaN.",
            )
        input_weight_full = np.zeros((int(adata_map.n_vars),), dtype=np.float32)
        input_sign_full = np.zeros((int(adata_map.n_vars),), dtype=np.float32)
        for local_pos, st_gene_idx in enumerate(program_positions):
            if 0 <= int(st_gene_idx) < input_weight_full.size:
                weight = float(program_weights[local_pos]) if local_pos < len(program_weights) else 1.0
                input_weight_full[int(st_gene_idx)] = max(weight if np.isfinite(weight) else 0.0, 0.0)
                input_sign_full[int(st_gene_idx)] = 1.0
        positive_anchor_weights_full = _directional_anchor_weights(
            input_weight_full,
            input_sign_full,
            direction="positive",
        )
        n_anchor_universe_genes = int(positive_anchor_weights_full.size)
        n_positive_anchor_genes = int(np.count_nonzero(positive_anchor_weights_full > EPS))
        sample_anchor_metadata: list[dict[str, Any]] = []
        for context in sample_contexts:
            idx = context.global_indices
            if large_effect:
                if idx.size == 0 or len(program_positions) == 0:
                    program_mlre = np.zeros((int(idx.size), 0), dtype=np.float32)
                else:
                    logger.info(
                        "[single_map]: large program MLRE sample_id=%s n_spots=%d "
                        "n_full_genes=%d n_program_genes=%d expression_gene_chunk_size=%d",
                        context.sample_id,
                        int(idx.size),
                        int(adata_map.n_vars),
                        int(len(program_positions)),
                        int(mlre_chunk),
                    )
                    program_mlre = _compute_mlre_matrix_large(
                        adata=adata_map,
                        row_indices=context.global_indices,
                        gene_positions=np.asarray(program_positions, dtype=np.int64),
                        scale=context.scale,
                        sample_ids=context.sample_ids,
                        micro_weights=context.micro_weights,
                        count_layer=getattr(cfg, "rep_count_layer", "counts"),
                        chunk_size=mlre_chunk,
                    )
                sample_profiles = program_mlre
                positive_anchor_scores = _compute_gene_set_anchor_scores_large(
                    adata_map=adata_map,
                    context=context,
                    program_mlre=program_mlre,
                    program_positions=np.asarray(program_positions, dtype=np.int64),
                    weights_full=positive_anchor_weights_full,
                    control_bin_codes=context.control_bin_codes,
                    n_perm=n_perm,
                    seed=int(seed) + 9001 * (int(program_offset) + 1) + 17,
                    perm_chunk_size=perm_chunk_size,
                    mlre_chunk_size=mlre_chunk,
                    z_max=z_max,
                )
            else:
                if idx.size == 0 or len(program_positions) == 0:
                    sample_profiles = np.zeros((int(idx.size), 0), dtype=np.float32)
                else:
                    sample_profiles = np.asarray(
                        context.mlre[:, program_positions],
                        dtype=np.float32,
                    )
                positive_anchor_scores = _compute_gene_set_anchor_scores(
                    mlre=context.mlre,
                    weights_full=positive_anchor_weights_full,
                    control_bin_codes=context.control_bin_codes,
                    n_perm=n_perm,
                    seed=int(seed) + 9001 * (int(program_offset) + 1) + 17,
                    perm_chunk_size=perm_chunk_size,
                    z_max=z_max,
                    sample_ids=context.sample_ids,
                )
            calibrated_anchor_values = np.asarray(
                positive_anchor_scores.get(
                    "null_z",
                    np.full((idx.size,), np.nan, dtype=np.float32),
                ),
                dtype=np.float32,
            ).reshape(-1)
            if calibrated_anchor_values.size != int(idx.size):
                raise RuntimeError(
                    "[single_map]: sample matched-null anchor length mismatch "
                    f"(program={program!r}, sample_id={context.sample_id!r}, "
                    f"expected={int(idx.size)}, observed={int(calibrated_anchor_values.size)})."
                )
            anchor_metadata = _formal_anchor_metadata(
                positive_anchor_scores,
                n_anchor_universe_genes=n_anchor_universe_genes,
                n_positive_anchor_genes=n_positive_anchor_genes,
            )
            sample_anchor_metadata.append(anchor_metadata)
            matched_null_metadata = positive_anchor_scores.get("matched_null_meta", {})
            if not isinstance(matched_null_metadata, dict):
                matched_null_metadata = {}
            sample_result = _run_program_sample(
                program=str(program),
                program_token=program_token,
                sample_id=context.sample_id,
                sample_profiles=sample_profiles,
                sample_gene_names=program_genes,
                calibrated_anchor_values_sample=calibrated_anchor_values,
                anchor_metadata=anchor_metadata,
                control_bin_metadata=context.control_bin_metadata,
                matched_null_metadata=matched_null_metadata,
                min_finite_genes=min_finite_genes,
                rho_min=rho_min,
                module_overlap_jaccard=module_overlap_jaccard,
            )
            sample_summary_row = sample_result["summary_row"]
            positive_vec[idx] = np.asarray(sample_result["positive"], dtype=np.float32)
            negative_vec[idx] = np.asarray(sample_result["negative"], dtype=np.float32)
            contrast_vec[idx] = np.asarray(sample_result["contrast"], dtype=np.float32)
            program_anchor_vec[idx] = np.asarray(
                sample_result["program_anchor"],
                dtype=np.float32,
            )
            all_edge_rows.extend(sample_result["edge_rows"])
            all_positive_module_rows.extend(sample_result["positive_module_rows"])
            all_negative_module_rows.extend(sample_result["negative_module_rows"])
            all_module_gene_rows.extend(sample_result["module_gene_rows"])
            all_sample_summary_rows.append(sample_summary_row)
            sample_context_details[context.sample_token]["matched_null_by_program"][program_token] = (
                sample_result["matched_null_context"]
            )
            if large_effect:
                del sample_profiles
                del program_mlre

        anchor_metadata = _aggregate_anchor_metadata(sample_anchor_metadata)

        program_sample_rows = [row for row in all_sample_summary_rows if row.get("program") == str(program)]
        n_pos_found = int(sum(int(row.get("n_positive_active_modules", 0) or 0) > 0 for row in program_sample_rows))
        n_neg_found = int(sum(int(row.get("n_negative_active_modules", 0) or 0) > 0 for row in program_sample_rows))
        n_neg_available = int(
            sum(bool(row.get("negative_projection_available", False)) for row in program_sample_rows)
        )
        program_result = {
            "program": str(program),
            "program_token": program_token,
            "positive": positive_vec,
            "negative": negative_vec,
            "contrast": contrast_vec,
            "program_anchor": program_anchor_vec,
            "program_weight": 1.0,
        }
        program_results.append(program_result)
        program_summary_rows.append(
            {
                "program": str(program),
                "program_token": program_token,
                "method": METHOD_NAME,
                "map_mode": MAP_MODE,
                "module_discovery_mode": SIGNED_MODULE_DISCOVERY_MODE,
                "positive_score_key": MAP_POSITIVE_KEY,
                "negative_score_key": MAP_NEGATIVE_KEY,
                "contrast_score_key": MAP_CONTRAST_KEY,
                "final_formula": FINAL_SCORE_FORMULA,
                "n_input_genes": int(program_df["gene_upper"].nunique()),
                "n_overlap_genes": int(len(program_positions)),
                "n_samples": int(len(sample_contexts)),
                "n_samples_positive_active_module_found": n_pos_found,
                "n_samples_negative_active_module_found": n_neg_found,
                "n_samples_negative_projection_available": n_neg_available,
                "n_samples_no_positive_active_module": int(len(sample_contexts) - n_pos_found),
                **dict(anchor_metadata),
            }
        )

    positive = _aggregate_program_vectors(program_results, "positive", int(adata_map.n_obs))
    negative = _aggregate_program_vectors(program_results, "negative", int(adata_map.n_obs))
    contrast = _aggregate_program_vectors(program_results, "contrast", int(adata_map.n_obs))
    _write_obs_vector(adata_map, MAP_POSITIVE_KEY, positive, output_obs_keys)
    _write_obs_vector(adata_map, MAP_NEGATIVE_KEY, negative, output_obs_keys)
    _write_obs_vector(adata_map, MAP_CONTRAST_KEY, contrast, output_obs_keys)
    _write_obs_vector(adata_map, MAP_RANK_KEY, _samplewise_rank01(positive, sample_ids), output_obs_keys)
    program_anchor = _aggregate_program_vectors(
        program_results,
        "program_anchor",
        int(adata_map.n_obs),
    )
    _write_obs_vector(
        adata_map,
        PROGRAM_ANCHOR_KEY,
        program_anchor,
        output_obs_keys,
    )

    paths = _write_tables(
        outdir,
        edge_rows=all_edge_rows,
        positive_module_rows=all_positive_module_rows,
        negative_module_rows=all_negative_module_rows,
        module_gene_rows=all_module_gene_rows,
        sample_summary_rows=all_sample_summary_rows,
        program_summary_rows=program_summary_rows,
    )
    paths["adata_map"] = str(outdir / "adata_map.h5ad")
    anchor_status_values = sorted(
        {
            str(row.get("anchor_status", ""))
            for row in program_summary_rows
            if str(row.get("anchor_status", "")).strip()
        }
    )
    anchor_effective_values = np.asarray(
        [
            float(row.get("anchor_effective_n_genes"))
            for row in program_summary_rows
            if row.get("anchor_effective_n_genes") is not None
            and np.isfinite(float(row.get("anchor_effective_n_genes")))
        ],
        dtype=float,
    )
    matched_null_status_values = sorted(
        {
            str(row.get("matched_null_status", ""))
            for row in program_summary_rows
            if str(row.get("matched_null_status", "")).strip()
        }
    )
    matched_null_available_values = [
        bool(row.get("matched_null_available", False))
        for row in program_summary_rows
    ]
    expression_sources = sorted({context.expression_source for context in sample_contexts})
    microdomain_graph_keys = sorted(
        {
            str(context.microdomain_graph_key)
            for context in sample_contexts
            if context.microdomain_graph_key is not None
        }
    )
    mean_expr_context_values = np.asarray(
        [float(np.nanmean(context.mean_expr)) for context in sample_contexts if context.mean_expr.size],
        dtype=float,
    )
    detect_rate_context_values = np.asarray(
        [float(np.nanmean(context.detect_rate)) for context in sample_contexts if context.detect_rate.size],
        dtype=float,
    )
    var_expr_context_values = np.asarray(
        [float(np.nanmean(context.var_expr)) for context in sample_contexts if context.var_expr.size],
        dtype=float,
    )
    summary = {
        "method": METHOD_NAME,
        "map_mode": MAP_MODE,
        "module_discovery_mode": SIGNED_MODULE_DISCOVERY_MODE,
        "negative_module_score_formula": NEGATIVE_MODULE_SCORE_FORMULA,
        "final_formula": FINAL_SCORE_FORMULA,
        "anchor_source": ANCHOR_SOURCE,
        "anchor_universe": ANCHOR_UNIVERSE,
        "program_anchor_reference": PROGRAM_ANCHOR_REFERENCE,
        "program_anchor_key": PROGRAM_ANCHOR_KEY,
        "n_anchor_universe_genes": int(adata_map.n_vars),
        "n_positive_anchor_genes": int(len(common_pos)),
        "anchor_status": ";".join(anchor_status_values),
        "anchor_effective_n_genes": (
            float(np.nanmean(anchor_effective_values))
            if anchor_effective_values.size
            else None
        ),
        "matched_null_status": ";".join(matched_null_status_values),
        "matched_null_available": bool(matched_null_available_values and all(matched_null_available_values)),
        "uses_full_gene_set_anchor": True,
        "positive_score_key": MAP_POSITIVE_KEY,
        "negative_score_key": MAP_NEGATIVE_KEY,
        "contrast_score_key": MAP_CONTRAST_KEY,
        "selected_score_key": MAP_POSITIVE_KEY,
        "single_map_score_key": MAP_POSITIVE_KEY,
        "map_scope": "samplewise",
        "sample_processing_scope": "per_sample",
        "sample_processing_scope_version": "per_sample_v1",
        "mlre_scope": "per_sample",
        "matched_null_matching_scope": "per_sample",
        "matched_null_evaluation_scope": "per_sample",
        "module_discovery_scope": "per_sample",
        "projection_scope": "per_sample",
        "cross_sample_parameter_sharing": False,
        "cross_sample_graph_edges": False,
        "negative_module_policy": (
            "negative_projection_is_nan_when_negative_active_module_unavailable; "
            "contrast_is_nan_when_negative_projection_unavailable"
        ),
        "formal_output_keys": [MAP_POSITIVE_KEY, MAP_NEGATIVE_KEY, MAP_CONTRAST_KEY],
        "auxiliary_output_keys": auxiliary_output_keys,
        "output_obs_keys": output_obs_keys,
        "sample_id_col": sample_col,
        "n_obs": int(adata_map.n_obs),
        "n_vars": int(adata_map.n_vars),
        "n_input_genes": int(normalized_gene_set["gene_upper"].nunique()),
        "n_overlap_genes": int(len(common_keys)),
        "expression_source": ";".join(expression_sources),
        "microdomain_graph_key": ";".join(microdomain_graph_keys) if microdomain_graph_keys else None,
        "sample_contexts": sample_context_details,
        "parameters": {
            "program_anchor_reference": PROGRAM_ANCHOR_REFERENCE,
            "anchor_source": ANCHOR_SOURCE,
            "anchor_universe": ANCHOR_UNIVERSE,
            "rho_min": float(rho_min),
            "edge_selection_rule": "samplewise_signed_spearman_threshold",
            "seed_policy": "nodewise_strongest_incident_edge_union",
            "support_factor_mode": "parameter_free_simpson_support_diversity",
            "support_factor_formula": "1_minus_sum_squared_normalized_module_gene_weights",
            "module_overlap_jaccard": float(module_overlap_jaccard),
            "negative_module_score_formula": NEGATIVE_MODULE_SCORE_FORMULA,
            "mlre_chunk_size": int(mlre_chunk),
            "min_finite_genes": int(min_finite_genes),
            "map_single_n_perm": int(n_perm),
            "map_single_control_bins": int(control_bins),
            "map_single_z_max": float(z_max),
            "seed": int(seed),
            "sample_seed_policy": "same_program_seed_for_each_sample",
            "map_single_perm_chunk_size": int(perm_chunk_size),
            "max_module_gene": None,
            "safety_guard": "module_growth_step_limit_only; stopped modules are not treated as confident modules",
        },
        "method_description": (
            "SpaRMap uses a full-ST-universe matched-null anchor to define program direction, discovers "
            "signed active modules within the input gene set, normalizes each module bank from the raw "
            "module scores, and produces spatial activity using active-module genes only."
        ),
        "mlre_feature_summary": {
            "scope": "per_sample",
            "mean_expr_mean": (
                float(np.nanmean(mean_expr_context_values)) if mean_expr_context_values.size else None
            ),
            "detect_rate_mean": (
                float(np.nanmean(detect_rate_context_values)) if detect_rate_context_values.size else None
            ),
            "var_expr_mean": (
                float(np.nanmean(var_expr_context_values)) if var_expr_context_values.size else None
            ),
        },
        "outputs": paths,
        "warnings": warnings,
        "runtime_seconds": float(time.perf_counter() - start),
        "enrichment_programs": {
            str(row["program"]): {
                "program_token": str(row["program_token"]),
                "positive_key": MAP_POSITIVE_KEY,
                "negative_key": MAP_NEGATIVE_KEY,
                "contrast_key": MAP_CONTRAST_KEY,
                "module_discovery_mode": SIGNED_MODULE_DISCOVERY_MODE,
            }
            for row in program_summary_rows
        },
    }
    adata_map.uns["single_map"] = _h5ad_safe_uns_value(summary)
    _save_json(summary, outdir / SUMMARY_JSON_FILENAME)
    logger.info(
        "[single_map]: active-module bank projection finished -> positive_key=%s negative_key=%s contrast_key=%s n_programs=%d runtime=%.2fs",
        MAP_POSITIVE_KEY,
        MAP_NEGATIVE_KEY,
        MAP_CONTRAST_KEY,
        len(programs),
        float(time.perf_counter() - start),
    )
    return adata_map
