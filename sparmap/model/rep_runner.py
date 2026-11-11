from __future__ import annotations

import hashlib
import re
from copy import deepcopy
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData, concat as concat_anndata

from sparmap._logging import get_logger
from sparmap.model.config import SpaRMAP_Config
from sparmap.model.rep import run_arm_rep
from sparmap.utils.map_utils import _resolve_microdomain_weights, _resolve_sample_ids
from sparmap.utils.rep_utils import count_cross_sample_edges

try:
    import torch
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None

logger = get_logger(__name__)
SPARMAP_INPUT_ROWID_KEY = "__sparmap_input_rowid"

def _validated_row_ids(adata: AnnData, *, upper_bound: int, label: str) -> np.ndarray:
    if SPARMAP_INPUT_ROWID_KEY not in adata.obs.columns:
        raise RuntimeError(
            f"[rep_runner]: {label} is missing obs['{SPARMAP_INPUT_ROWID_KEY}']."
        )
    rowid_series = adata.obs[SPARMAP_INPUT_ROWID_KEY]
    if int(rowid_series.shape[0]) != int(adata.n_obs):
        raise RuntimeError(
            f"[rep_runner]: {label} row ID length mismatch "
            f"(observed={rowid_series.shape[0]}, expected={adata.n_obs})."
        )
    if rowid_series.isna().any():
        raise RuntimeError(f"[rep_runner]: {label} row IDs contain missing values.")
    numeric = pd.to_numeric(rowid_series, errors="coerce").to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(numeric)) or not np.all(numeric == np.floor(numeric)):
        raise RuntimeError(f"[rep_runner]: {label} row IDs must be finite integers.")
    row_ids = numeric.astype(np.int64)
    if np.unique(row_ids).size != row_ids.size:
        raise RuntimeError(f"[rep_runner]: {label} row IDs contain duplicates.")
    if np.any(row_ids < 0) or np.any(row_ids >= int(upper_bound)):
        raise RuntimeError(
            f"[rep_runner]: {label} row IDs are outside [0, {int(upper_bound)})."
        )
    return row_ids

def _is_cuda_runtime_error(exc: BaseException) -> bool:
    msg = str(exc).lower()
    tokens = [
        "cuda error",
        "cuda out of memory",
        "cublas_status_alloc_failed",
        "cudnn",
        "cuda driver",
        "device-side assert",
    ]
    return any(token in msg for token in tokens)

def _empty_cuda_cache():
    if torch is not None and torch.cuda.is_available():
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

def _stale_rep_cache_error() -> RuntimeError:
    return RuntimeError(
        "[rep_runner]: unable to align REP-retained spots back to the current raw ST AnnData. "
        "Check that the saved REP and raw input refer to the same spots."
    )

def _copy_payload(value):
    if hasattr(value, "copy"):
        try:
            return value.copy()
        except TypeError:
            pass
    return deepcopy(value)

def _build_map_input_from_raw_and_rep(
    adata_raw: AnnData,
    adata_rep: AnnData,
    cfg: SpaRMAP_Config,
) -> tuple[AnnData, dict[str, Any]]:
    raw_row_ids = _validated_row_ids(
        adata_raw,
        upper_bound=int(adata_raw.n_obs),
        label="raw ST AnnData",
    )
    if not np.array_equal(raw_row_ids, np.arange(adata_raw.n_obs, dtype=np.int64)):
        raise RuntimeError("[rep_runner]: raw ST formal row IDs must equal the current positional row order.")
    rep_row_ids = _validated_row_ids(
        adata_rep,
        upper_bound=int(adata_raw.n_obs),
        label="REP AnnData",
    )
    adata_map_input = adata_raw[rep_row_ids, :].copy()
    aligned_by = f"obs[{SPARMAP_INPUT_ROWID_KEY}]"

    if adata_map_input.n_obs != adata_rep.n_obs:
        raise _stale_rep_cache_error()
    if not np.array_equal(
        np.asarray(adata_map_input.obs_names.astype(str)),
        np.asarray(adata_rep.obs_names.astype(str)),
    ):
        raise _stale_rep_cache_error()
    if not adata_map_input.var_names.equals(adata_raw.var_names):
        raise RuntimeError("[rep_runner]: MAP var_names must exactly match the raw ST gene universe.")

    for uns_key in ("microdomain_rep", "st_rep"):
        if uns_key in adata_rep.uns:
            adata_map_input.uns[uns_key] = deepcopy(adata_rep.uns[uns_key])

    for obsm_key in (str(cfg.rep_embed_key), str(cfg.rep_microdomain_embed_key)):
        if obsm_key in adata_rep.obsm:
            adata_map_input.obsm[obsm_key] = _copy_payload(adata_rep.obsm[obsm_key])

    for obsp_key in (
        str(cfg.rep_graph_key),
        str(cfg.rep_microdomain_graph_key),
        str(cfg.rep_graph_spatial_key),
    ):
        if obsp_key in adata_rep.obsp:
            adata_map_input.obsp[obsp_key] = _copy_payload(adata_rep.obsp[obsp_key])

    for obs_key in (
        str(cfg.rep_confidence_key),
        str(cfg.rep_boundary_obs_key),
        "rep_microdomain_radius",
    ):
        if obs_key in adata_rep.obs.columns:
            adata_map_input.obs[obs_key] = adata_rep.obs[obs_key].to_numpy(copy=True)

    summary = {
        "map_expression_base": "raw_st_subset_to_rep_retained_spots",
        "input_rowid_key": SPARMAP_INPUT_ROWID_KEY,
        "rep_retained_spot_alignment": aligned_by,
        "n_raw_genes": int(adata_raw.n_vars),
        "n_rep_genes": int(adata_rep.n_vars),
        "n_map_input_genes": int(adata_map_input.n_vars),
        "n_raw_spots": int(adata_raw.n_obs),
        "n_rep_retained_spots": int(adata_rep.n_obs),
    }
    return adata_map_input, summary

def _sample_cache_token(sample_id: str) -> str:
    raw = str(sample_id)
    base = re.sub(r"[^0-9A-Za-z._-]+", "_", raw.strip()).strip("._-") or "sample"
    digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()[:12]
    return f"{base[:64]}-{digest}"


def _run_rep(adata: AnnData, cfg: SpaRMAP_Config) -> AnnData:
    logger.info("[rep_runner]: active_rep='arm_rep' -> start REP stage.")
    try:
        adata_rep = run_arm_rep(adata=adata, cfg=cfg, device_override=cfg.rep_gpu)
    except RuntimeError as exc:
        if str(cfg.rep_gpu).strip() not in {"", "cpu"} and _is_cuda_runtime_error(exc):
            logger.warning("[rep_runner]: formal ARM-REP failed on CUDA, retry the same Torch model on CPU.")
            _empty_cuda_cache()
            adata_rep = run_arm_rep(adata=adata, cfg=cfg, device_override="cpu")
        else:
            raise
    return adata_rep

def _run_samplewise_arm_single_map_rep(
    adata_raw: AnnData,
    cfg: SpaRMAP_Config,
) -> tuple[AnnData, dict[str, Any]]:
    sample_col = str(getattr(cfg, "rep_graph_sample_id_col", "sample_id")).strip()
    if sample_col == "" or sample_col not in adata_raw.obs.columns:
        raise KeyError(
            "[rep_runner]: sample-wise arm_rep -> single_map requires "
            f"obs['{sample_col or 'sample_id'}']."
        )
    sample_series = adata_raw.obs[sample_col].astype("string").str.strip()
    if sample_series.isna().any() or (sample_series == "").any():
        raise ValueError(
            "[rep_runner]: sample-wise arm_rep -> single_map requires non-empty sample IDs in "
            f"obs['{sample_col}']."
        )
    sample_ids = sample_series.astype(str).to_numpy(copy=False)
    unique_samples = list(dict.fromkeys(sample_ids.tolist()))
    if not unique_samples:
        raise ValueError("[rep_runner]: sample-wise arm_rep -> single_map received no samples.")

    sample_map_inputs: list[AnnData] = []
    sample_summaries: dict[str, dict[str, Any]] = {}
    rep_gene_counts: dict[str, int] = {}

    for sample_id in unique_samples:
        sample_idx = np.flatnonzero(sample_ids == str(sample_id)).astype(np.int64, copy=False)
        sample_raw = adata_raw[sample_idx, :].copy()
        global_row_ids = _validated_row_ids(
            sample_raw,
            upper_bound=int(adata_raw.n_obs),
            label=f"raw sample {sample_id!r}",
        )
        sample_raw.obs[SPARMAP_INPUT_ROWID_KEY] = np.arange(sample_raw.n_obs, dtype=np.int64)

        sample_token = _sample_cache_token(str(sample_id))
        sample_cfg = deepcopy(cfg)
        sample_cfg.seed = int(getattr(cfg, "seed", 0))
        logger.info(
            "[rep_runner]: sample-wise ARM REP -> "
            f"sample_id={sample_id!r}, seed={sample_cfg.seed}"
        )
        sample_rep = _run_rep(sample_raw, sample_cfg)
        sample_map_input, sample_summary = _build_map_input_from_raw_and_rep(
            sample_raw,
            sample_rep,
            sample_cfg,
        )
        retained_local_row_ids = _validated_row_ids(
            sample_map_input,
            upper_bound=int(global_row_ids.size),
            label=f"REP-retained sample {sample_id!r}",
        )
        sample_map_input.obs[SPARMAP_INPUT_ROWID_KEY] = global_row_ids[retained_local_row_ids]
        if not sample_map_input.var_names.equals(adata_raw.var_names):
            raise RuntimeError(
                "[rep_runner]: per-sample map input changed the raw ST gene universe; "
                "sample-wise concatenation requires identical var_names."
            )

        sample_summary.update(
            {
                "sample_id": str(sample_id),
                "sample_token": sample_token,
                "seed": int(sample_cfg.seed),
                "sample_processing_scope": "per_sample",
                "rep_training_scope": "per_sample",
                "cross_sample_parameter_sharing": False,
                "cross_sample_graph_edges": False,
            }
        )
        sample_summaries[sample_token] = sample_summary
        rep_gene_counts[sample_token] = int(sample_rep.n_vars)
        sample_map_inputs.append(sample_map_input)

    adata_map_input = concat_anndata(
        sample_map_inputs,
        axis=0,
        join="inner",
        merge="same",
        uns_merge=None,
        pairwise=True,
        index_unique=None,
    )
    if not adata_map_input.var_names.equals(adata_raw.var_names):
        raise RuntimeError(
            "[rep_runner]: concatenated per-sample map input changed the raw ST gene universe."
        )
    final_row_ids = np.asarray(
        adata_map_input.obs[SPARMAP_INPUT_ROWID_KEY].to_numpy(),
        dtype=np.int64,
    )
    if (
        final_row_ids.size != adata_map_input.n_obs
        or np.unique(final_row_ids).size != final_row_ids.size
        or np.any(final_row_ids < 0)
        or np.any(final_row_ids >= adata_raw.n_obs)
    ):
        raise _stale_rep_cache_error()
    restore_order = np.argsort(final_row_ids, kind="stable")
    adata_map_input = adata_map_input[restore_order, :].copy()
    if bool(getattr(cfg, "map_use_rep_microdomain", True)):
        graph_sample_ids, _ = _resolve_sample_ids(adata_map_input, cfg, [])
        _resolve_microdomain_weights(adata_map_input, cfg, graph_sample_ids, [])

    rep_gene_values = np.asarray(list(rep_gene_counts.values()), dtype=np.int64)
    summary = {
        "map_expression_base": "raw_st_subset_to_per_sample_rep_retained_spots",
        "input_rowid_key": SPARMAP_INPUT_ROWID_KEY,
        "rep_retained_spot_alignment": f"per_sample_obs[{SPARMAP_INPUT_ROWID_KEY}]",
        "sample_id_col": sample_col,
        "sample_processing_scope": "per_sample",
        "rep_training_scope": "per_sample",
        "cross_sample_parameter_sharing": False,
        "cross_sample_graph_edges": False,
        "rep_seed_policy": "same_cfg_seed_for_each_sample",
        "rep_seed": int(getattr(cfg, "seed", 0)),
        "n_training_samples": int(len(unique_samples)),
        "n_raw_genes": int(adata_raw.n_vars),
        "n_map_input_genes": int(adata_map_input.n_vars),
        "n_raw_spots": int(adata_raw.n_obs),
        "n_rep_retained_spots": int(adata_map_input.n_obs),
        "n_rep_genes_min": int(np.min(rep_gene_values)),
        "n_rep_genes_max": int(np.max(rep_gene_values)),
        "n_rep_genes_per_sample": rep_gene_counts,
        "per_sample_map_input_summary": sample_summaries,
    }
    return adata_map_input, summary


def _run_multisample_arm_single_map_rep(
    adata_raw: AnnData,
    cfg: SpaRMAP_Config,
) -> tuple[AnnData, dict[str, Any]]:
    sample_col = str(getattr(cfg, "rep_graph_sample_id_col", "sample_id")).strip()
    if sample_col == "" or sample_col not in adata_raw.obs.columns:
        raise KeyError(
            "[rep_runner]: multi-sample arm_rep -> single_map requires "
            f"obs['{sample_col or 'sample_id'}']."
        )
    sample_series = adata_raw.obs[sample_col].astype("string").str.strip()
    if sample_series.isna().any() or (sample_series == "").any():
        raise ValueError(
            "[rep_runner]: multi-sample arm_rep -> single_map requires non-empty sample IDs in "
            f"obs['{sample_col}']."
        )
    sample_ids = sample_series.astype(str).to_numpy(copy=False)
    unique_samples = list(dict.fromkeys(sample_ids.tolist()))
    if len(unique_samples) < 2:
        raise ValueError(
            "[rep_runner]: multi-sample arm_rep -> single_map requires at least two unique samples."
        )

    raw_row_ids = _validated_row_ids(
        adata_raw,
        upper_bound=int(adata_raw.n_obs),
        label="raw multi-sample ST AnnData",
    )
    if not np.array_equal(raw_row_ids, np.arange(adata_raw.n_obs, dtype=np.int64)):
        raise RuntimeError(
            "[rep_runner]: raw multi-sample ST formal row IDs must equal the current positional row order."
        )

    multi_cfg = deepcopy(cfg)
    multi_cfg._rep_execution_mode = "multi"
    logger.info(
        "[rep_runner]: multi-sample ARM REP -> n_samples=%d, seed=%d",
        len(unique_samples),
        int(multi_cfg.seed),
    )
    adata_rep = run_arm_rep(
        adata=adata_raw,
        cfg=multi_cfg,
        device_override=multi_cfg.rep_gpu,
        training_scope="multi",
    )

    rep_row_ids = _validated_row_ids(
        adata_rep,
        upper_bound=int(adata_raw.n_obs),
        label="multi-sample REP AnnData",
    )
    if rep_row_ids.size > 1 and not np.all(rep_row_ids[1:] > rep_row_ids[:-1]):
        raise RuntimeError("[rep_runner]: multi-sample REP output did not preserve input row order.")
    retained_sample_ids = sample_ids[rep_row_ids]
    rep_sample_ids = adata_rep.obs[sample_col].astype("string").str.strip()
    if rep_sample_ids.isna().any() or not np.array_equal(
        rep_sample_ids.astype(str).to_numpy(copy=False),
        retained_sample_ids,
    ):
        raise RuntimeError("[rep_runner]: multi-sample REP output did not preserve original sample IDs.")

    graph_cross_sample_edges: dict[str, int] = {}
    for graph_key in dict.fromkeys(
        (
            str(multi_cfg.rep_graph_spatial_key),
            str(multi_cfg.rep_graph_key),
            str(multi_cfg.rep_microdomain_graph_key),
        )
    ):
        if graph_key not in adata_rep.obsp:
            raise RuntimeError(f"[rep_runner]: multi-sample REP output is missing obsp[{graph_key!r}].")
        graph_cross_sample_edges[graph_key] = int(
            count_cross_sample_edges(adata_rep.obsp[graph_key], retained_sample_ids)["cross_sample_edges"]
        )
    if any(value != 0 for value in graph_cross_sample_edges.values()):
        raise RuntimeError("[rep_runner]: multi-sample REP output contains cross-sample graph edges.")

    adata_map_input, map_input_summary = _build_map_input_from_raw_and_rep(
        adata_raw,
        adata_rep,
        multi_cfg,
    )
    if bool(getattr(multi_cfg, "map_use_rep_microdomain", True)):
        graph_sample_ids, _ = _resolve_sample_ids(adata_map_input, multi_cfg, [])
        _resolve_microdomain_weights(adata_map_input, multi_cfg, graph_sample_ids, [])

    summary = {
        **map_input_summary,
        "sample_id_col": sample_col,
        "execution_mode": "multi",
        "sample_processing_scope": "multi_sample_shared",
        "rep_training_scope": "multi_sample_shared",
        "training_scope": "multi_sample_shared",
        "shared_parameters_across_samples": True,
        "cross_sample_parameter_sharing": True,
        "cross_sample_graph_edges": 0,
        "cross_sample_graph_edges_by_key": graph_cross_sample_edges,
        "integration_method": "Harmony",
        "integration_space": "shared_PCA",
        "rep_input_basis": "Harmony_corrected_shared_PCA",
        "rep_seed": int(multi_cfg.seed),
        "n_input_samples": int(len(unique_samples)),
        "n_independent_training_runs": 1,
        "n_raw_genes": int(adata_raw.n_vars),
        "n_rep_genes": int(adata_rep.n_vars),
        "n_map_input_genes": int(adata_map_input.n_vars),
        "n_raw_spots": int(adata_raw.n_obs),
        "n_rep_retained_spots": int(adata_map_input.n_obs),
    }
    return adata_map_input, summary
