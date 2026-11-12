"""SpaRMap Python and command-line entry points."""

from __future__ import annotations

import argparse
import gc
from pathlib import Path

import anndata as ad
import numpy as np

from sparmap._logging import flush_logging, get_logger, setup_logging, shutdown_logging
from sparmap.model.config import SpaRMAP_Config
from sparmap.model.map import run_single_map
from sparmap.model.rep import ARM_MICRODOMAIN_GRAPH_METHOD
from sparmap.model.rep_runner import (
    SPARMAP_INPUT_ROWID_KEY,
    _build_map_input_from_raw_and_rep,
    _run_multisample_arm_single_map_rep,
    _run_samplewise_arm_single_map_rep,
    _validated_row_ids,
)
from sparmap.utils.map_utils import _resolve_microdomain_weights, _resolve_sample_ids
from sparmap.utils.rep_utils import count_cross_sample_edges
from sparmap.utils.utils import h5ad_safe_uns_value

logger = get_logger(__name__)


def _validate_parameters(mode, sample_col, harmony_batch_col, rep_graph_k, arm_refine_topk,
                         signed_mcap_rho_min, signed_mcap_module_overlap_jaccard):
    if mode not in {"single", "multi"}:
        raise ValueError("mode must be 'single' or 'multi'.")
    if not isinstance(sample_col, str) or not sample_col.strip():
        raise ValueError("sample_col must be a non-empty obs column name.")
    if harmony_batch_col is not None and (
        not isinstance(harmony_batch_col, str) or not harmony_batch_col.strip()
    ):
        raise ValueError("harmony_batch_col must be a non-empty obs column name or None.")
    for name, value in (("rep_graph_k", rep_graph_k), ("arm_refine_topk", arm_refine_topk)):
        if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)) or value < 1:
            raise ValueError(f"{name} must be a positive integer.")
    for name, value in (
        ("signed_mcap_rho_min", signed_mcap_rho_min),
        ("signed_mcap_module_overlap_jaccard", signed_mcap_module_overlap_jaccard),
    ):
        if not np.isfinite(float(value)) or not 0.0 <= float(value) <= 1.0:
            raise ValueError(f"{name} must be finite and in [0, 1].")


def _validate_input(raw: ad.AnnData, cfg: SpaRMAP_Config, mode: str) -> None:
    if cfg.rep_count_layer not in raw.layers:
        raise KeyError("Input H5AD must contain raw counts in layers['counts'].")
    if cfg.rep_coord_key not in raw.obsm:
        raise KeyError("Input H5AD must contain coordinates in obsm['spatial'].")
    sample_col = cfg.rep_graph_sample_id_col
    if sample_col not in raw.obs:
        raise KeyError(f"Input H5AD is missing obs[{sample_col!r}].")
    ids = raw.obs[sample_col].astype("string").str.strip()
    if ids.isna().any() or (ids == "").any():
        raise ValueError(f"obs[{sample_col!r}] contains missing or empty sample IDs.")
    if mode == "multi":
        if ids.nunique() < 2:
            raise ValueError("mode='multi' requires at least two sample IDs.")
        batch_col = cfg.rep_harmony_batch_col
        if batch_col not in raw.obs:
            raise KeyError(f"Input H5AD is missing Harmony batch obs[{batch_col!r}].")
        batches = raw.obs[batch_col].astype("string").str.strip()
        if batches.isna().any() or (batches == "").any():
            raise ValueError(f"obs[{batch_col!r}] contains missing or empty Harmony batch IDs.")


def _validate_rep(rep: ad.AnnData, raw: ad.AnnData, cfg: SpaRMAP_Config, mode: str) -> None:
    """Validate REP metadata, embeddings, graphs and spot alignment."""
    if not isinstance(rep.uns.get("st_rep"), dict):
        raise ValueError("Saved REP is missing uns['st_rep'] metadata.")
    if not isinstance(rep.uns.get("microdomain_rep"), dict):
        raise ValueError("Saved REP is missing uns['microdomain_rep'] metadata.")
    observed_mode = rep.uns["st_rep"].get("execution_mode")
    if observed_mode != mode:
        raise ValueError(f"Saved REP execution mode is {observed_mode!r}; requested {mode!r}.")
    rows = _validated_row_ids(rep, upper_bound=raw.n_obs, label="saved REP")
    if not np.array_equal(np.asarray(rep.obs_names.astype(str)),
                          np.asarray(raw.obs_names[rows].astype(str))):
        raise ValueError("Saved REP spots do not align with the input H5AD.")
    sample_col = cfg.rep_graph_sample_id_col
    if sample_col not in rep.obs or not np.array_equal(
        rep.obs[sample_col].astype("string").str.strip().to_numpy(),
        raw.obs[sample_col].iloc[rows].astype("string").str.strip().to_numpy(),
    ):
        raise ValueError("Saved REP sample IDs do not align with the input H5AD.")
    for key in (cfg.rep_embed_key, cfg.rep_microdomain_embed_key):
        if key not in rep.obsm:
            raise ValueError(f"Saved REP is missing obsm[{key!r}].")
        values = np.asarray(rep.obsm[key])
        if values.ndim != 2 or values.shape[0] != rep.n_obs or not np.all(np.isfinite(values)):
            raise ValueError(f"Saved REP obsm[{key!r}] has invalid dimensions or values.")
    sample_ids, _ = _resolve_sample_ids(rep, cfg, [])
    for key in dict.fromkeys((cfg.rep_graph_spatial_key, cfg.rep_graph_key,
                              cfg.rep_microdomain_graph_key)):
        if key not in rep.obsp or rep.obsp[key].shape != (rep.n_obs, rep.n_obs):
            raise ValueError(f"Saved REP is missing a valid obsp[{key!r}] graph.")
        if count_cross_sample_edges(rep.obsp[key], sample_ids)["cross_sample_edges"]:
            raise ValueError(f"Saved REP obsp[{key!r}] contains cross-sample edges.")
    _resolve_microdomain_weights(rep, cfg, sample_ids, [])


def _annotate_rep(rep: ad.AnnData, summary: dict, cfg: SpaRMAP_Config, mode: str) -> None:
    rep.uns.setdefault("st_rep", {}).update(
        active_rep="arm_rep", execution_mode=mode,
        training_scope="per_sample" if mode == "single" else "multi_sample_shared",
    )
    rep.uns.setdefault("microdomain_rep", {}).update(
        microdomain_graph_method=ARM_MICRODOMAIN_GRAPH_METHOD,
        microdomain_graph_key=cfg.rep_microdomain_graph_key,
        microdomain_scope="per_sample",
    )
    rep.uns["sparmap_rep"] = h5ad_safe_uns_value(summary)


def sparmap(
    raw_h5ad: str | Path,
    gene_set: str | Path,
    outdir: str | Path = "result",
    *,
    mode: str = "single",
    sample_col: str = "sample_id",
    harmony_batch_col: str | None = None,
    rep_graph_k: int = 10,
    arm_refine_topk: int = 14,
    signed_mcap_rho_min: float = 0.0,
    signed_mcap_module_overlap_jaccard: float = 0.80,
    large_effect: bool = False,
) -> ad.AnnData:
    """Run REP and MAP and return the final mapped AnnData."""
    _validate_parameters(mode, sample_col, harmony_batch_col, rep_graph_k, arm_refine_topk,
                         signed_mcap_rho_min, signed_mcap_module_overlap_jaccard)
    raw_path = Path(raw_h5ad).expanduser().resolve()
    genes_path = Path(gene_set).expanduser().resolve()
    for label, path in (("Input H5AD", raw_path), ("Gene-set file", genes_path)):
        if not path.is_file():
            raise FileNotFoundError(f"{label} does not exist: {path}")
    output = Path(outdir).expanduser().resolve()
    rep_dir, map_dir = output / "rep", output / "map"
    rep_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    cfg = SpaRMAP_Config(
        save_path=str(output), map_single_gene_set_path=str(genes_path),
        spot_sample_id_col=sample_col, rep_graph_sample_id_col=sample_col,
        rep_harmony_batch_col=sample_col if harmony_batch_col is None else harmony_batch_col,
        rep_graph_k=int(rep_graph_k), arm_refine_topk=int(arm_refine_topk),
        signed_mcap_rho_min=float(signed_mcap_rho_min),
        signed_mcap_module_overlap_jaccard=float(signed_mcap_module_overlap_jaccard),
    )
    cfg._rep_execution_mode = mode
    setup_logging(output)
    rep_path = rep_dir / "adata_rep.h5ad"
    try:
        logger.info("SpaRMap started | mode=%s raw_h5ad=%s count_layer=counts outdir=%s",
                    mode, raw_path, output)
        raw = ad.read_h5ad(raw_path)
        _validate_input(raw, cfg, mode)
        raw.obs[SPARMAP_INPUT_ROWID_KEY] = np.arange(raw.n_obs, dtype=np.int64)
        if rep_path.is_file():
            logger.info("Reusing REP: %s", rep_path)
            saved_rep = ad.read_h5ad(rep_path)
            _validate_rep(saved_rep, raw, cfg, mode)
            adata_input, summary = _build_map_input_from_raw_and_rep(raw, saved_rep, cfg)
            summary.update(rep_source="existing_rep", rep_path=str(rep_path))
            del saved_rep
        else:
            raw_for_rep = raw.copy()
            if mode == "single":
                adata_input, summary = _run_samplewise_arm_single_map_rep(raw_for_rep, cfg)
            else:
                adata_input, summary = _run_multisample_arm_single_map_rep(raw_for_rep, cfg)
            del raw_for_rep
            _annotate_rep(adata_input, summary, cfg, mode)
            _validate_rep(adata_input, raw, cfg, mode)
            adata_input.write_h5ad(rep_path)
            summary.update(rep_source="run_rep", rep_path=str(rep_path))
            logger.info("Saved final REP: %s", rep_path)
        if large_effect:
            del raw
            gc.collect()
        result = run_single_map(adata_input, cfg=cfg, output_dir=map_dir,
                                large_effect=large_effect)
        result.uns.setdefault("single_map", {}).setdefault("run_single_input", {}).update(summary)
        result.write_h5ad(map_dir / "adata_map.h5ad")
        logger.info("SpaRMap finished | output=%s n_spots=%d n_genes=%d",
                    map_dir / "adata_map.h5ad", result.n_obs, result.n_vars)
        return result
    except Exception:
        logger.exception("SpaRMap failed")
        raise
    finally:
        flush_logging()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Spatial representation learning and gene-set mapping from raw counts.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        epilog=("Outputs: OUTDIR/rep/adata_rep.h5ad, OUTDIR/map/adata_map.h5ad "
                "and MAP result tables."),
    )
    inputs = parser.add_argument_group("Input and output")
    inputs.add_argument("--raw-h5ad", required=True, metavar="PATH",
                        help="Input H5AD containing raw counts in layers['counts'] and coordinates in obsm['spatial'].")
    inputs.add_argument("--gene-set", required=True, metavar="PATH",
                        help="Gene-set CSV/TSV with a gene column and optional program and weight columns.")
    inputs.add_argument("--outdir", default="result", metavar="DIR",
                        help="Output directory; reuse rep/adata_rep.h5ad when present.")
    execution = parser.add_argument_group("Samples and execution")
    execution.add_argument("--mode", choices=("single", "multi"), default="single",
                           help="single: train REP independently for each sample; multi: shared PCA/Harmony and REP. MAP stays sample-local.")
    execution.add_argument("--sample-col", default="sample_id", metavar="COLUMN",
                           help="obs column identifying spatial samples; required in both modes. multi needs at least two samples.")
    execution.add_argument("--harmony-batch-col", default=None, metavar="COLUMN",
                           help="obs column defining Harmony batches in multi mode; defaults to --sample-col.")
    execution.add_argument("--large-effect", action="store_true", default=False,
                           help="Use memory-efficient MAP execution for large datasets.")
    model = parser.add_argument_group("Graph and module parameters")
    model.add_argument("--rep-graph-k", type=int, default=10, metavar="K",
                       help="Number of spatial KNN neighbors before graph symmetrization.")
    model.add_argument("--arm-refine-topk", type=int, default=14, metavar="K",
                       help="Maximum number of neighbors retained per microdomain.")
    model.add_argument("--signed-mcap-rho-min", type=float, default=0.0, metavar="RHO",
                       help="Module-graph correlation threshold in [0,1]; also controls the negative-correlation threshold for positive/negative relations.")
    model.add_argument("--signed-mcap-module-overlap-jaccard", type=float, default=0.80, metavar="JACCARD",
                       help="Module redundancy threshold in [0,1]; discard a candidate when overlap with a retained module is greater than this value.")
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    try:
        sparmap(**vars(args))
    finally:
        shutdown_logging()


if __name__ == "__main__":
    main()
