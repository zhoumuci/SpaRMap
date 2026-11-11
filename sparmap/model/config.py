from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional, Union


@dataclass
class SpaRMAP_Config:

    # ARM-REP input and training.
    rep_gpu: str = "0"
    rep_hidden_dims: tuple[int, ...] = (128, 64)
    rep_dim_latent: int = 32
    rep_pca_dim: int = 50
    rep_lr: float = 1e-3
    rep_max_iter: int = 300
    rep_patience: int = 30
    rep_clip: float = 5.0
    rep_weight_decay: float = 1e-4
    rep_dropout: float = 0.10
    rep_w_recon: float = 1.0
    rep_w_local: float = 0.25
    rep_w_spatial_adj: float = 0.50
    rep_w_anti: float = 0.05
    rep_var_target: float = 1.0
    rep_graph_logit_scale: float = 5.0
    rep_graph_k: int = 10
    rep_graph_radius: float | None = None
    rep_negative_samples: int = 10
    rep_es_min_delta: float = 1e-4
    rep_count_layer: str | None = "counts"
    rep_coord_key: str = "spatial"
    rep_graph_sample_id_col: str = "sample_id"
    rep_harmony_batch_col: str | None = None
    rep_graph_strict_sample_safety: bool = True
    rep_use_hvg: bool = True
    rep_n_top_genes: int = 2000
    rep_hvg_flavor: str = "seurat_v3"
    rep_normalize_target_sum: float = 1e4
    rep_min_spots_per_gene: int = 3
    rep_min_genes_per_spot: int = 50
    rep_filter_spots: bool = True
    rep_filter_genes: bool = True
    rep_scale_max_value: float = 10.0
    rep_embed_key: str = "rep_z_main"
    rep_microdomain_embed_key: str = "rep_z_microdomain"
    rep_denoised_key: str = "denoised_mean"
    rep_graph_key: str = "spot_graph_map"
    rep_graph_spatial_key: str = "spot_graph_spatial"
    rep_microdomain_graph_key: str = "rep_microdomain_graph"
    rep_confidence_key: str = "rep_microdomain_confidence"
    rep_boundary_obs_key: str = "rep_boundary_uncertainty"
    arm_attn_temperature: float = 1.0
    arm_attn_entropy_floor_ratio: float = 0.55
    arm_structural_corr_scale: float = 0.30
    arm_structural_gate_floor: float = 0.05
    arm_w_depth_entropy: float = 0.0
    arm_w_view_entropy: float = 0.0
    arm_refine_topk: int = 14
    arm_refine_score_chunk_size: int = 200000

    # SpaRMap input, signed active modules, active-module projection, and matched null.
    spot_sample_id_col: str | None = "sample_id"
    map_single_gene_set_path: Optional[str] = None
    map_single_min_genes: int = 10
    map_use_rep_microdomain: bool = True
    signed_mcap_rho_min: float = 0.0
    signed_mcap_module_overlap_jaccard: float = 0.80
    signed_mcap_mlre_chunk_size: int = 512
    map_single_n_perm: int = 50
    map_single_control_bins: int = 20
    map_single_z_max: float = 5.0
    map_single_perm_chunk_size: int = 20

    # Shared runtime and cache controls.
    is_file_exist_loading: bool = False
    save_path: Optional[Union[str, Path]] = None
    seed: int = 42

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        for key, value in list(out.items()):
            if isinstance(value, Path):
                out[key] = str(value)
        return out
