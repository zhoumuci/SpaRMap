from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
import pandas as pd
from anndata import AnnData
from scipy.sparse import csr_matrix, issparse

from sparmap.utils.rep_utils import (
    TORCH_AVAILABLE,
    _cosine_similarity_rows,
    _csr_float32,
    _csr_to_torch_sparse,
    _edge_arrays,
    _graph_loss,
    _local_consistency_loss,
    _materialize_denoised_expression,
    _multisample_log_scale_pca_harmony,
    _normalize_sample_ids,
    _preselect_genes_sample_safe,
    _records_to_columnar,
    _resolve_device,
    _row_normalize_csr,
    _sample_expression_edge_weight,
    _sample_negative_edges,
    _sample_safe_filter_masks,
    _samplewise_log_scale_pca,
    _set_seed,
    _summary_stats,
    _variance_regularization,
    build_samplewise_spatial_graph,
    count_cross_sample_edges,
)

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
except ImportError:  # pragma: no cover - depends on runtime environment
    torch = None
    nn = None
    F = None

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None

from sparmap._logging import get_logger

logger = get_logger(__name__)


ARM_HISTORY_FIELDS = (
    "epoch",
    "loss_total",
    "loss_recon",
    "loss_local",
    "loss_adj",
    "loss_anti",
    "loss_depth_entropy",
    "loss_view_entropy",
)

ARM_FUSION_MODE = "anchored_correction"
ARM_STRUCTURAL_GATE_NAMES = ["depth", "spatial", "residual"]
ARM_MICRODOMAIN_GRAPH_METHOD = "samplewise_two_hop_continuous_affinity_topk"
AFFINITY_METHOD = "latent_expression_spatial_mean_affinity"
NEIGHBOR_SELECTION_METHOD = "fixed_topk_affinity"
AFFINITY_FORMULA = "(latent_sim + expr_sim + spatial_proximity) / 3"
WEIGHT_FORMULA = "row_softmax_over_selected_affinity"
CONFIDENCE_FORMULA = "weighted_sigmoid_of_affinity_minus_midpoint"


def _validate_structural_corr_scale(value: Any) -> float:
    scale = float(value)
    if not np.isfinite(scale) or scale < 0.0:
        raise ValueError("[arm_rep]: arm_structural_corr_scale must be a finite non-negative float.")
    return scale


def _validate_structural_gate_floor(value: Any) -> float:
    floor = float(value)
    if not np.isfinite(floor) or floor < 0.0 or floor >= 1.0:
        raise ValueError("[arm_rep]: arm_structural_gate_floor must be finite and in [0, 1).")
    return floor


def _row_sum_stats(graph: csr_matrix) -> dict[str, float | int | None]:
    row_sum = np.asarray(graph.sum(axis=1)).reshape(-1).astype(np.float32, copy=False)
    return _summary_stats(row_sum)


def _graph_indegree_stats(graph: csr_matrix) -> dict[str, float | int | None]:
    indegree = np.asarray(graph.getnnz(axis=0)).reshape(-1).astype(np.float32, copy=False)
    return _summary_stats(indegree)


def _make_empty_candidate_row() -> dict[str, np.ndarray]:
    return {
        "cols": np.zeros((0,), dtype=np.int64),
        "dist": np.zeros((0,), dtype=np.float32),
        "dist_norm": np.zeros((0,), dtype=np.float32),
    }


def _build_spatial_candidate_data(
    coord: np.ndarray,
    sample_ids: np.ndarray,
    spatial_adj: csr_matrix,
    *,
    base_spatial_graph_k: int,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    n_obs = int(coord.shape[0])
    if spatial_adj.shape != (n_obs, n_obs):
        raise ValueError(
            "[arm_rep]: spatial_adj shape must match coordinates when building two-hop spatial candidates."
        )
    if int(sample_ids.shape[0]) != n_obs:
        raise ValueError(
            "[arm_rep]: sample_ids length must match coordinates when building two-hop spatial candidates."
        )

    coord_array = np.asarray(coord, dtype=np.float32)
    adj = spatial_adj.tocsr(copy=False)
    edge_row, edge_col = _edge_arrays(adj)
    if edge_row.size > 0 and np.any(sample_ids[edge_row] != sample_ids[edge_col]):
        raise RuntimeError("[arm_rep]: spatial_adj must be sample-safe before building two-hop spatial candidates.")

    candidate_data = [_make_empty_candidate_row() for _ in range(n_obs)]
    candidate_counts = np.zeros((n_obs,), dtype=np.int64)
    for global_row in range(n_obs):
        first_start, first_end = adj.indptr[global_row], adj.indptr[global_row + 1]
        first_hop = adj.indices[first_start:first_end].astype(np.int64, copy=False)
        if first_hop.size == 0:
            continue

        two_hop_parts = [first_hop]
        for neighbor in first_hop.tolist():
            second_start, second_end = adj.indptr[neighbor], adj.indptr[neighbor + 1]
            two_hop_parts.append(adj.indices[second_start:second_end].astype(np.int64, copy=False))

        # np.unique both deduplicates the N1/N2 union and gives a deterministic global-index order.
        candidates = np.unique(np.concatenate(two_hop_parts))
        candidates = candidates[candidates != int(global_row)]
        if candidates.size == 0:
            continue

        local_dist = np.linalg.norm(coord_array[candidates] - coord_array[global_row], axis=1).astype(
            np.float32,
            copy=False,
        )
        denom = float(np.max(local_dist)) if local_dist.size > 0 else 0.0
        if denom <= 1e-8:
            local_dist_norm = np.zeros_like(local_dist, dtype=np.float32)
        else:
            local_dist_norm = (local_dist / denom).astype(np.float32, copy=False)

        candidate_data[global_row] = {
            "cols": candidates.astype(np.int64, copy=False),
            "dist": local_dist,
            "dist_norm": local_dist_norm,
        }
        candidate_counts[global_row] = int(candidates.size)

    per_sample: list[dict[str, Any]] = []
    for sid in pd.Index(sample_ids).unique().tolist():
        idx = np.where(sample_ids == sid)[0]
        sample_counts = candidate_counts[idx]
        sample_count_stats = _summary_stats(sample_counts.astype(np.float32, copy=False))
        per_sample.append(
            {
                "sample_id": str(sid),
                "n_spots": int(idx.size),
                "n_candidate_edges": int(np.sum(sample_counts)),
                **{f"candidate_count_{key}": value for key, value in sample_count_stats.items()},
            }
        )

    total_edges = int(np.sum(candidate_counts))
    candidate_count_stats = _summary_stats(candidate_counts.astype(np.float32, copy=False))
    return candidate_data, {
        "candidate_source": "two_hop_spatial_knn_graph",
        "candidate_hops": 2,
        "base_spatial_graph_k": int(base_spatial_graph_k),
        "n_candidate_edges": total_edges,
        "nnz_candidate": total_edges,
        "candidate_count_stats": candidate_count_stats,
        "overall": {
            "n_spots": n_obs,
            "n_candidate_edges": total_edges,
            "candidate_count_stats": candidate_count_stats,
        },
        "per_sample": _records_to_columnar(per_sample),
    }


def _candidate_data_to_arrays(
    candidate_data: list[dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    dist: list[np.ndarray] = []
    dist_norm: list[np.ndarray] = []
    for row_idx, item in enumerate(candidate_data):
        if int(item["cols"].shape[0]) <= 0:
            continue
        rows.append(np.full(item["cols"].shape[0], int(row_idx), dtype=np.int64))
        cols.append(item["cols"].astype(np.int64, copy=False))
        dist.append(item["dist"].astype(np.float32, copy=False))
        dist_norm.append(item["dist_norm"].astype(np.float32, copy=False))
    if len(rows) == 0:
        empty_i64 = np.zeros((0,), dtype=np.int64)
        empty_f32 = np.zeros((0,), dtype=np.float32)
        return empty_i64, empty_i64.copy(), empty_f32, empty_f32.copy()
    return (
        np.concatenate(rows),
        np.concatenate(cols),
        np.concatenate(dist),
        np.concatenate(dist_norm),
    )


def _safe_softmax_np(scores: np.ndarray) -> np.ndarray:
    x = np.asarray(scores, dtype=np.float32).reshape(-1)
    if x.size == 0:
        return x
    x = x - float(np.max(x))
    w = np.exp(x).astype(np.float32, copy=False)
    denom = float(np.sum(w))
    if denom <= 0:
        return np.full_like(w, 1.0 / float(w.shape[0]), dtype=np.float32)
    return (w / denom).astype(np.float32, copy=False)


def _numeric_mean_std_median(values: np.ndarray) -> dict[str, float | int | None]:
    arr = np.asarray(values, dtype=np.float32).reshape(-1)
    if arr.size == 0:
        return {
            "n": 0,
            "mean": None,
            "std": None,
            "median": None,
        }
    return {
        "n": int(arr.size),
        "mean": float(np.mean(arr)),
        "std": float(np.std(arr)),
        "median": float(np.median(arr)),
    }


def _empty_edge_score_component_summary() -> dict[str, Any]:
    return {
        "available": False,
        "component_summary_basis": None,
        "latent_sim": _numeric_mean_std_median(np.zeros((0,), dtype=np.float32)),
        "expr_sim": _numeric_mean_std_median(np.zeros((0,), dtype=np.float32)),
        "dist_norm": _numeric_mean_std_median(np.zeros((0,), dtype=np.float32)),
        "spatial_proximity": _numeric_mean_std_median(np.zeros((0,), dtype=np.float32)),
        "affinity": _numeric_mean_std_median(np.zeros((0,), dtype=np.float32)),
    }


def _edge_score_component_summary(
    component_values: dict[str, list[np.ndarray]],
    *,
    basis: str,
) -> dict[str, Any]:
    if len(component_values.get("affinity", [])) == 0:
        return _empty_edge_score_component_summary()
    summary: dict[str, Any] = {
        "available": True,
        "component_summary_basis": basis,
    }
    for key in (
        "latent_sim",
        "expr_sim",
        "dist_norm",
        "spatial_proximity",
        "affinity",
    ):
        values = component_values.get(key, [])
        arr = np.concatenate(values).astype(np.float32, copy=False) if len(values) > 0 else np.zeros((0,), dtype=np.float32)
        summary[key] = _numeric_mean_std_median(arr)
    return summary


def _normalize_arm_history_rows(history_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized_rows: list[dict[str, Any]] = []
    for idx, row in enumerate(history_rows, start=1):
        normalized_row = {
            "epoch": int(row.get("epoch", idx)),
        }
        for field in ARM_HISTORY_FIELDS[1:]:
            value = row.get(field, 0.0)
            normalized_row[field] = 0.0 if value is None else float(value)
        normalized_rows.append(normalized_row)
    return normalized_rows








if TORCH_AVAILABLE:

    class ARMGraphSAGELayer(nn.Module):
        def __init__(self, in_dim: int, out_dim: int, dropout: float):
            super().__init__()
            self.lin_self = nn.Linear(in_dim, out_dim)
            self.lin_neigh = nn.Linear(in_dim, out_dim)
            self.norm = nn.LayerNorm(out_dim)
            self.dropout = float(dropout)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> torch.Tensor:
            neigh = torch.sparse.mm(adj_norm, x)
            out = self.lin_self(x) + self.lin_neigh(neigh)
            out = F.gelu(out)
            out = self.norm(out)
            if self.dropout > 0:
                out = F.dropout(out, p=self.dropout, training=self.training)
            return out


    class DepthAttentionResidual(nn.Module):
        def __init__(self, state_dims: tuple[int, ...], latent_dim: int, temperature: float):
            super().__init__()
            self.temperature = float(temperature)
            self.projectors = nn.ModuleList([nn.Linear(int(dim), int(latent_dim)) for dim in state_dims])
            self.scorers = nn.ModuleList([nn.Linear(int(latent_dim), 1) for _ in state_dims])

        def forward(self, states: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
            projected = [proj(h) for proj, h in zip(self.projectors, states)]
            scores = torch.cat([score(h) for score, h in zip(self.scorers, projected)], dim=1)
            attn = F.softmax(scores / max(self.temperature, 1e-6), dim=1)
            stacked = torch.stack(projected, dim=1)
            z_depth = torch.sum(attn.unsqueeze(-1) * stacked, dim=1)
            return z_depth, attn


    class ViewAttentionFusion(nn.Module):
        def __init__(
            self,
            input_dim: int,
            latent_dim: int,
            structural_corr_scale: float = 0.30,
            structural_gate_floor: float = 0.05,
        ):
            super().__init__()
            self.fusion_mode = ARM_FUSION_MODE
            self.structural_corr_scale = _validate_structural_corr_scale(structural_corr_scale)
            self.structural_gate_floor = _validate_structural_gate_floor(structural_gate_floor)
            self.eps = 1e-8
            self.expr_proj = nn.Linear(int(input_dim), int(latent_dim))
            self.spatial_proj = nn.Linear(int(latent_dim), int(latent_dim))
            self.residual_norm = nn.LayerNorm(int(latent_dim))
            self.depth_corr_proj = nn.Linear(int(latent_dim), int(latent_dim))
            self.spatial_corr_proj = nn.Linear(int(latent_dim), int(latent_dim))
            self.residual_corr_proj = nn.Linear(int(latent_dim), int(latent_dim))
            self.gate_mlp = nn.Sequential(
                nn.Linear(int(latent_dim) * 4, int(latent_dim)),
                nn.GELU(),
                nn.Linear(int(latent_dim), 3),
            )

        def forward(self, x: torch.Tensor, z_depth: torch.Tensor, adj_norm: torch.Tensor) -> dict[str, Any]:
            z_expr = self.expr_proj(x)
            z_spatial = self.spatial_proj(torch.sparse.mm(adj_norm, z_depth))
            z_residual = self.residual_norm(z_depth - z_spatial)
            views = [z_expr, z_depth, z_spatial, z_residual]
            gate_logits = self.gate_mlp(torch.cat(views, dim=1))
            gate = torch.sigmoid(gate_logits)
            gate = self.structural_gate_floor + (1.0 - self.structural_gate_floor) * gate
            depth_corr = torch.tanh(self.depth_corr_proj(z_depth))
            spatial_corr = torch.tanh(self.spatial_corr_proj(z_spatial))
            residual_corr = torch.tanh(self.residual_corr_proj(z_residual))
            depth_component = self.structural_corr_scale * gate[:, 0:1] * depth_corr
            spatial_component = self.structural_corr_scale * gate[:, 1:2] * spatial_corr
            residual_component = self.structural_corr_scale * gate[:, 2:3] * residual_corr
            structural_corr = depth_component + spatial_component + residual_component
            z_main = z_expr + structural_corr

            contribution_norms = torch.stack(
                [
                    torch.norm(z_expr, p=2, dim=1),
                    torch.norm(depth_component, p=2, dim=1),
                    torch.norm(spatial_component, p=2, dim=1),
                    torch.norm(residual_component, p=2, dim=1),
                ],
                dim=1,
            )
            contribution_denom = torch.sum(contribution_norms, dim=1, keepdim=True)
            raw_contribution = contribution_norms / torch.clamp(contribution_denom, min=self.eps)
            attn = torch.where(
                contribution_denom > self.eps,
                raw_contribution,
                torch.full_like(raw_contribution, 1.0 / float(raw_contribution.shape[1])),
            )
            structural_corr_norm_ratio = torch.norm(structural_corr, p=2, dim=1) / (
                torch.norm(z_expr, p=2, dim=1) + self.eps
            )
            return {
                "z_main": z_main,
                "view_attn": attn,
                "z_expr": z_expr,
                "z_spatial": z_spatial,
                "z_residual": z_residual,
                "structural_gates": gate,
                "structural_corr_norm_ratio": structural_corr_norm_ratio,
                "fusion_mode": self.fusion_mode,
            }


    class ARMGraphSAGEAutoencoder(nn.Module):
        def __init__(
            self,
            input_dim: int,
            hidden_dims: tuple[int, ...],
            latent_dim: int,
            dropout: float,
            temperature: float,
            structural_corr_scale: float = 0.30,
            structural_gate_floor: float = 0.05,
        ):
            super().__init__()
            dims = [int(input_dim)] + [int(v) for v in hidden_dims] + [int(latent_dim)]
            self.encoder_layers = nn.ModuleList(
                [ARMGraphSAGELayer(dims[i], dims[i + 1], dropout=dropout) for i in range(len(dims) - 1)]
            )
            self.depth_attention = DepthAttentionResidual(tuple(dims), latent_dim=int(latent_dim), temperature=float(temperature))
            self.view_fusion = ViewAttentionFusion(
                input_dim=int(input_dim),
                latent_dim=int(latent_dim),
                structural_corr_scale=float(structural_corr_scale),
                structural_gate_floor=float(structural_gate_floor),
            )

            dec_dims = [int(latent_dim)] + list(reversed([int(v) for v in hidden_dims])) + [int(input_dim)]
            mlp: list[nn.Module] = []
            for i in range(len(dec_dims) - 1):
                mlp.append(nn.Linear(dec_dims[i], dec_dims[i + 1]))
                if i < len(dec_dims) - 2:
                    mlp.append(nn.GELU())
                    if dropout > 0:
                        mlp.append(nn.Dropout(p=float(dropout)))
            self.decoder = nn.Sequential(*mlp)

        def encode_states(self, x: torch.Tensor, adj_norm: torch.Tensor) -> list[torch.Tensor]:
            states = [x]
            h = x
            for layer in self.encoder_layers:
                h = layer(h, adj_norm)
                states.append(h)
            return states

        def decode(self, z: torch.Tensor) -> torch.Tensor:
            return self.decoder(z)

        def forward(self, x: torch.Tensor, adj_norm: torch.Tensor) -> dict[str, Any]:
            states = self.encode_states(x, adj_norm)
            z_depth, depth_attn = self.depth_attention(states)
            fused = self.view_fusion(x, z_depth, adj_norm)
            x_recon = self.decode(fused["z_main"])
            return {
                "z": fused["z_main"],
                "x_recon": x_recon,
                "depth_attn": depth_attn,
                "view_attn": fused["view_attn"],
                "z_depth": z_depth,
                "z_expr": fused["z_expr"],
                "z_spatial": fused["z_spatial"],
                "z_residual": fused["z_residual"],
                "structural_gates": fused["structural_gates"],
                "structural_corr_norm_ratio": fused["structural_corr_norm_ratio"],
                "fusion_mode": fused["fusion_mode"],
            }

def _continuous_microdomain_affinity(
    z_main: np.ndarray,
    row_idx: np.ndarray,
    col_idx: np.ndarray,
    expr_sim: np.ndarray,
    dist_norm: np.ndarray,
) -> dict[str, np.ndarray]:
    rows = np.asarray(row_idx, dtype=np.int64).reshape(-1)
    cols = np.asarray(col_idx, dtype=np.int64).reshape(-1)
    n_edges = int(rows.size)
    empty = np.zeros((0,), dtype=np.float32)
    if n_edges == 0:
        return {
            "latent_sim": empty,
            "expr_sim": empty,
            "dist_norm": empty,
            "spatial_proximity": empty,
            "affinity": empty,
        }
    z = np.nan_to_num(np.asarray(z_main, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    latent_cos = _cosine_similarity_rows(z[rows], z[cols])
    latent_sim = np.clip(np.nan_to_num((latent_cos + 1.0) * 0.5), 0.0, 1.0).astype(np.float32, copy=False)
    expr_sim = np.clip(np.nan_to_num(np.asarray(expr_sim, dtype=np.float32).reshape(-1)), 0.0, 1.0)
    dist_norm = np.clip(np.nan_to_num(np.asarray(dist_norm, dtype=np.float32).reshape(-1)), 0.0, 1.0)
    spatial_proximity = np.clip(1.0 - dist_norm, 0.0, 1.0).astype(np.float32, copy=False)
    affinity = np.clip(
        np.nan_to_num((latent_sim + expr_sim + spatial_proximity) / 3.0),
        0.0,
        1.0,
    ).astype(np.float32, copy=False)
    return {
        "latent_sim": latent_sim,
        "expr_sim": expr_sim.astype(np.float32, copy=False),
        "dist_norm": dist_norm.astype(np.float32, copy=False),
        "spatial_proximity": spatial_proximity,
        "affinity": affinity,
    }


def _attention_entropy_floor(attn: torch.Tensor, floor_ratio: float) -> torch.Tensor:
    if attn.numel() == 0:
        return attn.new_tensor(0.0)
    n_choices = int(attn.shape[1]) if attn.ndim >= 2 else 0
    if n_choices <= 1:
        return attn.new_tensor(0.0)
    max_entropy = math.log(n_choices)
    if max_entropy <= 0:
        return attn.new_tensor(0.0)
    entropy = -torch.sum(attn * torch.log(torch.clamp(attn, min=1e-8)), dim=1)
    floor = float(floor_ratio) * max_entropy
    return torch.mean(F.relu(attn.new_tensor(floor) - entropy) / max_entropy)


def _build_attention_summary(weights: np.ndarray, names: list[str], *, axis_label: str) -> dict[str, Any]:
    weight_np = np.asarray(weights, dtype=np.float32)
    if weight_np.ndim != 2 or weight_np.shape[1] != len(names):
        return {
            "available": False,
            axis_label: names,
            "mean": [],
            "std": [],
        }
    return {
        "available": True,
        axis_label: names,
        "mean": np.mean(weight_np, axis=0).astype(np.float32, copy=False),
        "std": np.std(weight_np, axis=0).astype(np.float32, copy=False),
    }


def _view_contribution_semantics() -> str:
    return "anchored_correction_contribution_ratio"


def _build_structural_correction_summary(
    *,
    structural_gates: np.ndarray | None,
    structural_corr_norm_ratio: np.ndarray | None,
    corr_scale: float,
    gate_floor: float,
) -> dict[str, Any]:
    summary: dict[str, Any] = {
        "available": False,
        "corr_scale": float(corr_scale),
        "gate_floor": float(gate_floor),
        "gate_names": list(ARM_STRUCTURAL_GATE_NAMES),
        "gate_mean": [],
        "gate_std": [],
        "structural_corr_norm_ratio_mean": None,
        "structural_corr_norm_ratio_median": None,
        "structural_corr_norm_ratio_std": None,
        "view_contribution_semantics": _view_contribution_semantics(),
    }
    if structural_gates is None or structural_corr_norm_ratio is None:
        summary["reason"] = "structural correction diagnostics unavailable"
        return summary

    gates_np = np.asarray(structural_gates, dtype=np.float32)
    ratio_np = np.asarray(structural_corr_norm_ratio, dtype=np.float32).reshape(-1)
    if gates_np.ndim != 2 or gates_np.shape[1] != len(ARM_STRUCTURAL_GATE_NAMES):
        summary["reason"] = "structural gate array has unexpected shape"
        return summary
    if ratio_np.shape[0] != gates_np.shape[0]:
        summary["reason"] = "structural correction ratio length does not match gates"
        return summary
    if gates_np.shape[0] == 0:
        summary["reason"] = "no spots available for structural correction diagnostics"
        return summary

    summary.update(
        {
            "available": True,
            "gate_mean": [float(v) for v in np.mean(gates_np, axis=0).tolist()],
            "gate_std": [float(v) for v in np.std(gates_np, axis=0).tolist()],
            "structural_corr_norm_ratio_mean": float(np.mean(ratio_np)),
            "structural_corr_norm_ratio_median": float(np.median(ratio_np)),
            "structural_corr_norm_ratio_std": float(np.std(ratio_np)),
        }
    )
    return summary


def _refine_arm_microdomain_outputs(
    z_main: np.ndarray,
    x_edge_expr: np.ndarray,
    sample_ids: np.ndarray,
    candidate_data: list[dict[str, np.ndarray]],
    *,
    refine_topk: int,
    score_chunk_size: int = 200000,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    csr_matrix,
    dict[str, Any],
    np.ndarray,
]:
    n_obs = int(z_main.shape[0])
    d = int(max(1, refine_topk))
    idx_out = np.repeat(np.arange(n_obs, dtype=np.int64)[:, None], d, axis=1)
    w_out = np.zeros((n_obs, d), dtype=np.float32)
    confidence = np.zeros(n_obs, dtype=np.float32)
    boundary = np.zeros(n_obs, dtype=np.float32)
    spatial_radius = np.zeros(n_obs, dtype=np.float32)
    effective_k = np.full(n_obs, d, dtype=np.int32)
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    active_edge_expr_cos: list[np.ndarray] = []
    n_edges_selected_after_topk = 0
    selected_components: dict[str, list[np.ndarray]] = {
        "latent_sim": [],
        "expr_sim": [],
        "dist_norm": [],
        "spatial_proximity": [],
        "affinity": [],
    }
    per_sample_tracker: dict[str, dict[str, list[float] | int]] = {
        str(sid): {
            "n_spots": int(np.sum(sample_ids == sid)),
            "confidence": [],
            "radius": [],
            "boundary": [],
        }
        for sid in pd.Index(sample_ids).unique().tolist()
    }

    try:
        chunk_size_raw = int(score_chunk_size)
    except (TypeError, ValueError):
        chunk_size_raw = 200000
    chunk_size = int(chunk_size_raw if chunk_size_raw > 0 else 200000)

    candidate_row_np, candidate_col_np, candidate_dist_np, candidate_dist_norm_np = _candidate_data_to_arrays(candidate_data)
    n_edges = int(candidate_row_np.shape[0])
    counts = np.bincount(candidate_row_np, minlength=n_obs).astype(np.int64, copy=False) if n_edges > 0 else np.zeros(n_obs, dtype=np.int64)
    row_ptr = np.concatenate(
        [
            np.zeros((1,), dtype=np.int64),
            np.cumsum(counts, dtype=np.int64),
        ]
    )

    if n_edges > 0:
        expr_cos_raw_all = _cosine_similarity_rows(
            np.asarray(x_edge_expr[candidate_row_np], dtype=np.float32),
            np.asarray(x_edge_expr[candidate_col_np], dtype=np.float32),
        ).astype(np.float32, copy=False)
        expr_sim01_all = np.clip((expr_cos_raw_all + 1.0) * 0.5, 0.0, 1.0).astype(np.float32, copy=False)
    else:
        expr_cos_raw_all = np.zeros((0,), dtype=np.float32)
        expr_sim01_all = np.zeros((0,), dtype=np.float32)

    latent_sim_all = np.zeros(n_edges, dtype=np.float32)
    expr_sim_all = np.zeros(n_edges, dtype=np.float32)
    dist_norm_all = np.zeros(n_edges, dtype=np.float32)
    spatial_proximity_all = np.zeros(n_edges, dtype=np.float32)
    affinity_all = np.zeros(n_edges, dtype=np.float32)
    for start in range(0, n_edges, chunk_size):
        end = int(min(start + chunk_size, n_edges))
        components = _continuous_microdomain_affinity(
            z_main,
            candidate_row_np[start:end],
            candidate_col_np[start:end],
            expr_sim01_all[start:end],
            candidate_dist_norm_np[start:end],
        )
        latent_sim_all[start:end] = components["latent_sim"]
        expr_sim_all[start:end] = components["expr_sim"]
        dist_norm_all[start:end] = components["dist_norm"]
        spatial_proximity_all[start:end] = components["spatial_proximity"]
        affinity_all[start:end] = components["affinity"]

    for global_row in range(n_obs):
        start = int(row_ptr[global_row])
        end = int(row_ptr[global_row + 1])
        if end <= start:
            effective_k[global_row] = 1
            idx_out[global_row, 0] = int(global_row)
            w_out[global_row, 0] = 1.0
            continue

        candidate_global = candidate_col_np[start:end]
        row_affinity = affinity_all[start:end]
        expr_cos = expr_cos_raw_all[start:end]
        dist = candidate_dist_np[start:end]
        candidate_count = int(candidate_global.shape[0])
        select_k_i = int(min(d, candidate_count))
        effective_k[global_row] = int(select_k_i)

        order = np.argsort(-row_affinity, kind="mergesort")[:select_k_i]
        selected_edge_indices = (start + order).astype(np.int64, copy=False)
        sel_neighbors = candidate_global[order].astype(np.int64, copy=False)
        selected_affinity = row_affinity[order].astype(np.float32, copy=False)
        sel_expr_cos = expr_cos[order].astype(np.float32, copy=False)
        sel_dist = dist[order].astype(np.float32, copy=False)

        weight = _safe_softmax_np(selected_affinity)
        k_eff = int(sel_neighbors.shape[0])
        n_edges_selected_after_topk += int(k_eff)
        idx_out[global_row, :k_eff] = sel_neighbors
        w_out[global_row, :k_eff] = weight
        affinity_margin = (selected_affinity - np.float32(0.5)).astype(np.float32, copy=False)
        conf_value = float(np.sum(weight * (1.0 / (1.0 + np.exp(-affinity_margin)))))
        confidence[global_row] = conf_value
        entropy = float(-np.sum(weight * np.log(np.clip(weight, 1e-8, None))) / np.log(max(k_eff, 2)))
        boundary[global_row] = float(np.clip(0.5 * entropy + 0.5 * (1.0 - conf_value), 0.0, 1.0))
        radius_value = float(np.sum(weight * sel_dist))
        spatial_radius[global_row] = radius_value
        rows.append(np.full(k_eff, int(global_row), dtype=np.int64))
        cols.append(sel_neighbors)
        vals.append(weight.astype(np.float32, copy=False))
        active_edge_expr_cos.append(sel_expr_cos.astype(np.float32, copy=False))
        selected_components["latent_sim"].append(latent_sim_all[selected_edge_indices].astype(np.float32, copy=False))
        selected_components["expr_sim"].append(expr_sim_all[selected_edge_indices].astype(np.float32, copy=False))
        selected_components["dist_norm"].append(dist_norm_all[selected_edge_indices].astype(np.float32, copy=False))
        selected_components["spatial_proximity"].append(
            spatial_proximity_all[selected_edge_indices].astype(np.float32, copy=False)
        )
        selected_components["affinity"].append(affinity_all[selected_edge_indices].astype(np.float32, copy=False))

        sid = str(sample_ids[global_row])
        tracker = per_sample_tracker[sid]
        tracker["confidence"].append(conf_value)
        tracker["radius"].append(radius_value)
        tracker["boundary"].append(float(boundary[global_row]))

    if len(rows) == 0:
        graph = csr_matrix((n_obs, n_obs), dtype=np.float32)
        edge_expr_cos_summary = _summary_stats(np.zeros((0,), dtype=np.float32))
    else:
        graph = csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
            shape=(n_obs, n_obs),
            dtype=np.float32,
        )
        graph.eliminate_zeros()
        edge_expr_cos_summary = _summary_stats(np.concatenate(active_edge_expr_cos))
    final_indegree = np.asarray(graph.getnnz(axis=0)).reshape(-1).astype(np.float32, copy=False)

    per_sample_rows = []
    for sid in pd.Index(sample_ids).unique().tolist():
        sample_mask = sample_ids == sid
        tracker = per_sample_tracker[str(sid)]
        conf_list = np.asarray(tracker["confidence"], dtype=np.float32)
        radius_list = np.asarray(tracker["radius"], dtype=np.float32)
        boundary_list = np.asarray(tracker["boundary"], dtype=np.float32)
        per_sample_rows.append(
            {
                "sample_id": str(sid),
                "n_spots": int(tracker["n_spots"]),
                "mean_confidence": float(np.mean(conf_list)) if conf_list.size > 0 else None,
                "mean_radius": float(np.mean(radius_list)) if radius_list.size > 0 else None,
                "mean_boundary_uncertainty": float(np.mean(boundary_list)) if boundary_list.size > 0 else None,
                "mean_effective_k": float(np.mean(effective_k[sample_mask])) if np.any(sample_mask) else None,
            }
        )

    graph_summary = {
        "nnz": int(graph.nnz),
        "microdomain_graph_method": ARM_MICRODOMAIN_GRAPH_METHOD,
        "affinity_method": AFFINITY_METHOD,
        "neighbor_selection_method": NEIGHBOR_SELECTION_METHOD,
        "neighbor_count": int(d),
        "refine_score_chunk_size": int(chunk_size),
        "n_candidate_edges_scored": int(n_edges),
        "n_edges_selected_after_topk": int(n_edges_selected_after_topk),
        "posthoc_indegree_max": int(np.max(final_indegree)) if final_indegree.size > 0 else 0,
        "posthoc_indegree_std": float(np.std(final_indegree)) if final_indegree.size > 0 else 0.0,
        "row_sum_stats": _row_sum_stats(graph),
        "indegree_stats": _graph_indegree_stats(graph),
        "cross_sample": count_cross_sample_edges(graph, sample_ids),
        "edge_expression_cosine": edge_expr_cos_summary,
        "edge_score_component_summary": _edge_score_component_summary(
            selected_components,
            basis="selected_arm_refined_edges",
        ),
        "spatial_radius_stats": _summary_stats(spatial_radius),
        "affinity_formula": AFFINITY_FORMULA,
        "weight_formula": WEIGHT_FORMULA,
        "confidence_formula": CONFIDENCE_FORMULA,
        "sample_processing_scope": "per_sample",
        "per_sample": _records_to_columnar(per_sample_rows),
    }
    return (
        idx_out,
        w_out,
        confidence,
        boundary,
        spatial_radius,
        graph,
        graph_summary,
        effective_k,
    )


def run_arm_rep(
    adata: AnnData,
    cfg,
    *,
    device_override: str | None = None,
    training_scope: str = "single",
) -> AnnData:
    _set_seed(int(cfg.seed))
    if not TORCH_AVAILABLE or torch is None or nn is None or F is None:
        raise RuntimeError(
            "[arm_rep]: formal ARM-REP requires PyTorch; install/import torch to run the formal GraphSAGE model."
        )
    device = _resolve_device(device_override if device_override is not None else getattr(cfg, "rep_gpu", None))
    fusion_mode = ARM_FUSION_MODE
    structural_corr_scale = _validate_structural_corr_scale(getattr(cfg, "arm_structural_corr_scale", 0.30))
    structural_gate_floor = _validate_structural_gate_floor(getattr(cfg, "arm_structural_gate_floor", 0.05))
    training_scope = str(training_scope).strip().lower()
    if training_scope not in {"single", "multi"}:
        raise ValueError("[arm_rep]: training_scope must be either 'single' or 'multi'.")
    logger.info(f"[arm_rep]: active_rep='arm_rep', device={device}.")
    if getattr(device, "type", str(device)) == "cpu":
        logger.warning("[arm_rep]: formal PyTorch GraphSAGE is running on CPU.")

    graph_sample_col = str(cfg.rep_graph_sample_id_col)
    if graph_sample_col not in adata.obs.columns:
        raise KeyError(f"[arm_rep]: required obs['{graph_sample_col}'] is missing.")
    harmony_batch_col = str(
        getattr(cfg, "rep_harmony_batch_col", None) or graph_sample_col
    ).strip()
    if training_scope == "multi" and harmony_batch_col not in adata.obs.columns:
        raise KeyError(f"[arm_rep]: required Harmony batch obs['{harmony_batch_col}'] is missing.")
    coord_key = str(cfg.rep_coord_key)
    if coord_key not in adata.obsm:
        raise KeyError(f"[arm_rep]: required obsm['{coord_key}'] is missing.")

    graph_sample_ids_full = _normalize_sample_ids(
        adata.obs[graph_sample_col], adata.n_obs
    )
    harmony_batch_ids_full = (
        _normalize_sample_ids(adata.obs[harmony_batch_col], adata.n_obs)
        if training_scope == "multi"
        else graph_sample_ids_full
    )
    unique_graph_sample_ids = pd.Index(graph_sample_ids_full).unique().tolist()
    unique_harmony_batch_ids = pd.Index(harmony_batch_ids_full).unique().tolist()
    if training_scope == "single" and len(unique_graph_sample_ids) != 1:
        raise ValueError(
            "[arm_rep]: run_arm_rep requires exactly one unique sample_id; "
            f"obs['{graph_sample_col}'] contains {len(unique_graph_sample_ids)} samples. "
            "Call it through the formal sample-wise arm_rep -> single_map wrapper."
        )
    if training_scope == "multi":
        if len(unique_graph_sample_ids) < 2:
            raise ValueError(
                "[arm_rep]: multi-sample training requires at least two unique sample IDs; "
                f"obs['{graph_sample_col}'] contains {len(unique_graph_sample_ids)} sample(s)."
            )
    metadata_training_scope = "per_sample" if training_scope == "single" else "multi_sample_shared"
    metadata_preprocessing_scope = "per_sample" if training_scope == "single" else "multi_sample_shared"
    metadata_spatial_graph_scope = "per_sample"
    metadata_microdomain_scope = "per_sample"
    metadata_scope_fields = {
        "sample_processing_scope": metadata_preprocessing_scope,
        "preprocessing_scope": metadata_preprocessing_scope,
        "training_scope": metadata_training_scope,
        "spatial_graph_scope": metadata_spatial_graph_scope,
        "microdomain_scope": metadata_microdomain_scope,
    }
    shared_parameters_across_samples = training_scope == "multi"
    n_input_samples = int(len(unique_graph_sample_ids))
    n_harmony_batches = int(len(unique_harmony_batch_ids))
    integration_metadata = (
        {}
        if training_scope == "single"
        else {
            "integration_method": "Harmony",
            "integration_space": "shared_PCA",
            "rep_input_basis": "Harmony_corrected_shared_PCA",
            "integration_scope": (
                "cross_donor"
                if harmony_batch_col == "donor_id" and harmony_batch_col != graph_sample_col
                else "multi_sample_shared"
            ),
            "graph_sample_id_col": graph_sample_col,
            "harmony_batch_col": harmony_batch_col,
            "n_harmony_batches": n_harmony_batches,
        }
    )
    coord_full = np.asarray(adata.obsm[coord_key], dtype=np.float32)
    if coord_full.shape[0] != adata.n_obs:
        raise ValueError("[arm_rep]: spatial coordinate rows do not match n_obs.")

    x_count_source = adata.layers[cfg.rep_count_layer] if cfg.rep_count_layer is not None else adata.X
    x_count_source = _csr_float32(x_count_source) if issparse(x_count_source) else np.asarray(x_count_source, dtype=np.float32)

    requested_pca_dim = int(cfg.rep_pca_dim)
    requested_latent_dim = int(cfg.rep_dim_latent)
    effective_pca_dim = requested_pca_dim
    effective_latent_dim = requested_latent_dim

    spot_mask, gene_mask, filter_summary = _sample_safe_filter_masks(
        x_count_source,
        graph_sample_ids_full,
        min_spots_per_gene=int(cfg.rep_min_spots_per_gene),
        min_genes_per_spot=int(cfg.rep_min_genes_per_spot),
        filter_spots=bool(cfg.rep_filter_spots),
        filter_genes=bool(cfg.rep_filter_genes),
    )
    if int(np.sum(spot_mask)) <= 0 or int(np.sum(gene_mask)) <= 0:
        raise RuntimeError("[arm_rep]: sample-safe filtering removed all spots or genes.")

    adata_rep = adata[spot_mask, gene_mask].copy()
    graph_sample_ids = _normalize_sample_ids(
        adata_rep.obs[graph_sample_col], adata_rep.n_obs
    )
    harmony_batch_ids = (
        _normalize_sample_ids(
            adata_rep.obs[harmony_batch_col], adata_rep.n_obs
        )
        if training_scope == "multi"
        else graph_sample_ids
    )
    coord = np.asarray(adata_rep.obsm[coord_key], dtype=np.float32)
    x_count = x_count_source[spot_mask][:, gene_mask]

    gene_select_mask, hvg_summary = _preselect_genes_sample_safe(
        x_count,
        graph_sample_ids,
        normalize_target_sum=float(cfg.rep_normalize_target_sum),
        use_hvg=bool(cfg.rep_use_hvg),
        n_top_genes=int(cfg.rep_n_top_genes),
        hvg_flavor=str(cfg.rep_hvg_flavor),
    )
    adata_rep = adata_rep[:, gene_select_mask].copy()
    x_count = x_count[:, gene_select_mask]

    if training_scope == "multi" and set(
        pd.Index(graph_sample_ids).unique().tolist()
    ) != set(unique_graph_sample_ids):
        raise RuntimeError(
            "[arm_rep]: sample-safe filtering removed all retained spots from at least one input sample."
        )

    if training_scope == "single":
        x_log, x_pca, library_size, projections, pca_summary = _samplewise_log_scale_pca(
            x_count,
            graph_sample_ids,
            normalize_target_sum=float(cfg.rep_normalize_target_sum),
            pca_dim=effective_pca_dim,
            scale_max_value=float(cfg.rep_scale_max_value),
            seed=int(cfg.seed),
        )
    else:
        x_log, x_pca, library_size, projections, pca_summary = _multisample_log_scale_pca_harmony(
            x_count,
            graph_sample_ids,
            harmony_batch_ids=harmony_batch_ids,
            normalize_target_sum=float(cfg.rep_normalize_target_sum),
            pca_dim=effective_pca_dim,
            scale_max_value=float(cfg.rep_scale_max_value),
            seed=int(cfg.seed),
        )
        pca_summary["integration_batch_key"] = harmony_batch_col
    del library_size
    adata_rep.X = x_log.astype(np.float32, copy=False)
    adata_rep.layers["counts"] = _csr_float32(x_count) if issparse(x_count) else np.asarray(x_count, dtype=np.float32)
    adata_rep.layers["log1p_norm"] = x_log.astype(np.float32, copy=False)
    adata_rep.obsm["rep_pca_input"] = x_pca.astype(np.float32, copy=False)

    spatial_adj, spatial_graph_summary = build_samplewise_spatial_graph(
        coord,
        graph_sample_ids,
        k=int(cfg.rep_graph_k),
        radius=cfg.rep_graph_radius,
        include_self=False,
        sym=True,
    )
    cross_stats = spatial_graph_summary["sample_safety"]
    if bool(cfg.rep_graph_strict_sample_safety) or training_scope == "multi":
        if int(cross_stats.get("cross_sample_edges", 0)) > 0:
            raise RuntimeError("[arm_rep]: spot_graph_spatial is not sample-safe.")
    cross_sample_graph_edges = int(cross_stats.get("cross_sample_edges", 0))

    adj_norm = _row_normalize_csr(spatial_adj)
    pos_row_np, pos_col_np = _edge_arrays(spatial_adj)
    local_weight_np = _sample_expression_edge_weight(x_pca, pos_row_np, pos_col_np)
    logger.info("[arm_rep]: negative sampling mode=exact")
    neg_row_np, neg_col_np = _sample_negative_edges(
        spatial_adj,
        graph_sample_ids,
        negatives_per_node=int(cfg.rep_negative_samples),
        seed=int(cfg.seed),
    )

    arm_refine_topk = int(getattr(cfg, "arm_refine_topk", 14))
    if arm_refine_topk < 1:
        raise ValueError("[arm_rep]: arm_refine_topk must be at least 1.")
    microdomain_graph_metadata = {
        "microdomain_graph_method": ARM_MICRODOMAIN_GRAPH_METHOD,
        "affinity_method": AFFINITY_METHOD,
        "neighbor_selection_method": NEIGHBOR_SELECTION_METHOD,
        "neighbor_count": int(arm_refine_topk),
        "affinity_formula": AFFINITY_FORMULA,
        "weight_formula": WEIGHT_FORMULA,
        "confidence_formula": CONFIDENCE_FORMULA,
        "sample_processing_scope": "per_sample",
    }
    candidate_data, candidate_summary = _build_spatial_candidate_data(
        coord,
        graph_sample_ids,
        spatial_adj,
        base_spatial_graph_k=int(spatial_graph_summary["k"]),
    )

    logger.info(
        "[arm_rep]: fusion -> "
        f"mode={fusion_mode}, corr_scale={structural_corr_scale:.3f}, gate_floor={structural_gate_floor:.3f}"
    )
    logger.info(
        "[arm_rep]: microdomain graph -> "
        f"method={ARM_MICRODOMAIN_GRAPH_METHOD}, topk={arm_refine_topk}, "
        "candidates=samplewise_two_hop, affinity=latent_expression_spatial_mean"
    )
    logger.info(
        "[arm_rep]: training objective -> "
        f"recon={float(cfg.rep_w_recon):.3f}, local={float(cfg.rep_w_local):.3f}, "
        f"adj={float(cfg.rep_w_spatial_adj):.3f}, anti={float(cfg.rep_w_anti):.3f}, "
        f"depth_entropy={float(cfg.arm_w_depth_entropy):.3f}, view_entropy={float(cfg.arm_w_view_entropy):.3f}, "
        f"entropy_floor_ratio={float(getattr(cfg, 'arm_attn_entropy_floor_ratio', 0.55)):.3f}"
    )
    logger.info(
        "[arm_rep]: sample-safe preprocessing -> "
        f"spots={adata.n_obs}->{adata_rep.n_obs}, genes={adata.n_vars}->{adata_rep.n_vars}, "
        f"pca_dim={x_pca.shape[1]}, graph_k={int(cfg.rep_graph_k)}, "
        "candidate_source=two_hop_spatial_knn_graph, candidate_hops=2, "
        f"refine_topk={arm_refine_topk}"
    )

    training_max_iter = int(cfg.rep_max_iter)
    training_patience = int(cfg.rep_patience)

    if TORCH_AVAILABLE:
        hidden_dims = tuple(int(v) for v in cfg.rep_hidden_dims)
        model = ARMGraphSAGEAutoencoder(
            input_dim=int(x_pca.shape[1]),
            hidden_dims=hidden_dims,
            latent_dim=int(effective_latent_dim),
            dropout=float(cfg.rep_dropout),
            temperature=float(cfg.arm_attn_temperature),
            structural_corr_scale=structural_corr_scale,
            structural_gate_floor=structural_gate_floor,
        ).to(device)
        x_pca_t = torch.tensor(x_pca, dtype=torch.float32, device=device)

        optimizer = torch.optim.AdamW(
            list(model.parameters()),
            lr=float(cfg.rep_lr),
            weight_decay=float(cfg.rep_weight_decay),
        )

        adj_t = _csr_to_torch_sparse(adj_norm, device=device)
        pos_row_t = torch.tensor(pos_row_np, dtype=torch.long, device=device)
        pos_col_t = torch.tensor(pos_col_np, dtype=torch.long, device=device)
        neg_row_t = torch.tensor(neg_row_np, dtype=torch.long, device=device)
        neg_col_t = torch.tensor(neg_col_np, dtype=torch.long, device=device)
        local_weight_t = torch.tensor(local_weight_np, dtype=torch.float32, device=device)

        best_metric = float("inf")
        best_epoch = -1
        wait = 0
        best_state = None
        history_rows: list[dict[str, Any]] = []
        iterator = range(1, int(cfg.rep_max_iter) + 1)
        progress_bar = None
        if tqdm is not None:
            progress_bar = tqdm(
                iterator,
                desc="[arm_rep] training",
                dynamic_ncols=True,
                leave=True,
            )
            iterator = progress_bar

        try:
            for epoch in iterator:
                model.train()
                optimizer.zero_grad(set_to_none=True)
                out = model(x_pca_t, adj_t)
                loss_recon = F.mse_loss(out["x_recon"], x_pca_t)
                loss_local = _local_consistency_loss(out["z"], pos_row_t, pos_col_t, local_weight_t)
                loss_adj = _graph_loss(
                    out["z"],
                    pos_row_t,
                    pos_col_t,
                    neg_row_t,
                    neg_col_t,
                    scale=float(cfg.rep_graph_logit_scale),
                )
                loss_anti = _variance_regularization(out["z"], target_std=float(cfg.rep_var_target))
                loss_depth_entropy = _attention_entropy_floor(
                    out["depth_attn"],
                    float(getattr(cfg, "arm_attn_entropy_floor_ratio", 0.55)),
                )
                loss_view_entropy = _attention_entropy_floor(
                    out["view_attn"],
                    float(getattr(cfg, "arm_attn_entropy_floor_ratio", 0.55)),
                )
                loss_total = (
                    float(cfg.rep_w_recon) * loss_recon
                    + float(cfg.rep_w_local) * loss_local
                    + float(cfg.rep_w_spatial_adj) * loss_adj
                    + float(cfg.rep_w_anti) * loss_anti
                    + float(cfg.arm_w_depth_entropy) * loss_depth_entropy
                    + float(cfg.arm_w_view_entropy) * loss_view_entropy
                )
                loss_total.backward()
                if float(cfg.rep_clip) > 0:
                    torch.nn.utils.clip_grad_norm_(
                        list(model.parameters()),
                        max_norm=float(cfg.rep_clip),
                    )
                optimizer.step()

                metric = float(loss_total.detach().cpu().item())
                history_rows.append(
                    {
                        "epoch": int(epoch),
                        "loss_total": metric,
                        "loss_recon": float(loss_recon.detach().cpu().item()),
                        "loss_local": float(loss_local.detach().cpu().item()),
                        "loss_adj": float(loss_adj.detach().cpu().item()),
                        "loss_anti": float(loss_anti.detach().cpu().item()),
                        "loss_depth_entropy": float(loss_depth_entropy.detach().cpu().item()),
                        "loss_view_entropy": float(loss_view_entropy.detach().cpu().item()),
                    }
                )
                if progress_bar is not None:
                    progress_bar.set_postfix(loss=f"{metric:.4f}")

                if metric + float(cfg.rep_es_min_delta) < best_metric:
                    best_metric = metric
                    best_epoch = int(epoch)
                    wait = 0
                    best_state = {
                        "model": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
                    }
                else:
                    wait += 1
                    if wait >= int(cfg.rep_patience):
                        logger.info(
                            "[arm_rep]: early stop -> "
                            f"epoch={epoch}, best_epoch={best_epoch}, best_metric={best_metric:.6f}"
                        )
                        break
        finally:
            if progress_bar is not None:
                progress_bar.close()

        if best_state is not None:
            model.load_state_dict(best_state["model"], strict=True)
        training_backend = "torch_arm_graphsage_autoencoder"

        model.eval()
        with torch.no_grad():
            encoded = model(x_pca_t, adj_t)
        z_main = encoded["z"].detach().cpu().numpy().astype(np.float32, copy=False)
        x_pca_recon = encoded["x_recon"].detach().cpu().numpy().astype(np.float32, copy=False)
        depth_attn_np = encoded["depth_attn"].detach().cpu().numpy().astype(np.float32, copy=False)
        view_attn_np = encoded["view_attn"].detach().cpu().numpy().astype(np.float32, copy=False)
        structural_gates_np = encoded["structural_gates"].detach().cpu().numpy().astype(np.float32, copy=False)
        structural_corr_norm_ratio_np = (
            encoded["structural_corr_norm_ratio"].detach().cpu().numpy().astype(np.float32, copy=False)
        )
        fusion_mode_runtime = ARM_FUSION_MODE
    history_rows = _normalize_arm_history_rows(history_rows)

    x_denoised = _materialize_denoised_expression(x_pca_recon, graph_sample_ids, projections)
    (
        micro_idx_spatial,
        micro_w_spatial,
        micro_conf_spatial,
        boundary_spatial,
        spatial_radius_spatial,
        micro_graph_spatial,
        arm_graph_summary,
        effective_k_spatial,
    ) = _refine_arm_microdomain_outputs(
        z_main,
        x_pca,
        graph_sample_ids,
        candidate_data,
        refine_topk=arm_refine_topk,
        score_chunk_size=getattr(cfg, "arm_refine_score_chunk_size", 200000),
    )
    if training_scope == "multi" and int(
        arm_graph_summary.get("cross_sample", {}).get("cross_sample_edges", 0)
    ) > 0:
        raise RuntimeError("[arm_rep]: refined microdomain graph is not sample-safe.")
    logger.info(
        "[arm_rep]: refined microdomain graph -> "
        f"selected_edges={int(arm_graph_summary['n_edges_selected_after_topk'])}, "
        f"target_k={arm_refine_topk}"
    )

    micro_idx = micro_idx_spatial
    micro_w = micro_w_spatial
    micro_conf = micro_conf_spatial
    boundary_uncertainty = boundary_spatial
    spatial_radius = spatial_radius_spatial
    micro_graph = micro_graph_spatial
    edge_score_component_summary = dict(arm_graph_summary.get("edge_score_component_summary", _empty_edge_score_component_summary()))
    if bool(edge_score_component_summary.get("available", False)):
        edge_score_component_summary["component_summary_basis"] = "selected_active_edges"

    layer_names = ["h0_input"] + [f"h{i}" for i in range(1, 1 + len(tuple(int(v) for v in cfg.rep_hidden_dims)) + 1)]
    view_names = ["expr", "depth", "spatial", "residual"]
    depth_attention_summary = _build_attention_summary(depth_attn_np, layer_names, axis_label="layers")
    view_attention_summary = _build_attention_summary(view_attn_np, view_names, axis_label="views")
    view_semantics = _view_contribution_semantics()
    view_attention_summary["semantics"] = view_semantics
    view_attention_summary["view_contribution_semantics"] = view_semantics
    structural_correction_summary = _build_structural_correction_summary(
        structural_gates=structural_gates_np,
        structural_corr_norm_ratio=structural_corr_norm_ratio_np,
        corr_scale=structural_corr_scale,
        gate_floor=structural_gate_floor,
    )
    adata_rep.obsm[str(cfg.rep_embed_key)] = z_main.astype(np.float32, copy=False)
    adata_rep.obsm[str(cfg.rep_microdomain_embed_key)] = z_main.astype(np.float32, copy=False)
    adata_rep.layers[str(cfg.rep_denoised_key)] = x_denoised.astype(np.float32, copy=False)
    adata_rep.obsp[str(cfg.rep_graph_spatial_key)] = spatial_adj.astype(np.float32)
    adata_rep.obsp[str(cfg.rep_microdomain_graph_key)] = micro_graph.astype(np.float32)
    adata_rep.obsp[str(cfg.rep_graph_key)] = micro_graph.astype(np.float32)
    adata_rep.obs[str(cfg.rep_confidence_key)] = micro_conf.astype(np.float32, copy=False)
    adata_rep.obs[str(cfg.rep_boundary_obs_key)] = boundary_uncertainty.astype(np.float32, copy=False)
    adata_rep.obs["rep_microdomain_radius"] = spatial_radius.astype(np.float32, copy=False)
    adata_rep.obs["rep_microdomain_effective_k"] = effective_k_spatial.astype(np.int32, copy=False)

    graph_usage_contract = (
        "spot_graph_map is the active sample-safe ARM microdomain aggregation graph. For each focal spot, "
        "candidates are defined by the union of its one-hop and two-hop neighborhoods in the sample-wise spatial "
        "graph. Candidate affinity is the equal-weight mean of latent similarity, expression similarity, and "
        "spatial proximity. Up to arm_refine_topk candidates with the highest affinity are retained and normalized "
        "row-wise by softmax."
    )
    st_rep = {
        "active_rep": "arm_rep",
        "execution_mode": training_scope,
        "sample_processing_scope": "per_sample",
        "training_scope": metadata_training_scope,
        "n_training_samples": n_input_samples,
        "n_input_samples": n_input_samples,
        "n_independent_training_runs": 1,
        "shared_parameters_across_samples": shared_parameters_across_samples,
        "cross_sample_parameter_sharing": shared_parameters_across_samples,
        "cross_sample_graph_edges": cross_sample_graph_edges,
        **integration_metadata,
        "fusion_mode": fusion_mode_runtime,
        "primary_latent_key": str(cfg.rep_embed_key),
        "primary_struct_latent_key": str(cfg.rep_embed_key),
        "primary_microdomain_latent_key": str(cfg.rep_microdomain_embed_key),
        "primary_expr_key": None,
        "primary_domain_key": None,
        "primary_denoised_key": str(cfg.rep_denoised_key),
        "primary_graph_key": str(cfg.rep_graph_key),
        "primary_graph_map_key": str(cfg.rep_graph_key),
        "primary_graph_refined_key": str(cfg.rep_graph_key),
        "primary_graph_spatial_key": str(cfg.rep_graph_spatial_key),
        "primary_microdomain_graph_key": str(cfg.rep_microdomain_graph_key),
        "graph_sample_id_col": graph_sample_col,
        "graph_usage_contract": graph_usage_contract,
        **microdomain_graph_metadata,
        "edge_semantics": "microdomain_weights[i, j] is how strongly focal spot i aggregates homogeneous spot j",
        "uses_survival": False,
        "uses_classification": False,
    }
    st_rep.update(metadata_scope_fields)

    microdomain_summary = {
        "active_rep": "arm_rep",
        "execution_mode": training_scope,
        "sample_processing_scope": "per_sample",
        "training_scope": metadata_training_scope,
        "n_training_samples": n_input_samples,
        "n_input_samples": n_input_samples,
        "n_independent_training_runs": 1,
        "shared_parameters_across_samples": shared_parameters_across_samples,
        "cross_sample_parameter_sharing": shared_parameters_across_samples,
        "cross_sample_graph_edges": cross_sample_graph_edges,
        **integration_metadata,
        "architecture": "attention_residual_microdomain_rep",
        "fusion_mode": fusion_mode_runtime,
        "sample_id_col": str(cfg.rep_graph_sample_id_col),
        "coord_key": str(cfg.rep_coord_key),
        "input_count_layer": None if cfg.rep_count_layer is None else str(cfg.rep_count_layer),
        "n_input_spots": int(adata.n_obs),
        "n_output_spots": int(adata_rep.n_obs),
        "n_output_genes": int(adata_rep.n_vars),
        "filter_summary": filter_summary,
        "hvg_summary": hvg_summary,
        "pca_summary": pca_summary,
        "candidate_summary": candidate_summary,
        **microdomain_graph_metadata,
        "training": {
            "training_scope": metadata_training_scope,
            "n_training_samples": n_input_samples,
            "n_input_samples": n_input_samples,
            "n_independent_training_runs": 1,
            "shared_parameters_across_samples": shared_parameters_across_samples,
            "cross_sample_graph_edges": cross_sample_graph_edges,
            **integration_metadata,
            "device": str(device),
            "backend": training_backend,
            "best_epoch": int(best_epoch),
            "best_metric": float(best_metric),
            "max_iter": int(training_max_iter),
            "patience": int(training_patience),
            "history": _records_to_columnar(history_rows),
            "loss_weight_recon": float(cfg.rep_w_recon),
            "loss_weight_local": float(cfg.rep_w_local),
            "loss_weight_spatial_adj": float(cfg.rep_w_spatial_adj),
            "loss_weight_anti": float(cfg.rep_w_anti),
            "loss_weight_depth_entropy": float(cfg.arm_w_depth_entropy),
            "loss_weight_view_entropy": float(cfg.arm_w_view_entropy),
            "attention_entropy_floor_ratio": float(getattr(cfg, "arm_attn_entropy_floor_ratio", 0.55)),
            "fusion_mode": fusion_mode_runtime,
            **microdomain_graph_metadata,
        },
        "loss_history": _records_to_columnar(history_rows),
        "graphs": {
            "spot_graph_spatial_key": str(cfg.rep_graph_spatial_key),
            "spot_graph_map_key": str(cfg.rep_graph_key),
            "rep_microdomain_graph_key": str(cfg.rep_microdomain_graph_key),
            "spot_graph_spatial_summary": spatial_graph_summary,
            "microdomain_graph_summary": arm_graph_summary,
            "rep_microdomain_graph_nnz": int(micro_graph.nnz),
        },
        "depth_attention_summary": depth_attention_summary,
        "view_attention_summary": view_attention_summary,
        "structural_correction_summary": structural_correction_summary,
        "edge_score_component_summary": edge_score_component_summary,
        "arm_params": {
            "arm_attn_temperature": float(cfg.arm_attn_temperature),
            "arm_attn_entropy_floor_ratio": float(getattr(cfg, "arm_attn_entropy_floor_ratio", 0.55)),
            "arm_fusion_mode": fusion_mode_runtime,
            "arm_structural_corr_scale": float(structural_corr_scale),
            "arm_structural_gate_floor": float(structural_gate_floor),
            "arm_refine_topk": int(arm_refine_topk),
            "arm_w_depth_entropy": float(cfg.arm_w_depth_entropy),
            "arm_w_view_entropy": float(cfg.arm_w_view_entropy),
            "arm_refine_score_chunk_size": int(getattr(cfg, "arm_refine_score_chunk_size", 200000)),
            **microdomain_graph_metadata,
        },
        "microdomain_indices": micro_idx.astype(np.int64, copy=False),
        "microdomain_weights": micro_w.astype(np.float32, copy=False),
        "microdomain_confidence_stats": _summary_stats(micro_conf),
        "microdomain_radius_stats": _summary_stats(spatial_radius),
        "boundary_uncertainty_stats": _summary_stats(boundary_uncertainty),
        "microdomain_per_sample": arm_graph_summary["per_sample"],
        "output_keys": {
            "rep_z_main": str(cfg.rep_embed_key),
            "rep_z_microdomain": str(cfg.rep_microdomain_embed_key),
            "spot_graph_spatial": str(cfg.rep_graph_spatial_key),
            "spot_graph_map": str(cfg.rep_graph_key),
            "rep_microdomain_graph": str(cfg.rep_microdomain_graph_key),
            "rep_microdomain_confidence": str(cfg.rep_confidence_key),
            "rep_boundary_uncertainty": str(cfg.rep_boundary_obs_key),
            "denoised_mean": str(cfg.rep_denoised_key),
        },
    }
    microdomain_summary.update(metadata_scope_fields)
    microdomain_summary["training"].update(metadata_scope_fields)
    adata_rep.uns["st_rep"] = st_rep
    adata_rep.uns["microdomain_rep"] = microdomain_summary

    final_history_row = history_rows[-1] if len(history_rows) > 0 else {}
    training_summary = {
        "active_rep": "arm_rep",
        "execution_mode": training_scope,
        "sample_processing_scope": "per_sample",
        "training_scope": metadata_training_scope,
        "n_training_samples": n_input_samples,
        "n_input_samples": n_input_samples,
        "n_independent_training_runs": 1,
        "shared_parameters_across_samples": shared_parameters_across_samples,
        "cross_sample_parameter_sharing": shared_parameters_across_samples,
        "cross_sample_graph_edges": cross_sample_graph_edges,
        **integration_metadata,
        "backend": training_backend,
        "fusion_mode": fusion_mode_runtime,
        "best_epoch": int(best_epoch),
        "best_metric": float(best_metric),
        "max_iter": int(training_max_iter),
        "patience": int(training_patience),
        "n_epochs_recorded": int(len(history_rows)),
        "final_loss_total": final_history_row.get("loss_total"),
        "min_loss_total": min((row["loss_total"] for row in history_rows), default=None),
        "final_loss_recon": final_history_row.get("loss_recon"),
        "final_loss_local": final_history_row.get("loss_local"),
        "final_loss_adj": final_history_row.get("loss_adj"),
        "final_loss_anti": final_history_row.get("loss_anti"),
        "final_loss_depth_entropy": final_history_row.get("loss_depth_entropy"),
        "final_loss_view_entropy": final_history_row.get("loss_view_entropy"),
        "attention_entropy_floor_ratio": float(getattr(cfg, "arm_attn_entropy_floor_ratio", 0.55)),
        **microdomain_graph_metadata,
        "arm_params": {
            "arm_attn_temperature": float(cfg.arm_attn_temperature),
            "arm_attn_entropy_floor_ratio": float(getattr(cfg, "arm_attn_entropy_floor_ratio", 0.55)),
            "arm_fusion_mode": fusion_mode_runtime,
            "arm_structural_corr_scale": float(structural_corr_scale),
            "arm_structural_gate_floor": float(structural_gate_floor),
            "arm_refine_topk": int(arm_refine_topk),
            "arm_w_depth_entropy": float(cfg.arm_w_depth_entropy),
            "arm_w_view_entropy": float(cfg.arm_w_view_entropy),
            "arm_refine_score_chunk_size": int(getattr(cfg, "arm_refine_score_chunk_size", 200000)),
            **microdomain_graph_metadata,
        },
        "depth_attention_summary": depth_attention_summary,
        "view_attention_summary": view_attention_summary,
        "structural_correction_summary": structural_correction_summary,
        "edge_score_component_summary": edge_score_component_summary,
        "graph_nnz": int(micro_graph.nnz),
        "confidence_mean": float(np.mean(micro_conf)) if micro_conf.size > 0 else None,
        "radius_mean": float(np.mean(spatial_radius)) if spatial_radius.size > 0 else None,
        "boundary_uncertainty_mean": float(np.mean(boundary_uncertainty)) if boundary_uncertainty.size > 0 else None,
    }
    training_summary.update(metadata_scope_fields)

    logger.info(
        "[arm_rep]: finished -> "
        f"latent_dim={z_main.shape[1]}, micro_graph_nnz={micro_graph.nnz}, "
        f"confidence_mean={float(np.mean(micro_conf)) if micro_conf.size > 0 else float('nan'):.4f}"
    )
    return adata_rep
