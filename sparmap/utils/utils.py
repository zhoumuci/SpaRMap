from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

def json_safe(obj: Any):
    if isinstance(obj, Path):
        return str(obj)
    if obj is pd.NA:
        return None
    if isinstance(obj, dict):
        return {str(k): json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [json_safe(v) for v in obj]
    if isinstance(obj, pd.DataFrame):
        return [json_safe(v) for v in obj.to_dict(orient="records")]
    if isinstance(obj, pd.Series):
        return json_safe(obj.tolist())
    if isinstance(obj, np.ndarray):
        return json_safe(obj.tolist())
    if isinstance(obj, (np.integer, np.floating, np.bool_)):
        return json_safe(obj.item())
    if isinstance(obj, float):
        return float(obj) if math.isfinite(float(obj)) else None
    if isinstance(obj, bool):
        return bool(obj)
    if isinstance(obj, int):
        return int(obj)
    try:
        if pd.isna(obj):
            return None
    except Exception:
        pass
    return obj

def h5ad_safe_uns_value(value: Any):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(k): h5ad_safe_uns_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return h5ad_safe_uns_value(value.tolist())
    if isinstance(value, (pd.Series, pd.Index)):
        return h5ad_safe_uns_value(value.tolist())
    if isinstance(value, (list, tuple)):
        values = list(value)
        if len(values) == 0:
            return []
        if all(isinstance(v, str) for v in values):
            return [str(v) for v in values]
        if all(isinstance(v, (bool, np.bool_)) for v in values):
            return [bool(v) for v in values]
        if all(isinstance(v, (int, np.integer)) and not isinstance(v, (bool, np.bool_)) for v in values):
            return [int(v) for v in values]
        if all(isinstance(v, (int, float, np.integer, np.floating)) and not isinstance(v, (bool, np.bool_)) for v in values):
            out = []
            for item in values:
                number = float(item)
                out.append(number if math.isfinite(number) else np.nan)
            return out
        return json.dumps(json_safe(values), ensure_ascii=False, allow_nan=False)
    if value is None or value is pd.NA:
        return ""
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        out = float(value)
        return out if math.isfinite(out) else np.nan
    if isinstance(value, (np.bool_,)):
        return bool(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else np.nan
    if isinstance(value, (str, int, bool)):
        return value
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return str(value)

def save_json(obj: Any, path: str | Path) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(json_safe(obj), f, ensure_ascii=False, indent=2, allow_nan=False)

def finite_float32(values: np.ndarray, *, posinf: float = 0.0, neginf: float = 0.0) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).reshape(-1)
    out = np.nan_to_num(out, nan=0.0, posinf=float(posinf), neginf=float(neginf))
    return out.astype(np.float32, copy=False)

def summary_stats(values: np.ndarray | None) -> dict[str, Any]:
    if values is None:
        return {
            "status": "missing",
            "count": 0,
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q05": None,
            "q95": None,
        }
    x = np.asarray(values, dtype=np.float64).reshape(-1)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return {
            "status": "missing",
            "count": int(x.size),
            "mean": None,
            "std": None,
            "min": None,
            "max": None,
            "median": None,
            "q05": None,
            "q95": None,
        }
    return {
        "status": "ok",
        "count": int(finite.size),
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite)),
        "min": float(np.min(finite)),
        "max": float(np.max(finite)),
        "median": float(np.median(finite)),
        "q05": float(np.quantile(finite, 0.05)),
        "q95": float(np.quantile(finite, 0.95)),
    }

def normalize_weights(weights: np.ndarray) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w[~np.isfinite(w)] = 0.0
    w = np.maximum(w, 0.0)
    total = float(np.sum(w))
    if total <= 1e-12:
        if w.size == 0:
            return np.zeros((0,), dtype=np.float32)
        return np.full(w.shape, 1.0 / float(w.size), dtype=np.float32)
    return (w / total).astype(np.float32, copy=False)

def effective_gene_number(weights: np.ndarray) -> float:
    w = np.asarray(weights, dtype=np.float64).reshape(-1)
    w = np.where(np.isfinite(w) & (w > 0.0), w, 0.0)
    denom = float(np.sum(w * w))
    if denom <= 1e-12:
        return 0.0
    total = float(np.sum(w))
    return float((total * total) / denom)

def sample_index_groups(n_values: int, sample_ids: np.ndarray | None) -> list[np.ndarray]:
    n = int(n_values)
    if n <= 0:
        return []
    if sample_ids is None:
        return [np.arange(n, dtype=np.int64)]
    sid = np.asarray(sample_ids, dtype=object).astype(str).reshape(-1)
    if sid.shape[0] != n:
        return [np.arange(n, dtype=np.int64)]
    return [np.where(sid == sample)[0].astype(np.int64, copy=False) for sample in pd.Index(sid).unique().tolist()]

def bh_qvalues(p_values: np.ndarray) -> np.ndarray:
    p = np.asarray(p_values, dtype=np.float64).reshape(-1)
    q = np.full_like(p, np.nan, dtype=np.float64)
    valid = np.isfinite(p)
    if not np.any(valid):
        return q.astype(np.float32)
    pv = np.clip(p[valid], 0.0, 1.0)
    order = np.argsort(pv, kind="mergesort")
    ranked = pv[order]
    n = ranked.size
    adj = ranked * n / np.arange(1, n + 1, dtype=np.float64)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    qv = np.empty_like(adj)
    qv[order] = adj
    q[valid] = qv
    return q.astype(np.float32, copy=False)
