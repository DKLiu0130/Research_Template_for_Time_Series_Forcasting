"""Model factory. Keep construction in one place."""
from __future__ import annotations

from typing import Any, Dict

from utils.config import flatten_for_model
from utils.registry import get_model_cls

# Import side-effect registrations here.
from models import example_linear  # noqa: F401


def build_model(cfg: Dict[str, Any]):
    flat = flatten_for_model(cfg)
    name = cfg['model']['name']
    cls = get_model_cls(name)
    return cls(
        seq_len=flat.seq_len,
        pred_len=flat.pred_len,
        enc_in=flat.enc_in,
        individual=getattr(flat, 'individual', False),
    )
