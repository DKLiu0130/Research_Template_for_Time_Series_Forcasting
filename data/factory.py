"""Dataset factory. Import concrete modules so registries fill up."""
from __future__ import annotations

from typing import Any, Dict

from utils.registry import get_dataset_builder

# Import side-effect registrations here.
from data import example_csv  # noqa: F401


def build_dataloaders(cfg: Dict[str, Any]):
    name = cfg['data']['name']
    builder = get_dataset_builder(name)
    return builder(cfg)
