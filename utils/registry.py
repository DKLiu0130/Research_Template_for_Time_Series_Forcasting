"""Tiny registries for datasets and models."""
from __future__ import annotations

from typing import Any, Callable, Dict

DATASET_REGISTRY: Dict[str, Callable[..., Any]] = {}
MODEL_REGISTRY: Dict[str, Callable[..., Any]] = {}


def register_dataset(name: str):
    def decorator(fn):
        DATASET_REGISTRY[name] = fn
        return fn
    return decorator


def register_model(name: str):
    def decorator(cls):
        MODEL_REGISTRY[name] = cls
        return cls
    return decorator


def get_dataset_builder(name: str):
    if name not in DATASET_REGISTRY:
        raise KeyError(
            f'Unknown dataset {name!r}. Available: {sorted(DATASET_REGISTRY)}'
        )
    return DATASET_REGISTRY[name]


def get_model_cls(name: str):
    if name not in MODEL_REGISTRY:
        raise KeyError(
            f'Unknown model {name!r}. Available: {sorted(MODEL_REGISTRY)}'
        )
    return MODEL_REGISTRY[name]
