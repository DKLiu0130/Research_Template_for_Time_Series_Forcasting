"""Config helpers for research experiments."""
from __future__ import annotations

import copy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, Mapping, MutableMapping, Optional


def _deep_update(base: MutableMapping[str, Any], overrides: Mapping[str, Any]) -> MutableMapping[str, Any]:
    for key, value in overrides.items():
        if (
            key in base
            and isinstance(base[key], dict)
            and isinstance(value, Mapping)
        ):
            _deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(path: str | Path, overrides: Optional[Mapping[str, Any]] = None) -> Dict[str, Any]:
    path = Path(path)
    with open(path, 'r', encoding='utf-8-sig') as f:
        cfg = json.load(f)
    if overrides:
        cfg = _deep_update(cfg, overrides)
    return cfg


def save_config(cfg: Mapping[str, Any], path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(cfg, f, indent=2, ensure_ascii=False)
        f.write(chr(10))


def config_get(cfg: Mapping[str, Any], dotted_key: str, default: Any = None) -> Any:
    cur: Any = cfg
    for part in dotted_key.split('.'):
        if not isinstance(cur, Mapping) or part not in cur:
            return default
        cur = cur[part]
    return cur


def to_namespace(obj: Any) -> Any:
    if isinstance(obj, Mapping):
        return SimpleNamespace(**{k: to_namespace(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [to_namespace(v) for v in obj]
    return obj


def flatten_for_model(cfg: Mapping[str, Any]) -> SimpleNamespace:
    """Flatten nested config into the flat attribute style many papers use."""
    flat: Dict[str, Any] = {}
    flat.update(copy.deepcopy(cfg.get('experiment', {})))
    flat.update(copy.deepcopy(cfg.get('data', {})))
    flat.update(copy.deepcopy(cfg.get('train', {})))
    flat.update(copy.deepcopy(cfg.get('eval', {})))

    model_cfg = copy.deepcopy(cfg.get('model', {}))
    params = model_cfg.pop('params', {}) or {}
    flat['model_name'] = model_cfg.get('name')
    flat.update(model_cfg)
    flat.update(params)
    return SimpleNamespace(**flat)
