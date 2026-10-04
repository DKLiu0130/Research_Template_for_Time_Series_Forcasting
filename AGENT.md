# AGENT.md — How To Extend This Research Template

> Audience: other coding agents / future collaborators.
> Read this before adding a model, dataset, loss, or experiment variant.

## 0. Non-negotiable architecture

Always preserve this call chain:

```text
run.py
  -> utils.config.load_config
  -> data.factory.build_dataloaders
  -> models.factory.build_model
  -> exp.<Experiment>.train/test
```

If a change cannot fit this chain, add a hook or a specialized experiment class.
Do **not** create a second unrelated entrypoint unless the project truly needs
multiple lifecycle modes (example: MillGNN MoFo training vs ReLag resume/inference).
Even then, keep `run.py` as the dispatcher.

### Directory responsibilities

| Path | Own | Do not put here |
|---|---|---|
| `configs/` | experiment JSON | large data, checkpoints |
| `data/` | datasets, splits, loaders, data registry | model math |
| `models/` | top-level nn.Module + model registry | one-off scripts |
| `layers/` | reusable blocks shared by models | experiment orchestration |
| `exp/` | train/val/test/rolling loops | raw CSV parsing |
| `utils/` | config/seed/metrics/checkpoint/loss helpers | dataset-specific IO |
| `scripts/` | launchers, smoke tests, analysis | core training logic |
| `tests/` | fast correctness checks | overnight benchmarks |
| `docs/` | human notes, audits, evidence | executable source of truth |

## 1. Add a dataset

### Minimal checklist

1. Create `data/<my_dataset>.py`.
2. Implement a `Dataset` (or adapter) returning model-ready batches.
3. Register builder with `@register_dataset('name')`.
4. Import the module in `data/factory.py` so registration happens.
5. Add `configs/<exp>.json` with `data.name = "name"`.
6. Add a smoke path: either `scripts/smoke_check.py` extension or a tiny test.

### Required builder contract

`build_<dataset>(cfg) -> dict` must return at least:

```python
{
  'train_loader': DataLoader,
  'val_loader': DataLoader,
  'test_loader': DataLoader,
}
```

Optional but recommended keys:

- `train_set` / `val_set` / `test_set`
- `scaler`
- `meta` (freq, adjacency, column names, split borders)

### Batch contract

Default template experiment expects:

```python
batch_x, batch_y = batch
# batch_x: [B, L, C]
# batch_y: [B, label_len + pred_len, C]
```

If your dataset needs more fields (time marks, graph, mask, group ids):

1. Keep returning a tuple/dict from the dataset.
2. Override `exp` methods:
   - `_unpack_batch`
   - `_forward`
3. Document the batch schema in the dataset module docstring.

### Split / scaling rules

- Fit scalers on **train only**.
- Keep split borders explicit and commented (paper protocol vs custom).
- If a dataset needs external metadata (like MillGNN TFB meta), put IO in `data/`
  and pass resolved tensors/meta into `exp/`, never parse CSV inside the model.

## 2. Add a model

### Minimal checklist

1. Put reusable operators in `layers/`.
2. Put the top-level model in `models/<my_model>.py`.
3. Register with `@register_model('ModelName')`.
4. Import it in `models/factory.py`.
5. Add model params under `configs/*.json -> model.params`.
6. Ensure `build_model(cfg)` can construct it without hidden globals.

### Model contract

Prefer:

```python
class MyModel(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, **params): ...
    def forward(self, x, *extra): ...
```

Rules:

- Top-level model owns the paper method story.
- `layers/` owns reusable math (attention, gnn, decomp, graph learning).
- No dataset path reading inside `models/`.
- If construction needs verified external artifacts (e.g. adjacency), pass them
  explicitly through factory kwargs. Fail loud if missing.

### When models diverge a lot

Use a project-local factory branch, but keep one registry surface:

```python
def build_model(cfg, **context):
    name = cfg['model']['name']
    if name == 'ReLag':
        return ReLag.from_config(cfg, context['verified_adjacency'])
    return get_model_cls(name)(...)
```

## 3. Add / change an experiment loop

Start from `exp/base_exp.py`.

Override only what you must:

- `_unpack_batch` for richer batches
- `_forward` for marks / teacher forcing / graph inputs
- `build_criterion` for custom losses
- `train` / `validate` / `test` for resume, rolling forecast, multi-stage loops

Patterns covered by MillGNN-like complexity:

1. Standard fit/validate/test (DLinear)
2. Frequency loss + early stopping + LR schedule (MillGNN MoFo)
3. Resume / inference-only / smoke-batch limits (ReLag lifecycle)
4. Rolling forecast evaluation (`scripts/` + specialized exp helpers)

If you add a new lifecycle, expose it via `run.py` flags and keep config validation
in `utils/`.

## 4. Add a config

Nested schema:

```json
{
  "experiment": {"name": "...", "seed": 2021, "device": "auto", ...},
  "data": {"name": "...", "seq_len": 96, "pred_len": 96, ...},
  "model": {"name": "...", "enc_in": 7, "params": {...}},
  "train": {"epochs": 10, "batch_size": 32, "learning_rate": 1e-3, ...},
  "eval": {"metrics": ["mse", "mae"]}
}
```

Rules:

- Put method-specific knobs under `model.params` or a dedicated top-level section
  (`graph`, `execution`, `loss`) when the project grows.
- CLI overrides must be applied in one place (`run.py` or `utils/config.py`).
- Save resolved config into `outputs/<setting>/resolved_config.json`.

## 5. Add scripts without poisoning the core

Allowed in `scripts/`:

- smoke tests
- multi-run launchers
- preflight / GPU checks
- result aggregation

Not allowed:

- secretly reimplementing training loops that diverge from `exp/`
- hard-coded absolute machine paths as required inputs

## 6. Mapping from messy research repos

If you are cleaning a MillGNN-like repo, map old folders like this:

| Old messy area | New home |
|---|---|
| `data_provider/*` | `data/` (+ keep advanced helpers as modules) |
| `models/*` + `layers/*` | keep split: top models vs reusable layers |
| `exp/*` | keep, but inherit/adapt `BaseForecastExp` ideas |
| ad-hoc root scripts | `scripts/` |
| scattered JSON | `configs/` with provenance notes in `docs/` |
| giant AGENTS notes | project `AGENTS.md` + this template `AGENT.md` |

Coverage goal: the template must be able to host both

- DLinear: almost no `layers/`, one dataset, one exp class
- MillGNN: rich `layers/`, multiple data entrypoints, specialized exp + scripts

## 7. Definition of done for an extension

Before claiming a model/dataset is integrated:

1. `python -m py_compile` on touched files succeeds.
2. `python run.py --config ... --dry-run` constructs data/model (or project equivalent).
3. One smoke train path exists (`--epochs 1` or smoke-batch flags).
4. README / AGENT notes list the new registry name and batch/config contract.
5. No new absolute local path is required for core execution.

## 8. Suggested copy workflow for a new project

```bash
cp -r research_template MyPaperMethod
cd MyPaperMethod
# delete example model/dataset if unused
# add data/<...>.py and models/<...>.py
# register them in factories
# write configs/<...>.json
# run smoke + short train
```

Keep project-specific research memory in `AGENTS.md` or `docs/`,
but keep extension mechanics in sync with this file.
