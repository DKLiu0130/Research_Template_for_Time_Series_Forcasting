# Research Template

通用 AI 科研代码骨架。目标不是绑死某个模型，而是固定一条可扩展主线：

`config -> data -> model -> exp -> run`

本模板覆盖两类常见项目：

1. 简单复现（如 DLinear）：少量数据集、单模型、标准 train/val/test。
2. 复杂方法（如 MillGNN / ReLag）：多数据入口、layers、特殊 loss、滚动评估、脚本矩阵。

## Layout

```text
research_template/
  run.py                 # 唯一主入口
  configs/               # 实验配置
  data/                  # dataset + factory + registry
  models/                # model + factory + registry
  layers/                # 可复用模块（attention/gnn/decomp/...）
  exp/                   # 训练/验证/测试编排
  utils/                 # config/seed/metrics/tools/registry
  scripts/               # smoke / launch / analysis
  tests/                 # 轻量检查
  docs/                  # 给人看的说明
  AGENT.md               # 给其他 Agent 的扩展手册（必读）
  checkpoints/           # 权重
  outputs/               # 指标与预测
```

## Quick start

```bash
cd research_template
pip install -r requirements.txt
python scripts/smoke_check.py
python run.py --config configs/example.json --dry-run
```

真正训练前，请把 `configs/example.json` 里的数据路径换成你的数据。

## Design rules

1. 新能力优先通过 registry / factory 接入，不要改散落的 if-else。
2. 复杂项目可以增加 `layers/`、`scripts/`、专项 `exp/`，但不要破坏主入口。
3. 配置用嵌套 JSON；模型如果需要 flat args，走 `utils.config.flatten_for_model`。
4. 数据集和模型都必须可替换；实验循环通过 override hooks 扩展。

更完整的增删指南见 `AGENT.md`。

## Sibling instances

- `../DLinear`：本模板的最小实例（单模型 + 单数据集）
- `../MillGNN`：本模板能力覆盖后的复杂实例（多数据入口、layers、滚动评估、脚本矩阵）

两者都应按本目录的 `AGENT.md` 继续扩展，而不是各自长成新风格。
