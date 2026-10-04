"""One-command entry for the research template."""
from __future__ import annotations

import argparse
from pathlib import Path

from data.factory import build_dataloaders
from exp.forecast_exp import ForecastExp
from models.factory import build_model
from utils.config import load_config, save_config
from utils.device import resolve_device
from utils.seed import set_seed


def parse_args():
    parser = argparse.ArgumentParser(description='Research template entry')
    parser.add_argument('--config', type=str, default='configs/example.json')
    parser.add_argument('--device', type=str, default=None)
    parser.add_argument('--seed', type=int, default=None)
    parser.add_argument('--epochs', type=int, default=None)
    parser.add_argument('--batch-size', type=int, default=None)
    parser.add_argument('--learning-rate', type=float, default=None)
    parser.add_argument('--pred-len', type=int, default=None)
    parser.add_argument('--seq-len', type=int, default=None)
    parser.add_argument('--dry-run', action='store_true', help='Only validate config and construction')
    return parser.parse_args()


def apply_cli_overrides(cfg, args):
    if args.device is not None:
        cfg['experiment']['device'] = args.device
    if args.seed is not None:
        cfg['experiment']['seed'] = args.seed
    if args.epochs is not None:
        cfg['train']['epochs'] = args.epochs
    if args.batch_size is not None:
        cfg['train']['batch_size'] = args.batch_size
    if args.learning_rate is not None:
        cfg['train']['learning_rate'] = args.learning_rate
    if args.pred_len is not None:
        cfg['data']['pred_len'] = args.pred_len
    if args.seq_len is not None:
        cfg['data']['seq_len'] = args.seq_len
    return cfg


def main():
    args = parse_args()
    cfg = load_config(args.config)
    cfg = apply_cli_overrides(cfg, args)

    set_seed(cfg['experiment'].get('seed', 2021))
    device = resolve_device(cfg['experiment'].get('device', 'auto'))

    print('=' * 64)
    print(f"Experiment: {cfg['experiment'].get('name')}")
    print(f"Model: {cfg['model']['name']} | Data: {cfg['data']['name']}")
    print(
        f"seq_len={cfg['data']['seq_len']} pred_len={cfg['data']['pred_len']} "
        f"batch_size={cfg['train']['batch_size']} lr={cfg['train']['learning_rate']}"
    )
    print(f'device={device}')
    print('=' * 64)

    loaders = build_dataloaders(cfg)
    model = build_model(cfg)

    if args.dry_run:
        print('Dry-run OK: config/data/model constructed successfully.')
        print(f'Model params: {sum(p.numel() for p in model.parameters())}')
        return

    exp = ForecastExp(cfg, model, loaders, device)
    setting = exp.setting_name()
    out_dir = Path(cfg['experiment']['outputs']) / setting
    out_dir.mkdir(parents=True, exist_ok=True)
    save_config(cfg, out_dir / 'resolved_config.json')

    exp.train()
    metrics = exp.test()
    print('Done.')
    print(metrics)


if __name__ == '__main__':
    main()
