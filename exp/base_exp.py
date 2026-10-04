"""Base experiment loop for forecasting research."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch
import torch.nn as nn

from utils.metrics import metric
from utils.tools import EarlyStopping, adjust_learning_rate


class BaseForecastExp:
    """Train / validate / test skeleton that concrete projects can reuse."""

    def __init__(self, cfg: Dict[str, Any], model, loaders, device):
        self.cfg = cfg
        self.model = model.to(device)
        self.train_loader = loaders['train_loader']
        self.val_loader = loaders['val_loader']
        self.test_loader = loaders['test_loader']
        self.device = device

        train_cfg = cfg['train']
        self.criterion = self.build_criterion(train_cfg.get('loss', 'mse'))
        self.optimizer = self.build_optimizer(train_cfg)

    def build_criterion(self, name: str):
        if name == 'mse':
            return nn.MSELoss()
        if name == 'mae':
            return nn.L1Loss()
        raise ValueError(f'Unsupported loss: {name}')

    def build_optimizer(self, train_cfg: Dict[str, Any]):
        name = train_cfg.get('optimizer', 'adam').lower()
        lr = train_cfg['learning_rate']
        if name == 'adam':
            return torch.optim.Adam(self.model.parameters(), lr=lr)
        if name == 'sgd':
            return torch.optim.SGD(self.model.parameters(), lr=lr)
        raise ValueError(f'Unsupported optimizer: {name}')

    def setting_name(self) -> str:
        exp = self.cfg['experiment']
        data = self.cfg['data']
        model = self.cfg['model']
        train = self.cfg['train']
        return (
            f"{exp.get('name', 'exp')}_{model['name']}_{data['name']}_"
            f"sl{data['seq_len']}_pl{data['pred_len']}_bs{train['batch_size']}_"
            f"lr{train['learning_rate']}_seed{exp.get('seed', 0)}"
        )

    def _unpack_batch(self, batch):
        # Template default: (batch_x, batch_y)
        # Complex projects can override for marks / graphs / masks.
        if isinstance(batch, (list, tuple)) and len(batch) >= 2:
            return batch[0], batch[1]
        raise ValueError('Unexpected batch format')

    def _forward(self, batch_x, batch_y):
        outputs = self.model(batch_x)
        features = self.cfg['data'].get('features', 'M')
        pred_len = self.cfg['data']['pred_len']
        f_dim = -1 if features == 'MS' else 0
        outputs = outputs[:, -pred_len:, f_dim:]
        target = batch_y[:, -pred_len:, f_dim:].to(self.device)
        return outputs, target

    def validate(self) -> float:
        self.model.eval()
        losses = []
        with torch.no_grad():
            for batch in self.val_loader:
                batch_x, batch_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, target = self._forward(batch_x, batch_y)
                loss = self.criterion(outputs, target)
                losses.append(loss.item())
        self.model.train()
        return float(np.average(losses)) if losses else float('inf')

    def train(self) -> Path:
        setting = self.setting_name()
        ckpt_dir = Path(self.cfg['experiment']['checkpoints']) / setting
        early_stopping = EarlyStopping(patience=self.cfg['train'].get('patience', 3))

        for epoch in range(self.cfg['train']['epochs']):
            self.model.train()
            train_losses = []
            t0 = time.time()
            for batch in self.train_loader:
                self.optimizer.zero_grad()
                batch_x, batch_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, target = self._forward(batch_x, batch_y)
                loss = self.criterion(outputs, target)
                loss.backward()
                self.optimizer.step()
                train_losses.append(loss.item())

            train_loss = float(np.average(train_losses)) if train_losses else float('inf')
            val_loss = self.validate()
            print(
                f'Epoch: {epoch + 1:03d} | Train Loss: {train_loss:.7f} '
                f'Vali Loss: {val_loss:.7f} | Cost: {time.time() - t0:.2f}s'
            )

            early_stopping(val_loss, self.model, ckpt_dir)
            if early_stopping.early_stop:
                print('Early stopping')
                break

            adjust_learning_rate(
                self.optimizer,
                epoch + 1,
                self.cfg['train']['learning_rate'],
                self.cfg['train'].get('lradj', 'type1'),
            )

        best_path = ckpt_dir / 'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_path, map_location=self.device))
        return best_path

    def test(self) -> Dict[str, float]:
        setting = self.setting_name()
        self.model.eval()
        preds, trues = [], []
        with torch.no_grad():
            for batch in self.test_loader:
                batch_x, batch_y = self._unpack_batch(batch)
                batch_x = batch_x.float().to(self.device)
                batch_y = batch_y.float().to(self.device)
                outputs, target = self._forward(batch_x, batch_y)
                preds.append(outputs.detach().cpu().numpy())
                trues.append(target.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        trues = np.concatenate(trues, axis=0)
        names = self.cfg.get('eval', {}).get('metrics', ['mse', 'mae'])
        scores = metric(preds, trues, names=names)

        out_dir = Path(self.cfg['experiment']['outputs']) / setting
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / 'pred.npy', preds)
        np.save(out_dir / 'true.npy', trues)
        with open(out_dir / 'metrics.txt', 'w', encoding='utf-8') as f:
            for k, v in scores.items():
                f.write(f'{k}:{v}' + chr(10))

        print(' | '.join([f'Test {k.upper()}: {v:.6f}' for k, v in scores.items()]))
        print(f'Saved metrics to {out_dir / "metrics.txt"}')
        return scores
