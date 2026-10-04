"""Example multivariate CSV dataset."""
from __future__ import annotations

import os
from typing import Any, Dict

import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

from data.base import BaseForecastDataset, split_loaders
from utils.registry import register_dataset


class ExampleCSVDataset(BaseForecastDataset):
    def __init__(self, root_path, flag='train', size=None, features='M',
                 data_path='example.csv', target='OT', scale=True):
        if size is None:
            self.seq_len, self.label_len, self.pred_len = 96, 48, 96
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in ['train', 'val', 'test']
        self.set_type = {'train': 0, 'val': 1, 'test': 2}[flag]
        self.features = features
        self.target = target
        self.scale = scale
        self.root_path = root_path
        self.data_path = data_path
        self.__read_data__()

    def __read_data__(self):
        self.scaler = StandardScaler()
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))

        # Simple 70/10/20 chronological split for the template example.
        n = len(df_raw)
        n_train = int(n * 0.7)
        n_val = int(n * 0.1)
        border1s = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
        border2s = [n_train, n_train + n_val, n]
        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in ['M', 'MS']:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            train_data = df_data[border1s[0]:border2s[0]]
            self.scaler.fit(train_data.values)
            data = self.scaler.transform(df_data.values)
        else:
            data = df_data.values

        self.data_x = data[border1:border2]
        self.data_y = data[border1:border2]

    def __getitem__(self, index):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len
        seq_x = torch.from_numpy(self.data_x[s_begin:s_end]).float()
        seq_y = torch.from_numpy(self.data_y[r_begin:r_end]).float()
        return seq_x, seq_y

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1


@register_dataset('example_csv')
def build_example_csv(cfg: Dict[str, Any]):
    data_cfg = cfg['data']
    train_cfg = cfg['train']
    size = [data_cfg['seq_len'], data_cfg['label_len'], data_cfg['pred_len']]
    common = dict(
        root_path=data_cfg['root_path'],
        size=size,
        features=data_cfg.get('features', 'M'),
        data_path=data_cfg['data_path'],
        target=data_cfg.get('target', 'OT'),
        scale=data_cfg.get('scale', True),
    )
    train_set = ExampleCSVDataset(flag='train', **common)
    val_set = ExampleCSVDataset(flag='val', **common)
    test_set = ExampleCSVDataset(flag='test', **common)
    loaders = split_loaders(
        train_set,
        val_set,
        test_set,
        batch_size=train_cfg['batch_size'],
        num_workers=data_cfg.get('num_workers', 0),
    )
    return {
        'train_set': train_set,
        'val_set': val_set,
        'test_set': test_set,
        'train_loader': loaders[0],
        'val_loader': loaders[1],
        'test_loader': loaders[2],
    }
