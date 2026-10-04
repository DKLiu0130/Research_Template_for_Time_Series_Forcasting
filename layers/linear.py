"""Reusable linear projection helpers."""
import torch.nn as nn


def build_channel_linears(seq_len, pred_len, channels, individual=False):
    if individual:
        return nn.ModuleList([nn.Linear(seq_len, pred_len) for _ in range(channels)])
    return nn.Linear(seq_len, pred_len)
