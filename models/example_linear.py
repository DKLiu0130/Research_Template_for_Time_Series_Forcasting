"""Tiny example model for the research template."""
import torch
import torch.nn as nn

from layers.linear import build_channel_linears
from utils.registry import register_model


@register_model('ExampleLinear')
class ExampleLinear(nn.Module):
    def __init__(self, seq_len, pred_len, enc_in, individual=False, **kwargs):
        super().__init__()
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.channels = enc_in
        self.individual = individual
        self.proj = build_channel_linears(seq_len, pred_len, enc_in, individual)

    def forward(self, x):
        # x: [B, L, C] -> [B, H, C]
        x = x.permute(0, 2, 1)
        if self.individual:
            out = torch.zeros(x.size(0), self.channels, self.pred_len, device=x.device, dtype=x.dtype)
            for i in range(self.channels):
                out[:, i, :] = self.proj[i](x[:, i, :])
        else:
            out = self.proj(x)
        return out.permute(0, 2, 1)
