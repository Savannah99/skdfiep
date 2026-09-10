"""
TeCNO: Temporal Convolutional Network with causal (online) inference.

Adapted from: Czempiel et al., "TeCNO: Surgical Phase Recognition with
Multi-Stage Temporal Convolutional Networks", MICCAI 2020.

Key difference from MS-TCN++: causal dilated convolutions (left-pad only),
so no future context leaks.  Multi-stage refinement via softmax feedback.

Input:  (B, F, T)
Output: list of (B, C, T), one per stage — compatible with mstcn_loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class CausalDilatedLayer(nn.Module):
    def __init__(self, d_model, dilation):
        super().__init__()
        # left-only padding = (kernel_size-1)*dilation = 2*dilation for kernel=3
        self.left_pad = 2 * dilation
        self.conv     = nn.Conv1d(d_model, d_model, kernel_size=3,
                                  padding=0, dilation=dilation)
        self.norm     = nn.LayerNorm(d_model)
        self.dropout  = nn.Dropout(0.3)

    def forward(self, x):
        # x: (B, d_model, T)
        out = F.pad(x, (self.left_pad, 0))      # causal: left only
        out = F.relu(self.conv(out))
        out = self.norm(out.transpose(1, 2)).transpose(1, 2)
        return self.dropout(out) + x             # residual


class _TeCNOStage(nn.Module):
    def __init__(self, input_dim, d_model, num_layers, num_classes):
        super().__init__()
        self.proj   = nn.Conv1d(input_dim, d_model, kernel_size=1)
        self.layers = nn.ModuleList([
            CausalDilatedLayer(d_model, dilation=2 ** i)
            for i in range(num_layers)
        ])
        self.out = nn.Conv1d(d_model, num_classes, kernel_size=1)

    def forward(self, x):
        x = self.proj(x)
        for layer in self.layers:
            x = layer(x)
        return self.out(x)    # (B, num_classes, T)


class TeCNO(nn.Module):
    """Multi-stage causal temporal convolutional network."""

    def __init__(self, input_dim=768, d_model=64, num_layers=10,
                 num_classes=7, num_stages=2):
        super().__init__()
        self.stage1 = _TeCNOStage(input_dim, d_model, num_layers, num_classes)
        self.stages = nn.ModuleList([
            _TeCNOStage(num_classes, d_model, num_layers, num_classes)
            for _ in range(num_stages - 1)
        ])

    def forward(self, x, frame_idx=None):
        # x: (B, F, T)
        out     = self.stage1(x)
        outputs = [out]
        for stage in self.stages:
            out = stage(F.softmax(out, dim=1))
            outputs.append(out)
        return outputs    # list of (B, C, T)
