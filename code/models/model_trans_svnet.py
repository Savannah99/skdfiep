"""
Trans-SVNet: hybrid embedding aggregation Transformer for surgical phase
recognition (Gao et al., MICCAI 2021 / IJCARS 2022 journal extension).

Faithful reproduction of the phase-classification path only (no anticipation
head — this project only does causal recognition, not anticipation). Ported
from the official implementation's `transformer2_3_1.py` / `tecno_trans.py`
(https://github.com/YuemingJin/Trans-SVNet_Journal), verified causal:

  - Stage 1 (TeCNO, `model_tecno.py`) is pretrained separately and FROZEN
    (its logits are detached before feeding the transformer) — matches the
    two-stage training procedure in the reference code.
  - Stage 2 (this file) is a 1-layer Transformer:
      * encoder self-attention runs over a causal sliding window of the
        last `len_q` TeCNO logit vectors (only past + current frame, the
        reference code left-pads with zeros for the first len_q-1 frames of
        each video — never reaches into the future)
      * decoder cross-attention uses the CURRENT frame's own raw spatial
        (backbone) embedding, linearly projected + tanh, as a single-token
        query against the encoder's window output — this is the "active
        query based on spatial information" the paper describes.
  - d_model = num_classes (matches the reference: TeCNO logits are already
    num_classes-dim, and the spatial embedding is projected down to the
    same size before being used as the query).

Input:  tecno_logits (B, T, num_classes) — TeCNO's frozen output, ALREADY
        transposed to (B, T, C); spatial_feats (B, T, spatial_dim) — raw
        backbone features for the same frames.
Output: (B, T, num_classes) refined logits.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class _ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v):
        # q: (N, H, Lq, d_k), k: (N, H, Lk, d_k), v: (N, H, Lk, d_v)
        scores = torch.matmul(q, k.transpose(-1, -2)) / (self.d_k ** 0.5)
        attn = F.softmax(scores, dim=-1)
        return torch.matmul(attn, v)


class _MultiHeadAttention(nn.Module):
    def __init__(self, d_model, d_k, d_v, n_heads):
        super().__init__()
        self.n_heads, self.d_k, self.d_v = n_heads, d_k, d_v
        self.w_q = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_k = nn.Linear(d_model, d_k * n_heads, bias=False)
        self.w_v = nn.Linear(d_model, d_v * n_heads, bias=False)
        self.fc = nn.Linear(n_heads * d_v, d_model, bias=False)
        self.attn = _ScaledDotProductAttention(d_k)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, q_in, k_in, v_in):
        # q_in: (N, Lq, d_model), k_in/v_in: (N, Lk, d_model)
        N = q_in.size(0)
        residual = q_in
        q = self.w_q(q_in).view(N, -1, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(k_in).view(N, -1, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(v_in).view(N, -1, self.n_heads, self.d_v).transpose(1, 2)
        context = self.attn(q, k, v)  # (N, H, Lq, d_v)
        context = context.transpose(1, 2).reshape(N, -1, self.n_heads * self.d_v)
        out = self.fc(context)
        return self.norm(out + residual)


class _PoswiseFFN(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.ReLU(),
            nn.Linear(d_ff, d_model, bias=False),
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        return self.norm(self.fc(x) + x)


class _EncoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.self_attn = _MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.ffn = _PoswiseFFN(d_model, d_ff)

    def forward(self, x):
        x = self.self_attn(x, x, x)
        return self.ffn(x)


class _DecoderLayer(nn.Module):
    def __init__(self, d_model, d_ff, d_k, d_v, n_heads):
        super().__init__()
        self.cross_attn = _MultiHeadAttention(d_model, d_k, d_v, n_heads)
        self.ffn = _PoswiseFFN(d_model, d_ff)

    def forward(self, query, enc_out):
        x = self.cross_attn(query, enc_out, enc_out)
        return self.ffn(x)


def _causal_windows(x, len_q):
    """x: (B, T, D) -> (B, T, len_q, D), window i = x[max(0,i-len_q+1):i+1],
    left-padded with zeros for the first len_q-1 positions (never looks
    into the future)."""
    B, T, D = x.shape
    pad = x.new_zeros(B, len_q - 1, D)
    x_padded = torch.cat([pad, x], dim=1)          # (B, T+len_q-1, D)
    windows = x_padded.unfold(1, len_q, 1)          # (B, T, D, len_q)
    return windows.permute(0, 1, 3, 2).contiguous()  # (B, T, len_q, D)


class TransSVNetStage(nn.Module):
    """The Transformer refinement stage only — TeCNO (stage 1) is separate
    and expected to be frozen/detached by the caller."""

    def __init__(self, num_classes=7, spatial_dim=2048, d_ff=32,
                 n_heads=8, len_q=30):
        super().__init__()
        self.len_q = len_q
        d_k = d_v = d_ff
        self.spatial_proj = nn.Linear(spatial_dim, num_classes, bias=False)
        self.encoder = _EncoderLayer(num_classes, d_ff, d_k, d_v, n_heads)
        self.decoder = _DecoderLayer(num_classes, d_ff, d_k, d_v, n_heads)

    def forward(self, tecno_logits, spatial_feats):
        # tecno_logits: (B, T, num_classes), spatial_feats: (B, T, spatial_dim)
        B, T, _ = tecno_logits.shape
        windows = _causal_windows(tecno_logits, self.len_q)      # (B, T, len_q, C)
        windows = windows.view(B * T, self.len_q, -1)             # (B*T, len_q, C)
        enc_out = self.encoder(windows)                           # (B*T, len_q, C)

        query = torch.tanh(self.spatial_proj(spatial_feats))      # (B, T, C)
        query = query.view(B * T, 1, -1)                          # (B*T, 1, C)

        dec_out = self.decoder(query, enc_out)                    # (B*T, 1, C)
        return dec_out.view(B, T, -1)                             # (B, T, C)
