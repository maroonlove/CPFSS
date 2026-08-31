from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x):
        return self.dropout(x + self.pe[:, : x.size(1), :])


class ProteinGoTransformer(nn.Module):
    """Transformer classifier for multi-label GO prediction.

    Input per protein:
      seq_token: [1280]
      sequence : [L_seq, 1536]
      struc    : [L_str, 1280]
    """

    def __init__(
        self,
        input_dim: int,
        model_dim: int,
        num_heads: int,
        num_layers: int,
        num_labels: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.ln_token = nn.LayerNorm(1280)
        self.ln_seq = nn.LayerNorm(1536)
        self.ln_str = nn.LayerNorm(1280)

        self.proj_k1 = nn.Linear(1536, 1280)
        self.proj_v1 = nn.Linear(1536, 1280)
        self.proj_k2 = nn.Linear(1280, 1280)
        self.proj_v2 = nn.Linear(1280, 1280)

        self.embedding = nn.Linear(input_dim, model_dim)
        self.pos_encoder = PositionalEncoding(model_dim, dropout)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=model_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.head = nn.Sequential(
            nn.Linear(model_dim, 512),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(512, num_labels),
        )

    @staticmethod
    def _weighted_sum(q, k, v):
        score = torch.matmul(q, k.T) / math.sqrt(q.size(-1))
        score = score - score.max(dim=-1, keepdim=True).values
        attn = score.softmax(dim=-1)
        return attn @ v

    def forward(self, feat_batch):
        device = next(self.parameters()).device
        seq_agg, str_agg = [], []
        for feat in feat_batch:
            token = self.ln_token(feat["token"].to(device)).unsqueeze(0)
            seq = self.ln_seq(feat["seq"].to(device))
            struc = self.ln_str(feat["struc"].to(device))

            k1, v1 = self.proj_k1(seq), self.proj_v1(seq)
            k2, v2 = self.proj_k2(struc), self.proj_v2(struc)
            seq_agg.append(self._weighted_sum(token, k1, v1))
            str_agg.append(self._weighted_sum(token, k2, v2))

        seq_agg = torch.cat(seq_agg, dim=0)
        str_agg = torch.cat(str_agg, dim=0)
        fused = torch.cat([seq_agg, str_agg], dim=1).unsqueeze(1)
        x = self.embedding(fused)
        x = self.pos_encoder(x)
        x = self.transformer_encoder(x).mean(dim=1)
        return self.head(x)
