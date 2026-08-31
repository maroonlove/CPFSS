"""Model definition for hierarchical transporter / ion-channel classification."""

from __future__ import annotations

import math

import numpy as np
import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding used after feature fusion."""

    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 2) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1), :]
        return self.dropout(x)


class ProteinTransformer(nn.Module):
    """Two-stage hierarchical classifier.

    Input features per protein:
        token: [1280]       ESM2 first-token embedding
        seq:   [L1, 1536]   ESM3 function/sequence embedding
        struc: [L2, 1280]   SaProt structure embedding

    The token embedding is used as query; sequence and structure embeddings are used
    as key/value memories. The aggregated sequence and structure vectors are
    concatenated and classified by a transformer encoder.
    """

    def __init__(
        self,
        input_dim: int = 2560,
        model_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 4,
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

        self.fc_stage1_1 = nn.Linear(model_dim, 64)
        self.fc_stage1_2 = nn.Linear(64, 5)

        # Kept for state-dict compatibility with earlier checkpoints although the
        # forward path uses fc_fusion -> fc_stage2_2 directly.
        self.fc_stage2_1 = nn.Linear(model_dim, 64)
        self.fc_fusion = nn.Linear(model_dim + 64 + 5, 64)
        self.fc_stage2_2 = nn.Linear(64, 3)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    @staticmethod
    def _weighted_sum(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
        score = torch.matmul(q, k.T) / math.sqrt(q.size(-1))
        score = score + 1e-8
        attn = score.softmax(dim=-1)
        return attn @ v

    def forward(self, feat_batch):
        device = next(self.parameters()).device
        seq_agg, str_agg = [], []

        for feature in feat_batch:
            token = self.ln_token(feature["token"].to(device)).unsqueeze(0)
            seq = self.ln_seq(feature["seq"].to(device))
            struc = self.ln_str(feature["struc"].to(device))

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

        x_stage1 = self.dropout(self.relu(self.fc_stage1_1(x)))
        stage1_output = self.fc_stage1_2(x_stage1)

        fusion = torch.cat([x, x_stage1, stage1_output], dim=1)
        x_stage2 = self.dropout(self.relu(self.fc_fusion(fusion)))
        stage2_output = self.fc_stage2_2(x_stage2)

        return stage1_output, stage2_output
