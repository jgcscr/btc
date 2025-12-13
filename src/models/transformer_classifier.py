import math
from typing import Optional

import torch
from torch import nn


class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 2048) -> None:
        super().__init__()
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.pe.size(1) < x.size(1):
            raise ValueError("Input sequence length exceeds positional encoding table")
        return x + self.pe[:, : x.size(1), :]


class TransformerDirectionClassifier(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_dim: int,
        num_heads: int,
        ffn_dim: int,
        num_layers: int,
        dropout: float,
        max_seq_len: int,
        use_layer_norm: bool = True,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(input_size, hidden_dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=ffn_dim,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_layers,
            enable_nested_tensor=False,
        )
        self.positional = SinusoidalPositionalEncoding(hidden_dim, max_len=max_seq_len)
        self.dropout = nn.Dropout(dropout)
        self.norm = nn.LayerNorm(hidden_dim) if use_layer_norm else nn.Identity()
        self.use_layer_norm = use_layer_norm
        self.head = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        if x.ndim != 3:
            raise ValueError("Expected input of shape [batch, seq_len, features]")
        features = self.input_proj(x)
        features = self.positional(features)
        encoded = self.encoder(features, mask=attn_mask)
        encoded = self.dropout(encoded)
        token = encoded[:, -1, :]
        token = self.norm(token)
        logits = self.head(token)
        return logits.squeeze(-1)


__all__ = ["TransformerDirectionClassifier"]
