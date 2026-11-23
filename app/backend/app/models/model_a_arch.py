"""Model A architecture, copied verbatim from notebooks/train_model_a_colab_(updatedd).ipynb.

Cell "LABEL2ID / ModelAConfig" and the architecture cell (SinusoidalPositionalEncoding ...
ScratchTransformerClassifier). Only the ``try: from src.config import ...`` block was removed.
DO NOT EDIT: the checkpoint's state-dict keys must match these modules exactly.
"""

from __future__ import annotations

from dataclasses import dataclass

LABEL2ID: dict[str, int] = {
    "negative": 0,
    "neutral": 1,
    "positive": 2,
}

ID2LABEL: dict[int, str] = {
    0: "negative",
    1: "neutral",
    2: "positive",
}


@dataclass
class ModelAConfig:
    """Architecture hyperparameters for the scratch encoder-only Transformer."""

    vocab_size: int = 30522
    pad_token_id: int = 0
    d_model: int = 512
    num_layers: int = 8
    num_heads: int = 8
    d_ff: int = 2048
    max_seq_len: int = 256
    dropout: float = 0.10
    num_classes: int = 3
    head_tokens: int = 192
    tail_tokens: int = 64

    def __post_init__(self) -> None:
        # Token IDs are serialized to uint16 memmap storage, so vocab_size must fit in uint16.
        if self.vocab_size <= 0 or self.vocab_size > 65535:
            raise ValueError(f"vocab_size must be in [1, 65535], got {self.vocab_size}")
        if not (0 <= self.pad_token_id < self.vocab_size):
            raise ValueError(
                f"pad_token_id ({self.pad_token_id}) must be in [0, {self.vocab_size})"
            )
        if self.d_model <= 0 or self.num_heads <= 0 or self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be positive and divisible by num_heads ({self.num_heads})"
            )
        if self.num_layers <= 0 or self.d_ff <= 0 or self.num_classes <= 0:
            raise ValueError("num_layers, d_ff, and num_classes must be positive")
        if self.head_tokens < 0 or self.tail_tokens < 0:
            raise ValueError("head_tokens and tail_tokens must be non-negative")
        if self.head_tokens + self.tail_tokens != self.max_seq_len:
            raise ValueError(
                f"head_tokens ({self.head_tokens}) + tail_tokens ({self.tail_tokens}) "
                f"must equal max_seq_len ({self.max_seq_len})"
            )
        if not (0.0 <= self.dropout < 1.0):
            raise ValueError(f"dropout must be in [0.0, 1.0), got {self.dropout}")


# PyTorch encoder-only Transformer classifier built strictly from manual primitives.

import math

import torch
import torch.nn as nn
import torch.nn.functional as F



class SinusoidalPositionalEncoding(nn.Module):
    """Fixed sinusoidal positional encoding added to scaled token embeddings."""

    def __init__(self, d_model: int, max_len: int, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model, dtype=torch.float32)
        position = torch.arange(0, max_len, dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term[: d_model // 2])
        # Register as a non-trainable buffer with leading batch dim (1, max_len, d_model).
        self.register_buffer("pe", pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Match input dtype so bf16/fp16 autocast activations are not silently promoted to fp32.
        x = x + self.pe[:, : x.size(1)].to(dtype=x.dtype)
        return self.dropout(x)


class ManualMultiHeadSelfAttention(nn.Module):
    """Manual multi-head self-attention with fp32 softmax masking for AMP stability."""

    def __init__(self, d_model: int, num_heads: int, dropout: float) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError(f"d_model ({d_model}) must be divisible by num_heads ({num_heads})")
        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        self.w_q = nn.Linear(d_model, d_model, bias=True)
        self.w_k = nn.Linear(d_model, d_model, bias=True)
        self.w_v = nn.Linear(d_model, d_model, bias=True)
        self.w_o = nn.Linear(d_model, d_model, bias=True)
        self.attn_drop = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        bsz, seq_len, _ = x.shape

        # Project and reshape from (B, T, D) into (B, H, T, d_k).
        q = self.w_q(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(bsz, seq_len, self.num_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Compute masked softmax in fp32 to prevent overflow/underflow under bf16/fp16 autocast.
        scores = scores.to(torch.float32)
        if key_padding_mask is not None:
            pad_positions = ~key_padding_mask.bool()
            scores = scores.masked_fill(pad_positions[:, None, None, :], float("-inf"))

        attn_probs = F.softmax(scores, dim=-1).to(v.dtype)
        attn_probs = self.attn_drop(attn_probs)

        out = torch.matmul(attn_probs, v)
        out = out.transpose(1, 2).contiguous().view(bsz, seq_len, self.d_model)
        return self.w_o(out)


class PositionWiseFeedForward(nn.Module):
    """Two-layer position-wise FFN with GELU activation."""

    def __init__(self, d_model: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.act = nn.GELU()
        self.drop = nn.Dropout(dropout)
        self.fc2 = nn.Linear(d_ff, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.drop(self.act(self.fc1(x))))


class PreLNEncoderBlock(nn.Module):
    """Pre-LayerNorm encoder block with residual self-attention and FFN sublayers."""

    def __init__(self, d_model: int, num_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = ManualMultiHeadSelfAttention(d_model, num_heads, dropout)
        self.drop1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = PositionWiseFeedForward(d_model, d_ff, dropout)
        self.drop2 = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,
        key_padding_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = x + self.drop1(self.attn(self.ln1(x), key_padding_mask))
        x = x + self.drop2(self.ffn(self.ln2(x)))
        return x


class ScratchTransformerClassifier(nn.Module):
    """Encoder-only Transformer classifier with masked mean pooling over token states."""

    def __init__(self, config: ModelAConfig) -> None:
        super().__init__()
        self.config = config
        self.d_model = config.d_model
        self.pad_token_id = config.pad_token_id

        self.token_emb = nn.Embedding(
            config.vocab_size,
            config.d_model,
            padding_idx=config.pad_token_id,
        )
        self.pos_enc = SinusoidalPositionalEncoding(
            config.d_model,
            config.max_seq_len,
            config.dropout,
        )
        self.layers = nn.ModuleList(
            [
                PreLNEncoderBlock(
                    config.d_model,
                    config.num_heads,
                    config.d_ff,
                    config.dropout,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_ln = nn.LayerNorm(config.d_model)
        self.head_drop = nn.Dropout(config.dropout)
        self.classifier = nn.Linear(config.d_model, config.num_classes)

        self._init_weights()

    def _init_weights(self) -> None:
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.padding_idx is not None:
                    with torch.no_grad():
                        module.weight[module.padding_idx].zero_()
            elif isinstance(module, nn.LayerNorm):
                nn.init.ones_(module.weight)
                nn.init.zeros_(module.bias)

        # Scale residual output projections by 1 / sqrt(2 * num_layers) to stabilize
        # variance growth across stacked Pre-LN residual additions.
        resid_std = 0.02 / math.sqrt(2 * self.config.num_layers)
        for layer in self.layers:
            nn.init.normal_(layer.attn.w_o.weight, mean=0.0, std=resid_std)
            nn.init.normal_(layer.ffn.fc2.weight, mean=0.0, std=resid_std)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        mask = (input_ids != self.pad_token_id) if attention_mask is None else attention_mask.bool()

        # Scale embeddings by sqrt(d_model) so token magnitude balances unit sinusoidal PE.
        x = self.token_emb(input_ids) * math.sqrt(self.d_model)
        x = self.pos_enc(x)

        for layer in self.layers:
            x = layer(x, mask)

        x = self.final_ln(x)

        # Masked mean pooling averages only non-pad positions and clamps denom >= 1.0 for safety.
        mask_f = mask.unsqueeze(-1).type_as(x)
        pooled = (x * mask_f).sum(dim=1) / mask_f.sum(dim=1).clamp(min=1.0)
        return self.classifier(self.head_drop(pooled))

    def count_parameters(self) -> dict[str, int]:
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        embedding = sum(p.numel() for p in self.token_emb.parameters())
        return {
            "total": total,
            "trainable": trainable,
            "embedding": embedding,
            "non_embedding": total - embedding,
        }
