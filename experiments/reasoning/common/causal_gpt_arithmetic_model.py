"""Small frozen transformer retaining validated modded-nanogpt ideas.

The implementation is original glue around public PyTorch primitives.  It does
not copy the monolithic upstream trainer.  Static shapes, RoPE, QK norm, SDPA,
ReLU-squared MLPs, untied embeddings, zero output projections, value residuals,
and soft logit capping are frozen architecture choices.
"""

from __future__ import annotations

import copy
import math
from dataclasses import dataclass
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True, slots=True)
class ModelConfig:
    vocab_size: int = 259
    sequence_length: int = 128
    width: int = 128
    layers: int = 4
    heads: int = 4
    mlp_ratio: int = 4
    rope_base: float = 10_000.0
    logit_cap: float = 30.0

    def __post_init__(self) -> None:
        if min(self.vocab_size, self.sequence_length, self.width, self.layers, self.heads, self.mlp_ratio) < 1:
            raise ValueError("all model dimensions must be positive")
        if self.width % self.heads:
            raise ValueError("width must be divisible by heads")
        if (self.width // self.heads) % 2:
            raise ValueError("head dimension must be even for RoPE")


def _rope(length: int, dimension: int, base: float, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    positions = torch.arange(length, device=device, dtype=torch.float32)
    inverse = 1.0 / (base ** (torch.arange(0, dimension, 2, device=device, dtype=torch.float32) / dimension))
    angles = torch.outer(positions, inverse)
    return angles.cos().to(dtype), angles.sin().to(dtype)


def _apply_rope(value: torch.Tensor, cosine: torch.Tensor, sine: torch.Tensor) -> torch.Tensor:
    even = value[..., 0::2]
    odd = value[..., 1::2]
    rotated = torch.stack((even * cosine - odd * sine, even * sine + odd * cosine), dim=-1)
    return rotated.flatten(-2)


class Attention(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.rope_base = config.rope_base
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.projection = nn.Linear(config.width, config.width, bias=False)
        self.projection.weight.data.zero_()
        self.value_mix = nn.Parameter(torch.tensor(0.5))

    def forward(self, hidden: torch.Tensor, first_value: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        batch, length, width = hidden.shape
        qkv = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        query, key, value = qkv.unbind(dim=2)
        if first_value is None:
            first_value = value
        mix = self.value_mix.sigmoid()
        value = (1.0 - mix) * value + mix * first_value
        query = F.rms_norm(query, (self.head_dim,))
        key = F.rms_norm(key, (self.head_dim,))
        cosine, sine = _rope(length, self.head_dim, self.rope_base, hidden.device, hidden.dtype)
        query = _apply_rope(query, cosine[None, :, None, :], sine[None, :, None, :])
        key = _apply_rope(key, cosine[None, :, None, :], sine[None, :, None, :])
        output = F.scaled_dot_product_attention(
            query.transpose(1, 2), key.transpose(1, 2), value.transpose(1, 2), is_causal=True
        )
        return self.projection(output.transpose(1, 2).reshape(batch, length, width)), first_value


class Block(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.attention = Attention(config)
        hidden = config.width * config.mlp_ratio
        self.up = nn.Linear(config.width, hidden, bias=False)
        self.down = nn.Linear(hidden, config.width, bias=False)
        self.down.weight.data.zero_()

    def forward(self, hidden: torch.Tensor, first_value: torch.Tensor | None) -> tuple[torch.Tensor, torch.Tensor]:
        attention, first_value = self.attention(F.rms_norm(hidden, (hidden.shape[-1],)), first_value)
        hidden = hidden + attention
        mlp = F.relu(self.up(F.rms_norm(hidden, (hidden.shape[-1],)))).square()
        return hidden + self.down(mlp), first_value


class TinyTransformer(nn.Module):
    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.embedding = nn.Embedding(config.vocab_size, config.width)
        self.embedding_shortcut = nn.Linear(config.width, config.width, bias=False)
        self.blocks = nn.ModuleList(Block(config) for _ in range(config.layers))
        self.output = nn.Linear(config.width, config.vocab_size, bias=False)
        self.apply(self._initialize)
        for block in self.blocks:
            block.attention.projection.weight.data.zero_()
            block.down.weight.data.zero_()

    @staticmethod
    def _initialize(module: nn.Module) -> None:
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        embedding = self.embedding(tokens)
        hidden = embedding
        first_value = None
        for block in self.blocks:
            hidden, first_value = block(hidden, first_value)
        hidden = F.rms_norm(hidden + self.embedding_shortcut(embedding), (hidden.shape[-1],))
        logits = self.output(hidden).float()
        cap = self.config.logit_cap
        return cap * torch.tanh(logits / cap)

    def fresh_state(self) -> dict[str, torch.Tensor]:
        return {name: value.detach().cpu().clone() for name, value in self.state_dict().items()}

    def reset_from(self, state: dict[str, torch.Tensor]) -> None:
        self.load_state_dict(copy.deepcopy(state), strict=True)


def parameter_groups(model: TinyTransformer, *, weight_decay: float) -> list[dict[str, Any]]:
    """Keep embeddings/heads/norm-scalars separate from eligible matrices."""

    matrix, ordinary = [], []
    for name, parameter in model.named_parameters():
        if parameter.ndim == 2 and "embedding" not in name and "output" not in name:
            matrix.append(parameter)
        else:
            ordinary.append(parameter)
    # Muon remains a preregistered ablation until the eight-coalition agreement
    # test is run; both groups use AdamW in the primary discovery backend.
    return [
        {"params": matrix, "weight_decay": weight_decay, "optimizer_role": "muon_eligible_adamw_primary"},
        {"params": ordinary, "weight_decay": 0.0, "optimizer_role": "adamw"},
    ]
