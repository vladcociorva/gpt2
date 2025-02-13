import torch
import torch.nn as nn
from torch.nn import functional as F
from dataclasses import dataclass


@dataclass(frozen=True)
class GPTConfig:
    # GPT2 paper uses 50257 -- but it's not the best for cuda
    # most cuda kernels use block sizes which are power of 2 (e.g. 16, 32, 64)
    vocab_size = 50304  # number of possible tokens
    n_ctx = 1024  # max block size
    n_layers = 12  # number of decoder layers
    d_model = 768  # dimension in each bottleneck layer
    n_heads = 12  # number of attention heads


class SelfAttentionHead(nn.Module):
    def __init__(self, d_model: int, d_head: int, n_ctx: int):
        super().__init__()

        self.key = nn.Linear(d_model, d_head, bias=False)
        self.query = nn.Linear(d_model, d_head, bias=False)
        self.value = nn.Linear(d_model, d_head, bias=False)
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))

    def forward(self, idx: torch.Tensor):
        B, T, C = idx.shape

        q = self.query(idx)  # (B, T, d_head)
        k = self.key(idx)  # (B, T, d_head)
        v = self.value(idx)  # (B, T, d_head)

        affinities = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5  # (B, T, T)
        affinities = affinities.masked_fill(self.mask[:T, :T] == 0, float("-inf"))
        affinities = F.softmax(affinities, dim=-1)

        out = affinities @ v  # (B, T, d_head)
        return out


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_ctx: int):
        super().__init__()

        assert d_model % n_heads == 0
        d_head = d_model // n_heads
        self.heads = nn.ModuleList([SelfAttentionHead(d_model, d_head, n_ctx) for _ in range(n_heads)])
        self.proj = nn.Linear(d_model, d_model)  # projection back to the residual path

    def forward(self, x: torch.Tensor):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        return self.proj(x)


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(
            4 * d_model, d_model
        )  # projection back to the residual path

    def forward(self, x: torch.Tensor):
        x = self.fc(x)
        x = self.gelu(x)
        x = self.proj(x)
        return x


class DecoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_ctx: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.multi_head_attn = MultiHeadAttention(d_model, n_heads, n_ctx)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = MLP(d_model)

    def forward(self, x: torch.Tensor):
        x = x + self.multi_head_attn(self.ln1(x))
        x = x + self.mlp(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_embd = nn.Embedding(config.vocab_size, config.d_model)
        self.position_embd = nn.Embedding(config.n_ctx, config.d_model)
        self.decoder_layers = nn.ModuleList(
            [
                DecoderLayer(config.d_model, config.n_heads, config.n_ctx)
                for _ in range(config.n_layers)
            ]
        )
        self.ln = nn.LayerNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size)
        self.lm_head.weight = self.token_embd.weight

    def forward(self, idx: torch.Tensor):
        B, T = idx.shape

        assert T <= self.config.n_ctx

        tok_embd = self.token_embd(idx) # (B, T, d_model)
        pos_embd = self.position_embd(torch.arange(T, dtype=torch.long, device=idx.device)) # (T, d_model)
        x = tok_embd + pos_embd # (B, T, d_model)

        for layer in self.decoder_layers:
            x = layer(x)

        return self.lm_head(self.ln(x)) # (B, T, vocab_size)
