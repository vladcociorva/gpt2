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


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, n_ctx: int):
        super().__init__()

        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.d_model = d_model

        self.self_attention = nn.Linear(d_model, 3 * self.d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model)  # projection back to the residual path
        self.proj.is_residual_projection = True 
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)).view(1, 1, n_ctx, n_ctx))

    def forward(self, x: torch.Tensor):
        B, T, C = x.shape

        attn = self.self_attention(x) # (B, T, 3 * d_model)
        q, k, v = attn.split(self.d_model, dim=-1) # each (B, T, d_model)

        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, T, d_head)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, T, d_head)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2) # (B, n_heads, T, d_head)

        affinities = q @ k.transpose(-2, -1) * k.shape[-1] ** -0.5 # (B, n_heads, T, d_head) @ (B, n_heads, d_head, T) -> (B, n_heads, T, T)
        affinities = affinities.masked_fill(self.mask[:, :, :T, :T] == 0, float('-inf'))
        affinities = F.softmax(affinities, dim=-1)

        y = affinities @ v # (B, n_heads, T, T) @ (B, n_heads, T, d_head) -> (B, n_heads, T, d_head)
        y = y.transpose(1, 2) # (B, T, n_heads, d_head) - for each token, the attn heads are grouped; n_heads * d_head == d_model == C
        y = y.contiguous().view(B, T, C) # contiguous to change the memory layout after transpose
        
        y = self.proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fc = nn.Linear(d_model, 4 * d_model)
        self.gelu = nn.GELU()
        self.proj = nn.Linear(
            4 * d_model, d_model
        )  # projection back to the residual path
        self.proj.is_residual_projection = True

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
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        self.apply(self._init_weights)

        self.lm_head.weight = self.token_embd.weight

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0., std=0.02)  # https://github.com/openai/gpt-2/blob/master/src/model.py#L50
            if hasattr(module, "bias") and module.bias is not None:
                torch.nn.init.zeros_(module.bias) # https://github.com/openai/gpt-2/blob/master/src/model.py#L54
            if hasattr(module, 'is_residual_projection'):
                module.weight.data *= (2 * self.config.n_layers) ** -0.5 # gpt2 paper - scale the weights of residual layers at initialization by 1/sqrt(residual_layers)

        if isinstance(module, nn.Embedding):
            if module is self.position_embd:
                torch.nn.init.normal_(module.weight, mean=0., std=0.01)  # https://github.com/openai/gpt-2/blob/master/src/model.py#L152-L153
            else:
                torch.nn.init.normal_(module.weight, mean=0., std=0.02)  # https://github.com/openai/gpt-2/blob/master/src/model.py#L154-L155

    def forward(self, idx: torch.Tensor, targets:torch.Tensor|None = None):
        B, T = idx.shape

        assert T <= self.config.n_ctx

        tok_embd = self.token_embd(idx) # (B, T, d_model)
        pos_embd = self.position_embd(torch.arange(T, dtype=torch.long, device=idx.device)) # (T, d_model)
        x = tok_embd + pos_embd # (B, T, d_model)

        for layer in self.decoder_layers:
            x = layer(x)

        logits = self.lm_head(self.ln(x)) # (B, T, vocab_size)
        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.shape[-1]), targets.view(-1))

        return logits, loss

    @torch.no_grad
    def generate(self, idx: torch.Tensor, max_new_tokens: int): 
        self.eval()
        assert idx.dim() == 2

        for _ in range(max_new_tokens):
            idx_cond = idx if idx.shape[1] <= self.config.n_ctx else idx[:, -self.config.n_ctx:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, next_token), dim=-1)

        return idx
