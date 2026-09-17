"""
A small GPT-style language model for Experiment B (from-scratch pretraining).

Architecture: pre-norm Transformer decoder with RMSNorm, rotary position
embeddings (RoPE), causal self-attention, and a SwiGLU feed-forward block:

    MLP(x) = W_down( silu(W_gate x) * (W_up x) )

The only non-standard part is that each of the three MLP projections can be
replaced by a rank-r factorization (LowRankLinear). The paper's method,
"thin-gate GLU", sets gate_rank > 0 and leaves up_rank = down_rank = 0 (dense).
The controls "thin-up" / "thin-down" factorize the other matrices instead.
"""

import math
from dataclasses import dataclass, asdict

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class GPTConfig:
    vocab_size: int = 50304      # GPT-2 vocab (50257) rounded up to a multiple of 64
    seq_len: int = 1024
    n_layer: int = 6
    n_head: int = 6
    d_model: int = 384
    d_ff: int = 1024             # SwiGLU hidden width (about 8/3 * d_model)
    gate_rank: int = 0           # 0 = dense; r > 0 = rank-r factorization
    up_rank: int = 0
    down_rank: int = 0
    rope_base: float = 10000.0
    init_std: float = 0.02


class LowRankLinear(nn.Module):
    """y = B(A x) with A: in -> rank and B: rank -> out. No bias, no nonlinearity."""

    def __init__(self, in_features, out_features, rank, init_std=0.02):
        super().__init__()
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        # Balanced init: the product B @ A has the same entry variance as a dense
        # layer initialised with std=init_std, i.e. std_A^2 * std_B^2 * rank = init_std^2.
        s = (init_std ** 2 / rank) ** 0.25
        nn.init.normal_(self.A.weight, std=s)
        nn.init.normal_(self.B.weight, std=s)

    def forward(self, x):
        return self.B(self.A(x))

    @classmethod
    def from_dense(cls, weight, rank):
        """Best rank-r approximation (truncated SVD) of a dense weight [out, in]."""
        out_f, in_f = weight.shape
        U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
        layer = cls(in_f, out_f, rank)
        sqrt_s = S[:rank].sqrt()
        layer.A.weight.data = (sqrt_s[:, None] * Vh[:rank]).to(weight.dtype)
        layer.B.weight.data = (U[:, :rank] * sqrt_s[None, :]).to(weight.dtype)
        return layer


def make_linear(in_features, out_features, rank, init_std):
    if rank and rank > 0:
        return LowRankLinear(in_features, out_features, rank, init_std)
    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(layer.weight, std=init_std)
    return layer


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        # GPT-2 style: shrink the init of layers that write into the residual stream
        down_std = cfg.init_std / math.sqrt(2 * cfg.n_layer)
        self.gate = make_linear(cfg.d_model, cfg.d_ff, cfg.gate_rank, cfg.init_std)
        self.up = make_linear(cfg.d_model, cfg.d_ff, cfg.up_rank, cfg.init_std)
        self.down = make_linear(cfg.d_ff, cfg.d_model, cfg.down_rank, down_std)

    def forward(self, x):
        return self.down(F.silu(self.gate(x)) * self.up(x))


def apply_rope(x, cos, sin):
    # x: [B, H, T, D]; rotate pairs (x1, x2) -> (x1 cos - x2 sin, x1 sin + x2 cos)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        assert cfg.d_model % cfg.n_head == 0
        self.n_head = cfg.n_head
        self.qkv = nn.Linear(cfg.d_model, 3 * cfg.d_model, bias=False)
        self.proj = nn.Linear(cfg.d_model, cfg.d_model, bias=False)
        nn.init.normal_(self.qkv.weight, std=cfg.init_std)
        nn.init.normal_(self.proj.weight, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))

    def forward(self, x, cos, sin):
        B, T, C = x.shape
        q, k, v = self.qkv(x).view(B, T, 3, self.n_head, C // self.n_head).unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)  # [B, H, T, D]
        q, k = apply_rope(q, cos, sin), apply_rope(k, cos, sin)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)  # uses FlashAttention on GPU
        return self.proj(y.transpose(1, 2).reshape(B, T, C))


class Block(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.norm1 = nn.RMSNorm(cfg.d_model)
        self.attn = Attention(cfg)
        self.norm2 = nn.RMSNorm(cfg.d_model)
        self.mlp = SwiGLU(cfg)

    def forward(self, x, cos, sin):
        x = x + self.attn(self.norm1(x), cos, sin)
        x = x + self.mlp(self.norm2(x))
        return x


class GPT(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        nn.init.normal_(self.embed.weight, std=cfg.init_std)
        self.blocks = nn.ModuleList([Block(cfg) for _ in range(cfg.n_layer)])
        self.norm = nn.RMSNorm(cfg.d_model)
        # The output head is tied to the input embedding (no extra parameters).

        head_dim = cfg.d_model // cfg.n_head
        inv_freq = 1.0 / (cfg.rope_base ** (torch.arange(0, head_dim, 2).float() / head_dim))
        angles = torch.outer(torch.arange(cfg.seq_len).float(), inv_freq)  # [T, D/2]
        self.register_buffer("rope_cos", angles.cos()[None, None], persistent=False)
        self.register_buffer("rope_sin", angles.sin()[None, None], persistent=False)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        cos, sin = self.rope_cos[:, :, :T], self.rope_sin[:, :, :T]
        x = self.embed(idx)
        for block in self.blocks:
            x = block(x, cos, sin)
        x = self.norm(x)
        logits = F.linear(x, self.embed.weight)
        if targets is None:
            return logits
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss


def count_params(model: GPT):
    """Parameter counts, split the way the paper reports them."""
    total = sum(p.numel() for p in model.parameters())
    embed = model.embed.weight.numel()
    mlp = sum(p.numel() for b in model.blocks for p in b.mlp.parameters())
    attn = sum(p.numel() for b in model.blocks for p in b.attn.parameters())
    return {"total": total, "embedding": embed, "non_embedding": total - embed, "mlp": mlp, "attn": attn}


def flops_per_token(model: GPT):
    """Rough forward+backward FLOPs per token: 6 * (matmul params) + attention term.
    The tied head is one matmul over the embedding matrix, so it counts once."""
    cfg = model.cfg
    p = count_params(model)
    matmul_params = p["non_embedding"] + p["embedding"]
    attn_term = 12 * cfg.n_layer * cfg.d_model * cfg.seq_len  # standard 6 * 2 * L * d * T estimate
    return 6 * matmul_params + attn_term


def config_dict(cfg: GPTConfig):
    return asdict(cfg)
