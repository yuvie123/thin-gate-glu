"""
A small GPT-style language model for Experiment B (from-scratch pretraining).

Architecture: pre-norm Transformer decoder with RMSNorm, rotary position
embeddings (RoPE), causal self-attention, and a SwiGLU feed-forward block:

    MLP(x) = W_down( silu(W_gate x) * (W_up x) )

The only non-standard part is that each of the three MLP projections can be
replaced by a rank-r factorization (LowRankLinear). The paper's method,
"thin-gate GLU", sets gate_rank > 0 and leaves up_rank = down_rank = 0 (dense).
The controls "thin-up" / "thin-down" factorize the other matrices instead.

The screening variants (grid.py `screen`) add three more ways to make a projection cheap:
grouped (fewer outputs, each shared by g hidden units), Monarch (block-structured, full rank)
and a nonlinear bottleneck between the two low-rank factors.
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
    gate_groups: int = 1         # g > 1: the gate has d_ff/g outputs, each gating g neighbouring hidden units
    up_groups: int = 1           # g > 1: the up-projection has d_ff/g outputs, each repeated g times
    down_groups: int = 1         # g > 1: g neighbouring hidden units are averaged before the down-projection
    gate_monarch: int = 0        # nb > 0: the gate is a Monarch matrix with nb blocks (full rank, few parameters)
    lowrank_init: str = "balanced"   # "balanced" (random factors) or "spectral" (SVD of a random dense init)
    bottleneck: str = "linear"       # what sits between the two factors: "linear", "silu" or "norm_silu"
    rope_base: float = 10000.0
    init_std: float = 0.02


def svd_factors(weight, rank):
    """Truncated SVD of a dense weight [out, in], split symmetrically: returns (A [rank, in], B [out, rank])
    with B @ A the best rank-r approximation."""
    U, S, Vh = torch.linalg.svd(weight.float(), full_matrices=False)
    root = S[:rank].sqrt()
    return root[:, None] * Vh[:rank], U[:, :rank] * root[None, :]


class LowRankLinear(nn.Module):
    """y = B(h) with h = A x (in -> rank) and B: rank -> out. No bias.

    bottleneck = "linear":    h = A x                     (a rank-r linear map, the paper's thin gate)
    bottleneck = "silu":      h = silu(A x)               (the gate becomes a tiny two-layer network)
    bottleneck = "norm_silu": h = silu(RMSNorm(A x))      (same, with a normalised code; adds `rank` parameters)
    init = "balanced": random factors whose product has the entry variance of a dense init.
    init = "spectral": the factors are the top-r SVD of a random dense init (a low-rank *matrix* init).
    """

    def __init__(self, in_features, out_features, rank, init_std=0.02, init="balanced", bottleneck="linear"):
        super().__init__()
        assert init in ("balanced", "spectral"), init
        assert bottleneck in ("linear", "silu", "norm_silu"), bottleneck
        self.A = nn.Linear(in_features, rank, bias=False)
        self.B = nn.Linear(rank, out_features, bias=False)
        self.act = bottleneck != "linear"
        self.norm = nn.RMSNorm(rank) if bottleneck == "norm_silu" else None
        # Balanced init: the product B @ A has the same entry variance as a dense
        # layer initialised with std=init_std, i.e. std_A^2 * std_B^2 * rank = init_std^2.
        s = (init_std ** 2 / rank) ** 0.25
        nn.init.normal_(self.A.weight, std=s)
        nn.init.normal_(self.B.weight, std=s)
        if init == "spectral":
            # Start from the best rank-r approximation of a dense random init. Note that this product has a
            # smaller Frobenius norm than a dense init (only the top r singular directions are kept).
            A, B = svd_factors(torch.randn(out_features, in_features) * init_std, rank)
            self.A.weight.data.copy_(A)
            self.B.weight.data.copy_(B)

    def forward(self, x):
        h = self.A(x)
        if self.norm is not None:
            h = self.norm(h)
        if self.act:
            h = F.silu(h)
        return self.B(h)

    @classmethod
    def from_dense(cls, weight, rank):
        """Best rank-r approximation (truncated SVD) of a dense weight [out, in]."""
        out_f, in_f = weight.shape
        A, B = svd_factors(weight, rank)
        layer = cls(in_f, out_f, rank)
        layer.A.weight.data = A.to(weight.dtype)
        layer.B.weight.data = B.to(weight.dtype)
        return layer

    @classmethod
    def from_factors(cls, U, s, V, rank):
        """From the (U, s, V) triple returned by posthoc_truncate.factorize (plain or whitened SVD):
        the rank-r approximation is (U[:, :r] * s[:r]) @ V[:r]."""
        layer = cls(V.shape[1], U.shape[0], rank)
        root = s[:rank].sqrt()
        layer.A.weight.data = (root[:, None] * V[:rank]).contiguous()
        layer.B.weight.data = (U[:, :rank] * root[None, :]).contiguous()
        return layer


class MonarchLinear(nn.Module):
    """y = L(P(R x)): two block-diagonal matrices with a fixed perfect shuffle P between them.

    R has nb blocks of [m, m] (m = in / nb) and L has nb blocks of [m, out / nb]. After R, the shuffle sends
    m/nb entries of every block into each new block, so every output depends on every input (full rank) at
    (in^2 + in*out) / nb parameters. With nb = 4, in = d and out = d_ff this is exactly the parameter count of a
    rank-d/4 factorization. The output columns come out in shuffled order, which does not matter for a gate:
    the dense up- and down-projections learn the matching order.
    """

    def __init__(self, in_features, out_features, nb, init_std=0.02):
        super().__init__()
        assert in_features % nb == 0 and out_features % nb == 0, "in and out must be multiples of nb"
        m = in_features // nb
        assert m % nb == 0, "each block must split evenly across the blocks for the shuffle"
        self.nb, self.m, self.in_features, self.out_features = nb, m, in_features, out_features
        self.R = nn.Parameter(torch.empty(nb, m, m))
        self.L = nn.Parameter(torch.empty(nb, m, out_features // nb))
        # Every (input, output) pair is connected through m/nb paths; match a dense N(0, init_std^2) entry.
        s = (init_std ** 2 / (m // nb)) ** 0.25
        nn.init.normal_(self.R, std=s)
        nn.init.normal_(self.L, std=s)

    def forward(self, x):
        lead = x.shape[:-1]
        y = x.reshape(-1, self.nb, self.m)
        y = torch.einsum("bni,nij->bnj", y, self.R)                 # block-diagonal 1
        y = y.transpose(1, 2).reshape(-1, self.nb, self.m)          # perfect shuffle across blocks
        y = torch.einsum("bni,nio->bno", y, self.L)                 # block-diagonal 2
        return y.reshape(*lead, self.out_features)


def make_linear(in_features, out_features, rank, init_std, init="balanced", bottleneck="linear"):
    if rank and rank > 0:
        return LowRankLinear(in_features, out_features, rank, init_std, init=init, bottleneck=bottleneck)
    layer = nn.Linear(in_features, out_features, bias=False)
    nn.init.normal_(layer.weight, std=init_std)
    return layer


def _expand_groups(t, g):
    """[..., n] -> [..., n * g], each value repeated g times in a row (a view plus one copy)."""
    if g == 1:
        return t
    return t.unsqueeze(-1).expand(*t.shape, g).reshape(*t.shape[:-1], t.shape[-1] * g)


class SwiGLU(nn.Module):
    def __init__(self, cfg: GPTConfig):
        super().__init__()
        d, F_, std = cfg.d_model, cfg.d_ff, cfg.init_std
        assert F_ % cfg.gate_groups == 0 and F_ % cfg.up_groups == 0 and F_ % cfg.down_groups == 0
        assert sum([cfg.gate_rank > 0, cfg.gate_groups > 1, cfg.gate_monarch > 0]) <= 1, "one gate structure at a time"
        assert not (cfg.up_rank > 0 and cfg.up_groups > 1) and not (cfg.down_rank > 0 and cfg.down_groups > 1)
        self.gate_groups, self.up_groups, self.down_groups = cfg.gate_groups, cfg.up_groups, cfg.down_groups
        # GPT-2 style: shrink the init of layers that write into the residual stream
        down_std = std / math.sqrt(2 * cfg.n_layer)
        if cfg.gate_monarch:
            self.gate = MonarchLinear(d, F_, cfg.gate_monarch, std)
        else:
            self.gate = make_linear(d, F_ // cfg.gate_groups, cfg.gate_rank, std, cfg.lowrank_init, cfg.bottleneck)
        self.up = make_linear(d, F_ // cfg.up_groups, cfg.up_rank, std, cfg.lowrank_init, cfg.bottleneck)
        self.down = make_linear(F_ // cfg.down_groups, d, cfg.down_rank, down_std, cfg.lowrank_init, cfg.bottleneck)

    def forward(self, x):
        g = _expand_groups(self.gate(x), self.gate_groups)
        u = _expand_groups(self.up(x), self.up_groups)
        h = F.silu(g) * u
        if self.down_groups > 1:      # average g neighbouring hidden units so the residual write keeps its scale
            h = h.reshape(*h.shape[:-1], h.shape[-1] // self.down_groups, self.down_groups).mean(-1)
        return self.down(h)


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
