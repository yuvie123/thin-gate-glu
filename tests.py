"""
Correctness checks. Run with:  python tests.py     (a few seconds on CPU, no downloads)

If any of these fail, do not trust experiment numbers until it is fixed.
"""

import torch

import dataclasses
import json
import os
import subprocess
import sys

from grid import arms_for, mlp_params, screen2_arms, screen3_arms, screen4_arms, screen_arms
from model import GPT, GPTConfig, LowRankLinear, MonarchLinear, SwiGLU, _expand_groups, count_params
from posthoc_truncate import factorize
from train import SIZES, thin_gates

CFG_KEYS = set(GPTConfig.__dataclass_fields__)


def config_for(arm, L, H, d, F, **kw):
    """A GPTConfig from an arm dict (grid.py keys are train.py flags; only the config ones apply here)."""
    fields = {k: v for k, v in arm.items() if k in CFG_KEYS}
    return GPTConfig(n_layer=L, n_head=H, d_model=d, seq_len=32, **{"d_ff": F, **fields, **kw})


def test_full_rank_factorization_is_exact():
    torch.manual_seed(0)
    W = torch.randn(48, 16)
    layer = LowRankLinear.from_dense(W, rank=16)
    x = torch.randn(5, 16)
    assert torch.allclose(layer(x), x @ W.T, atol=1e-4), "full-rank SVD factorization should reproduce the dense layer"


def test_truncation_error_matches_singular_values():
    torch.manual_seed(0)
    W = torch.randn(48, 16)
    r = 6
    layer = LowRankLinear.from_dense(W, rank=r)
    approx = layer.B.weight @ layer.A.weight
    s = torch.linalg.svdvals(W)
    # Eckart-Young: the squared Frobenius error equals the sum of the discarded squared singular values
    assert torch.allclose((W - approx).pow(2).sum(), s[r:].pow(2).sum(), rtol=1e-4)


def test_whitened_truncation_beats_plain_on_outputs():
    torch.manual_seed(0)
    W = torch.randn(40, 24)
    X = torch.randn(500, 24) * torch.linspace(0.1, 5.0, 24)     # inputs with very unequal feature scales
    gram = X.T @ X
    r = 6

    def output_error(U, s, V):
        return ((X @ W.T) - (X @ ((U[:, :r] * s[:r]) @ V[:r]).T)).pow(2).sum()

    plain, white = output_error(*factorize(W)), output_error(*factorize(W, gram))
    assert white < plain, "activation-aware truncation must give a smaller output error than plain SVD"
    U, s, V = factorize(W, gram)
    assert torch.allclose((U * s) @ V, W, atol=1e-3), "at full rank the whitened factorization must reproduce W"


def test_param_counts_match_formula():
    for size, (L, H, d, F) in SIZES.items():
        if size == "L":
            continue                                    # skip the big one to keep the test fast
        for name, arm in {**arms_for(size, (4,)), **screen_arms(size), **screen2_arms(size), **screen3_arms(size),
                          **screen4_arms(size)}.items():
            counted = count_params(GPT(config_for(arm, L, H, d, F)))["mlp"]
            assert counted == L * mlp_params(size, arm), f"{size}/{name}: {counted} != formula"


def test_matched_arms_are_matched():
    for size in SIZES:
        arms = arms_for(size, (2, 4, 8))
        dense = mlp_params(size, {})
        for K in (2, 4, 8):
            thin = mlp_params(size, arms[f"thin_gate_r{K}"])
            for other in (f"shrunk_r{K}", f"all_lowrank_r{K}"):
                gap = abs(mlp_params(size, arms[other]) - thin) / thin
                assert gap < 0.015, f"{size}/{other} differs from thin gate by {100 * gap:.2f}%"
            gap = abs(mlp_params(size, arms[f"reinvest_r{K}"]) - dense) / dense
            assert gap < 0.015, f"{size}/reinvest_r{K} differs from dense by {100 * gap:.2f}%"
        thin = mlp_params(size, arms["thin_gate_r4"])
        for name, arm in screen_arms(size).items():
            gap = abs(mlp_params(size, arm) - thin) / thin
            assert gap < 0.015, f"{size}/{name} differs from thin gate by {100 * gap:.2f}%"
        s2 = screen2_arms(size)
        for name in ("tied_gate", "tied_gate_relu", "thin_tied_gate_r4", "shared_gate"):
            gap = abs(mlp_params(size, s2[name]) - thin) / thin
            assert gap < 0.015, f"{size}/{name} differs from thin gate by {100 * gap:.2f}%"
        assert mlp_params(size, s2["shrunk_w400"]) == mlp_params(size, s2["tied_gate_w600"]), "starved pair not matched"
        s3 = screen3_arms(size)
        for name in ("shrunk_relu_r4", "thin_tied_relu_r4", "thin_tied_relu_r8", "thin_tied_relu_r2", "tied_gate", "pair_tied_relu",
                     "thin_gate_relu_r4", "thin_tied_relu_r4_zero", "thin_tied_relu_r4_affine"):
            gap = abs(mlp_params(size, s3[name]) - thin) / thin
            assert gap < 0.015, f"{size}/{name} differs from thin gate by {100 * gap:.2f}%"
        assert mlp_params(size, s3["tied_gate_relu_w600"]) == mlp_params(size, s2["shrunk_w400"]), "starved pair not matched"
        for name, arm in screen4_arms(size).items():
            if name.startswith(("dense_spec", "warm")):
                continue
            gap = abs(mlp_params(size, arm) - thin) / thin
            assert gap < 0.015, f"{size}/{name} differs from thin gate by {100 * gap:.2f}%"
        assert mlp_params(size, s3["dense_relu"]) == mlp_params(size, {}), "dense_relu must have the dense count"


def test_model_is_causal():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=100, n_layer=2, n_head=2, d_model=32, d_ff=64, seq_len=16, gate_rank=8)).eval()
    x = torch.randint(0, 100, (1, 16))
    y = x.clone()
    y[0, 10:] = torch.randint(0, 100, (6,))             # change only the future
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5), "outputs before position 10 must not see the future"


def test_spectral_init_is_a_truncated_svd():
    torch.manual_seed(0)
    layer = LowRankLinear(64, 96, 8, init="spectral")
    AAt = layer.A.weight @ layer.A.weight.T                       # rows of A are scaled orthonormal directions
    assert torch.allclose(AAt, torch.diag(AAt.diagonal()), atol=1e-5), "spectral init: A A^T must be diagonal"
    assert torch.linalg.matrix_rank(layer.B.weight @ layer.A.weight) == 8


def test_bottleneck_forward():
    torch.manual_seed(0)
    x = torch.randn(7, 24)
    lin, act = LowRankLinear(24, 40, 6), LowRankLinear(24, 40, 6, bottleneck="silu")
    act.load_state_dict(lin.state_dict())
    assert torch.allclose(lin(x), lin.B(lin.A(x)))
    assert torch.allclose(act(x), act.B(torch.nn.functional.silu(act.A(x))))
    normed = LowRankLinear(24, 40, 6, bottleneck="norm_silu")
    assert sum(p.numel() for p in normed.parameters()) == 6 * (24 + 40) + 6, "norm_silu adds `rank` parameters"


def test_grouped_gate_repeats_values():
    torch.manual_seed(0)
    mlp = SwiGLU(GPTConfig(n_layer=2, d_model=16, d_ff=32, gate_groups=4))
    x = torch.randn(3, 5, 16)
    g = _expand_groups(mlp.gate(x), 4)
    assert g.shape[-1] == 32
    for k in range(32):
        assert torch.equal(g[..., k], g[..., k - k % 4]), "each gate value must be shared by 4 neighbouring units"
    assert mlp(x).shape == (3, 5, 16)
    down = SwiGLU(GPTConfig(n_layer=2, d_model=16, d_ff=32, down_groups=4))
    assert down(x).shape == (3, 5, 16) and down.down.in_features == 8


def test_monarch_mixes_all_inputs():
    torch.manual_seed(0)
    layer = MonarchLinear(16, 32, nb=4)
    J = layer(torch.eye(16))                                       # row i = response to input i
    assert J.shape == (16, 32) and (J != 0).all(), "every output must depend on every input"
    assert torch.linalg.matrix_rank(J) == 16, "Monarch is full rank"
    assert sum(p.numel() for p in layer.parameters()) == 16 * 16 // 4 + 16 * 32 // 4
    assert layer(torch.randn(2, 3, 16)).shape == (2, 3, 32)


def test_thin_swap_is_exact_at_full_rank_and_keeps_optimizer_state():
    torch.manual_seed(0)
    L, H, d, F = 2, 2, 32, 64
    model = GPT(GPTConfig(vocab_size=100, n_layer=L, n_head=H, d_model=d, d_ff=F, seq_len=16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x = torch.randint(0, 100, (2, 16))
    _, loss = model(x, x)
    loss.backward()
    opt.step()
    opt.zero_grad()
    with torch.no_grad():
        before = model(x)
    thin_gates(model, opt, rank=d, grams=[None] * L, factor_wd=0.0)        # full rank: exact
    with torch.no_grad():
        after = model(x)
    assert torch.allclose(before, after, atol=1e-4), "a full-rank swap must not change the model's outputs"
    assert all(isinstance(b.mlp.gate, LowRankLinear) for b in model.blocks)
    assert count_params(model)["mlp"] == L * (2 * d * F + d * (d + F)), "two dense projections + rank-d factors"
    in_groups = [id(p) for g in opt.param_groups for p in g["params"]]
    assert sorted(in_groups) == sorted(id(p) for p in model.parameters()), "every parameter in exactly one group"
    assert model.embed.weight in opt.state, "untouched parameters keep their Adam state"
    _, loss = model(x, x)                                                  # one more step must run
    loss.backward()
    opt.step()



def test_tied_gate_forward():
    """A tied gate is the up-projection: h = act(z) * z, and it owns no gate parameters."""
    torch.manual_seed(0)
    L, H, d, F_ = 2, 2, 16, 24
    x = torch.randn(3, 5, d)
    for act, fn in (("silu", torch.nn.functional.silu), ("relu", torch.relu)):
        m = SwiGLU(GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_tie="up", gate_act=act))
        assert m.gate is None and m.tie_scale is None
        assert {n for n, _ in m.named_parameters()} == {"up.weight", "down.weight"}
        z = x @ m.up.weight.T
        want = (fn(z) * z) @ m.down.weight.T
        assert torch.allclose(m(x), want, atol=1e-6), f"tied {act} forward differs from the hand computation"
    assert torch.allclose(torch.relu(z) * z, torch.relu(z) ** 2), "the relu tie is Primer's squared ReLU"


def test_pair_tied_gate_forward():
    """Partner-gated units: h_{2i} = act(z_{2i+1}) z_{2i}, h_{2i+1} = act(z_{2i}) z_{2i+1}; no gate parameters."""
    torch.manual_seed(0)
    L, H, d, F_ = 2, 2, 16, 24
    m = SwiGLU(GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_tie="pair", gate_act="relu"))
    assert m.gate is None and {n for n, _ in m.named_parameters()} == {"up.weight", "down.weight"}
    x = torch.randn(3, 5, d)
    z = x @ m.up.weight.T
    h = torch.empty_like(z)
    h[..., 0::2] = torch.relu(z[..., 1::2]) * z[..., 0::2]
    h[..., 1::2] = torch.relu(z[..., 0::2]) * z[..., 1::2]
    assert torch.allclose(m(x), h @ m.down.weight.T, atol=1e-6), "pair-tied forward differs from the hand computation"
    # every unit's gate is a different direction from its content (unlike the self-tie)
    assert not torch.allclose(m.up.weight[0], m.up.weight[1])


def test_whitening_survives_a_singular_gram():
    """A rank-deficient calibration gram (fewer tokens than channels, or dominant channels) must not crash the
    swap; the eigendecomposition path must still be exact at full rank."""
    torch.manual_seed(0)
    W = torch.randn(24, 16)
    X = torch.randn(5, 16)                      # 5 tokens for 16 channels: the gram is singular
    gram = X.T @ X
    U, s, V = factorize(W, gram)
    approx = (U * s) @ V
    assert torch.allclose(approx @ X.T, W @ X.T, atol=1e-3), "full-rank whitened factorization must reproduce W X"
    bad = gram.clone(); bad[0, 0] = float("inf")
    U, s, V = factorize(W, bad)                 # non-finite gram: plain SVD, no crash
    assert torch.allclose((U * s) @ V, W, atol=1e-4)


def test_tied_down_and_extra_activations():
    """One-matrix block: out = (act(z) z * s) @ U with U the up weight; step tie = plain relu FFN; abs tie = z|z|."""
    torch.manual_seed(0)
    L, H, d, F_ = 2, 2, 16, 24
    base = dict(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_tie="up")
    x = torch.randn(3, 5, d)
    m = SwiGLU(GPTConfig(**base, gate_act="relu", down_tie=1, down_scale=1))
    assert m.down is None and {n for n, _ in m.named_parameters()} == {"up.weight", "down_scale"}
    z = x @ m.up.weight.T
    want = (torch.relu(z) * z * m.down_scale) @ m.up.weight * m.down_gain
    assert torch.allclose(m(x), want, atol=1e-6), "tied down-projection forward differs from the hand computation"
    assert abs(m.down_gain - 0.375 / (2 * L) ** 0.5) < 1e-9, "tied down keeps the residual-write scaling"
    m2 = SwiGLU(GPTConfig(**base, gate_act="relu", down_tie=1))
    assert sum(p.numel() for p in m2.parameters()) == d * F_, "one matrix per block"
    step = SwiGLU(GPTConfig(**base, gate_act="step")); z = x @ step.up.weight.T
    assert torch.allclose(step(x), torch.relu(z) @ step.down.weight.T, atol=1e-6), "step tie must be a plain relu FFN"
    ab = SwiGLU(GPTConfig(**base, gate_act="abs")); z = x @ ab.up.weight.T
    assert torch.allclose(ab(x), (z * z.abs()) @ ab.down.weight.T, atol=1e-6), "abs tie must be z|z|"
    bl = SwiGLU(GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_act="identity"))
    assert torch.allclose(bl(x), ((x @ bl.gate.weight.T) * (x @ bl.up.weight.T)) @ bl.down.weight.T, atol=1e-6), "bilinear"


def test_weight_spectra_and_activity():
    from train import weight_spectra, truncate_in_place
    torch.manual_seed(0)
    model = GPT(GPTConfig(n_layer=2, n_head=2, d_model=16, d_ff=24, seq_len=32, gate_tie="up", gate_act="relu"))
    state = {}
    spec = weight_spectra(model, state)
    assert set(spec["stable_rank"]) == {"up", "down", "q", "k", "v", "o"} and len(spec["stable_rank"]["up"]) == 2
    assert all(1 <= r <= 16 for r in spec["r90"]["up"]) and all(1 <= s <= 16 for s in spec["stable_rank"]["up"])
    assert all(1 <= e <= 16.01 for e in spec["erank"]["up"]) and all(0 < t <= 1 for t in spec["top8"]["q"])
    assert spec["drift"]["up"] == [None, None] and spec["gate_up_align"] == [], "first call: no previous subspace, no gate"
    spec2 = weight_spectra(model, state)
    assert all(abs(x - 1) < 1e-4 for x in spec2["drift"]["up"]), "unchanged weights: the dominant subspace does not drift"
    dense = GPT(GPTConfig(n_layer=2, n_head=2, d_model=16, d_ff=24, seq_len=32))
    sd = weight_spectra(dense, {})
    assert len(sd["gate_up_align"]) == 2 and all(0 <= a <= 1 for a in sd["gate_up_align"])
    x = torch.randint(0, 512, (2, 8))
    before = dense(x).clone()
    truncate_in_place(dense, 16, [None, None])              # full rank: exact, and the gate stays a dense Linear
    assert isinstance(dense.blocks[0].mlp.gate, torch.nn.Linear) and torch.allclose(dense(x), before, atol=1e-4)
    saved = truncate_in_place(dense, 4, [None, None])
    assert torch.linalg.matrix_rank(dense.blocks[0].mlp.gate.weight) == 4, "in-place truncation to rank 4"
    for b, w in zip(dense.blocks, saved):
        b.mlp.gate.weight.data.copy_(w)
    assert torch.allclose(dense(x), before, atol=1e-4), "restoring the saved weights undoes the probe"
    saved = truncate_in_place(dense, 4, [None, None], proj="down")
    assert torch.linalg.matrix_rank(dense.blocks[1].mlp.down.weight) == 4 and isinstance(dense.blocks[1].mlp.down, torch.nn.Linear)
    for b in model.blocks:
        b.mlp.record_stats = True
    model(torch.randint(0, 512, (2, 8)))
    assert all(0 < b.mlp.last_active < 1 for b in model.blocks), "relu self-gate must leave some units off and some on"


def test_zero_init_and_affine_start_as_the_self_gate():
    """B = 0 makes thin+tied identical to the plain tie at init; bias and bypass at 0 change nothing until learned."""
    torch.manual_seed(0)
    L, H, d, F_, r = 2, 2, 16, 24, 4
    base = GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_tie="up", gate_act="relu")
    tie = SwiGLU(base)
    zero = SwiGLU(dataclasses.replace(base, gate_rank=r, lowrank_init="zero"))
    affine = SwiGLU(dataclasses.replace(base, gate_rank=r, lowrank_init="zero", gate_bias=1, gate_bypass=1))
    for m in (zero, affine):
        m.up.load_state_dict(tie.up.state_dict()); m.down.load_state_dict(tie.down.state_dict())
    assert torch.all(zero.gate.B.weight == 0) and zero.gate.A.weight.abs().sum() > 0
    x = torch.randn(3, 5, d)
    assert torch.allclose(zero(x), tie(x), atol=1e-6), "zero-init context must start as the plain self-gate"
    assert torch.allclose(affine(x), tie(x), atol=1e-6), "zero bias and bypass must start as the plain self-gate"
    n_tie = sum(p.numel() for p in tie.parameters())
    assert sum(p.numel() for p in affine.parameters()) == n_tie + r * (d + F_) + 3 * F_
    affine(x).sum().backward()
    assert affine.gate.B.weight.grad.abs().sum() > 0 and affine.gate_bypass.grad.abs().sum() > 0, "both learn from step 1"


def test_thin_tied_gate_reduces_to_thin_gate():
    """thin+tied with the per-unit scale at zero equals the plain thin gate built from the same factors."""
    torch.manual_seed(0)
    L, H, d, F_, r = 2, 2, 16, 24, 4
    tied = SwiGLU(GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_rank=r, gate_tie="up"))
    plain = SwiGLU(GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_rank=r))
    plain.load_state_dict({k: v for k, v in tied.state_dict().items() if k != "tie_scale"})
    assert tied.tie_scale.shape == (F_,) and torch.all(tied.tie_scale == 1), "the scale starts at 1"
    x = torch.randn(3, 5, d)
    with torch.no_grad():
        tied.tie_scale.zero_()
    assert torch.allclose(tied(x), plain(x), atol=1e-6), "scale 0 must give the plain thin gate"
    assert sum(p.numel() for p in tied.parameters()) == sum(p.numel() for p in plain.parameters()) + F_


def test_shared_gate_is_one_module():
    L, H, d, F_ = 3, 2, 16, 24
    model = GPT(GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=F_, gate_shared=1, seq_len=32))
    gates = [b.mlp.gate for b in model.blocks]
    assert all(g is gates[0] for g in gates), "every layer must hold the same gate object"
    assert sum(1 for p in model.parameters() if p is gates[0].weight) == 1, "listed once in parameters()"
    counted = count_params(model)["mlp"]
    assert counted == L * 2 * d * F_ + d * F_, f"shared gate counted wrongly: {counted}"
    x = torch.randint(0, 512, (2, 8))
    logits = model(x)
    assert logits.shape == (2, 8, model.cfg.vocab_size)
    logits.sum().backward()
    assert gates[0].weight.grad is not None, "the shared gate receives gradient from every layer"


def test_kill_rule():
    """A smoke run with a reference whose margin is impossible to meet stops at the kill step and writes a
    JSON with `killed` and no `final_val_loss`, which plot.py then ignores."""
    os.makedirs("results/smoke", exist_ok=True)
    ref = "results/smoke/_test_ref.json"
    with open(ref, "w") as f:
        json.dump({"name": "ref", "log": {"val": [{"step": 10, "loss": 0.0}, {"step": 20, "loss": 0.0}]}}, f)
    cmd = [sys.executable, "train.py", "--smoke", "--name", "_test_killed", "--ref_json", ref,
           "--kill_steps", "10:0.5", "--no_compile"]
    r = subprocess.run(cmd, capture_output=True, text=True)
    assert r.returncode == 0, r.stdout[-2000:] + r.stderr[-2000:]
    with open("results/smoke/_test_killed.json") as f:
        out = json.load(f)
    assert "killed" in out and "final_val_loss" not in out, "a killed run must not report a final loss"
    assert out["killed"]["step"] == 10 and out["killed"]["ref"] == "ref"
    assert out["log"]["val"][-1]["step"] == 10, "training must stop right after the kill eval"
    assert "KILLED at step 10" in r.stdout

if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"all {len(tests)} tests passed")
