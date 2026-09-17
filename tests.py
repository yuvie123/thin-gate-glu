"""
Correctness checks. Run with:  python tests.py     (a few seconds on CPU, no downloads)

If any of these fail, do not trust experiment numbers until it is fixed.
"""

import torch

from grid import arms_for, mlp_params
from model import GPT, GPTConfig, LowRankLinear, count_params
from posthoc_truncate import factorize
from train import SIZES


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
        for name, arm in arms_for(size, (4,)).items():
            cfg = GPTConfig(n_layer=L, n_head=H, d_model=d, d_ff=arm.get("d_ff", F), seq_len=32,
                            gate_rank=arm.get("gate_rank", 0), up_rank=arm.get("up_rank", 0),
                            down_rank=arm.get("down_rank", 0))
            counted = count_params(GPT(cfg))["mlp"]
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


def test_model_is_causal():
    torch.manual_seed(0)
    model = GPT(GPTConfig(vocab_size=100, n_layer=2, n_head=2, d_model=32, d_ff=64, seq_len=16, gate_rank=8)).eval()
    x = torch.randint(0, 100, (1, 16))
    y = x.clone()
    y[0, 10:] = torch.randint(0, 100, (6,))             # change only the future
    with torch.no_grad():
        a, b = model(x), model(y)
    assert torch.allclose(a[0, :10], b[0, :10], atol=1e-5), "outputs before position 10 must not see the future"


if __name__ == "__main__":
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    for t in tests:
        t()
        print("ok  ", t.__name__)
    print(f"all {len(tests)} tests passed")
