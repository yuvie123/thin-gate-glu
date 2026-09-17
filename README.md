# Thin-gate GLU: do GLU gates need full rank?

ICLR 2027 submission project. Abstract due **Sep 18 AOE (Sat Sep 19, 07:59 EDT)**, paper due **Sep 25 AOE (Sat Sep 26, 07:59 EDT)**.
The full plan (ideas, abstract, titles, schedule, go/no-go) is in [PLAN.md](PLAN.md). PC setup is in [SETUP_PC.md](SETUP_PC.md). Reading list: [related.md](related.md).

**Hypothesis.** In a SwiGLU block `down( silu(gate x) * (up x) )`, the gate only *selects* hidden units, while up/down carry *content*. So the gate should tolerate a low-rank factorization better than up or down. The hypothesis may be false; the experiments are designed to find out either way.

**Machines.** MacBook = editing, writing, plotting, `--smoke` checks. PC with the RTX 3070 = everything else.

## Files
| File | Purpose |
|---|---|
| `posthoc_truncate.py`, `run_posthoc.sh` | **Exp. A**: truncate gate vs up vs down in pretrained models, no training |
| `model.py`, `train.py`, `data.py`, `grid.py` | **Exp. B**: pretrain small GPTs from scratch, all arms and seeds |
| `heal.py`, `run_heal.sh` | **Exp. C**: truncate a pretrained model, train only the new factors |
| `bench.py` | measured tokens/s and memory per arm |
| `plot.py` | results/*.json -> `paper/figures/*.pdf`, `paper/tables/*.tex` |
| `tests.py` | correctness tests; run after every code change |
| `paper/` | ICLR 2027 template (`main.tex`), official style files untouched |

Every script has `--help`, and `--smoke` for a sub-minute toy check that needs no downloads.

## Commands, in order (GPU machine)

**Day 1**
```bash
python tests.py
python posthoc_truncate.py --model HuggingFaceTB/SmolLM2-135M            # minutes; first real result
python data.py --num_train_shards 6                                       # ~1.4 GB of tokens
python train.py --name throughput_check --size S --tokens 20e6 --out_dir results/scratch   # note the tok/s it prints
python grid.py --stage pilot --run                                        # overnight, inside tmux
```
*Baseline sanity:* the "baseline perplexity" printed for SmolLM2-135M on wikitext-2 should be a plausible number for a 135M model (roughly tens, not hundreds or thousands). If it is wildly off, stop: the evaluation is broken and nothing downstream can be trusted.

*Budget:* from the tok/s of the throughput check, one size-S run takes `300e6 / tok_per_s` seconds. If that is more than ~1.5 h, lower `TOKENS["S"]` in `grid.py` **before** starting the grid, and never change it between arms.

**Day 2**
```bash
bash run_posthoc.sh            # plain SVD, all models
bash run_posthoc.sh --whiten   # activation-aware SVD, all models
python plot.py
```

**Day 3: go / no-go (decide by 6 pm Sunday Sep 20)**. Look at `paper/figures/posthoc_*.pdf` and the pilot runs:
- *Noise floor* = |S_dense_s0 − S_dense_s1| in final val loss. A gap smaller than this is not a result.
- **GO** if (i) in most models the gate curve sits clearly below up and down at equal rank (whitened SVD is the one that counts), **or** (ii) thin_gate_r4 is within the noise floor of dense while thin_up_r4 / thin_down_r4 are clearly worse, or thin_gate_r4 beats shrunk_r4.
- **PIVOT** if a different projection is the cheap one: same paper, method renamed; the submitted title and abstract remain true.
- **NO-GO** if everything is symmetric and low-rank just loses: withdraw or retarget a workshop. Don't stretch claims.

**Day 4-5**
```bash
python grid.py --stage main_S --run      # key arms first, 3 seeds
python grid.py --stage main_M --run
bash run_heal.sh HuggingFaceTB/SmolLM2-135M
bash run_heal.sh HuggingFaceTB/SmolLM2-360M --micro_bs 1
python bench.py --size S && python bench.py --size M
python grid.py --stage lr --run          # if time: shows the comparison doesn't hinge on one learning rate
```
Optional zero-shot accuracy for a converted/truncated model: see `lm_eval --help` (tasks such as `arc_easy,hellaswag,piqa`). Only add this if A-C are finished.

**Day 6+**: `python plot.py`, then write. Experiments freeze Wednesday noon.

## Rules that keep the paper honest
1. Every number in the paper comes from a JSON file in `results/` via `plot.py`. No hand-typed numbers.
2. Same data order, token budget, learning-rate schedule and seeds for every arm. Never tune the method more than the baseline.
3. Report mean ± std over seeds and compare to the dense seed spread. One seed is an anecdote.
4. Report measured tokens/s and memory even if the low-rank arms are not faster on this GPU.
5. Failed and surprising runs go in a `notes.md` log with the date; reviewers' questions are easier to answer from a log.
6. State the scale limits in the abstract, introduction and limitations. Do not extrapolate to large models.
7. Read every paper you cite; export BibTeX from the source page. Keep `related.md` current.
8. Anonymity: no names, usernames, repo URLs or machine names in the PDF, code zip or result files (the JSONs record only the GPU model).
