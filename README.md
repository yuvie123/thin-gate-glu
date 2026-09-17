# Thin-gate GLU: do GLU gates need full rank?

This repo holds the code and paper for an ICLR 2027 submission. The abstract is due Sep 18 AOE (Sat Sep 19, 07:59 EDT) and the paper is due Sep 25 AOE (Sat Sep 26, 07:59 EDT).

[PLAN.md](PLAN.md) has the full plan: the ideas considered, the abstract, title options, the schedule and the go/no-go checkpoint. [SETUP_PC.md](SETUP_PC.md) covers setting up the GPU machine, and [related.md](related.md) is the reading list.

## Hypothesis

A SwiGLU block computes `down( silu(gate x) * (up x) )`. The gate decides which hidden units are active, and the up and down projections carry the content that gets written back. If that split is real, the gate should survive a low-rank factorization better than up or down do. We don't know yet whether it does, and the experiments are set up so that either answer is usable.

## Machines

The MacBook is for editing code, writing the paper, plotting, and `--smoke` checks. Everything else runs on the PC with the RTX 3070.

## Files

| File | Purpose |
|---|---|
| `posthoc_truncate.py`, `run_posthoc.sh` | Exp. A: truncate gate, up or down in pretrained models and compare perplexity, with no training |
| `model.py`, `train.py`, `data.py`, `grid.py` | Exp. B: pretrain small GPTs from scratch, all arms and seeds |
| `heal.py`, `run_heal.sh` | Exp. C: truncate a pretrained model, then train only the new factors |
| `bench.py` | measured tokens/s and memory for each arm |
| `plot.py` | turns results/*.json into `paper/figures/*.pdf` and `paper/tables/*.tex` |
| `tests.py` | correctness tests, to run after every code change |
| `paper/` | ICLR 2027 template (`main.tex`) with the official style files left untouched |

Every script accepts `--help`. Each also accepts `--smoke`, which runs a toy check in under a minute and downloads nothing.

## Commands, in order (GPU machine)

### Day 1

```bash
python tests.py
python posthoc_truncate.py --model HuggingFaceTB/SmolLM2-135M            # minutes; first real result
python data.py --num_train_shards 6                                       # ~1.4 GB of tokens
python train.py --name throughput_check --size S --tokens 20e6 --out_dir results/scratch   # note the tok/s it prints
python grid.py --stage pilot --run                                        # overnight, inside tmux
```

Check the baseline first. The "baseline perplexity" printed for SmolLM2-135M on wikitext-2 should be in the tens, which is normal for a 135M model. If it comes out in the hundreds or thousands, stop, because the evaluation code is broken and every later number depends on it.

Then work out the time budget. The throughput check prints tokens per second, and one size-S run takes `300e6 / tok_per_s` seconds. If that is more than about 1.5 hours, lower `TOKENS["S"]` in `grid.py` before starting the grid. Once a grid has started, the token budget has to stay the same for every arm.

### Day 2

```bash
bash run_posthoc.sh            # plain SVD, all models
bash run_posthoc.sh --whiten   # activation-aware SVD, all models
python plot.py
```

### Day 3: go or no-go (decide by 6 pm Sunday Sep 20)

Look at `paper/figures/posthoc_*.pdf` and at the pilot runs. The noise floor is |S_dense_s0 − S_dense_s1| in final validation loss, and a gap smaller than that doesn't count as a result.

- Go if, in most models, the gate curve sits clearly below the up and down curves at equal rank. The whitened SVD figure is the one that counts. Also go if thin_gate_r4 lands within the noise floor of dense while thin_up_r4 and thin_down_r4 are clearly worse, or if thin_gate_r4 beats shrunk_r4.
- Pivot if a different projection turns out to be the cheap one. The paper stays the same and the method gets renamed. The submitted title and abstract are still true in that case.
- No-go if all three projections behave alike and low rank simply loses. Then withdraw or send the negative result to a workshop. Don't stretch the claims to keep the submission alive.

### Day 4-5

```bash
python grid.py --stage main_S --run      # key arms first, 3 seeds
python grid.py --stage main_M --run
bash run_heal.sh HuggingFaceTB/SmolLM2-135M
bash run_heal.sh HuggingFaceTB/SmolLM2-360M --micro_bs 1
python bench.py --size S && python bench.py --size M
python grid.py --stage lr --run          # if time: shows the comparison doesn't hinge on one learning rate
```

Zero-shot accuracy for a converted or truncated model is optional. See `lm_eval --help` for tasks such as `arc_easy,hellaswag,piqa`, and only add it once experiments A to C are finished.

### Day 6 onward

Run `python plot.py`, then write. Experiments freeze at noon on Wednesday.

## Rules that keep the paper honest

1. Every number in the paper comes from a JSON file in `results/` by way of `plot.py`. Nobody types a number into the paper by hand.
2. Every arm gets the same data order, token budget, learning-rate schedule and seeds. The method never gets more tuning than the baseline.
3. Report mean ± std over seeds and compare each gap to the spread between dense seeds. A difference seen in one seed could be noise.
4. Report measured tokens/s and memory, even if the low-rank arms turn out no faster on this GPU.
5. Log failed and surprising runs in `notes.md` with the date. Reviewer questions are much easier to answer from a log.
6. State the scale limits in the abstract, the introduction and the limitations section. Don't extrapolate to large models.
7. Read every paper you cite, and export its BibTeX from the source page. Keep `related.md` current.
8. Keep names, usernames, repo URLs and machine names out of the PDF, the code zip and the result files. The JSONs record only the GPU model.
