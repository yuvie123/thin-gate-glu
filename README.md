# Thin-gate GLU: do GLU gates need full rank?

Code and paper for a submission to CPAL 2027 (Conference on Parsimony and Learning, Proceedings Track). Abstract registration is due Nov 23, 2026, the paper Dec 5, 2026. The project started out aimed at ICLR 2027 and moved when that abstract deadline went by unregistered.

[PLAN.md](PLAN.md) is the original plan: the ideas considered, the abstract, title options and the go/no-go criteria. Its 8-day schedule is out of date; the current one is in [HANDOFF.md](HANDOFF.md). [related.md](related.md) is the reading list. [SETUP_PC.md](SETUP_PC.md) describes the local GPU machine, which is no longer used.

## Hypothesis

A SwiGLU block computes `down( silu(gate x) * (up x) )`. The gate decides which hidden units are active; the up and down projections carry the content that gets written back. If that split is real, the gate should survive a low-rank factorization better than up or down do. We don't know yet whether it does. The experiments are built so that either answer makes a paper.

## Machines

The MacBook is for editing code, writing the paper, plotting, and `--smoke` checks. Every GPU experiment runs on Kaggle's free T4s (see "Running on Kaggle" below). The local PC sits this one out.

## Files

| File | Purpose |
|---|---|
| `posthoc_truncate.py`, `run_posthoc.sh` | Exp. A: truncate gate, up or down in pretrained models and compare perplexity, with no training |
| `model.py`, `train.py`, `data.py`, `grid.py` | Exp. B: pretrain small GPTs from scratch, all arms and seeds |
| `heal.py`, `run_heal.sh` | Exp. C: truncate a pretrained model, then train only the new factors |
| `bench.py` | measured tokens/s and memory for each arm |
| `plot.py` | turns results/*.json into `paper/figures/*.pdf` and `paper/tables/*.tex` |
| `tests.py` | correctness tests, to run after every code change |
| `cloud_run.py`, `make_cloud_notebook.py` | run the experiments on a free cloud GPU from one self-contained notebook |
| `paper/` | the paper (`main.tex`); the ICLR 2027 style file is a placeholder until CPAL publishes its template (same 9-page main text) |

Every script accepts `--help`. Each also accepts `--smoke`, which runs a toy check in under a minute and downloads nothing.

## Running on Kaggle (the route in use since 2026-09-17)

`python make_cloud_notebook.py` writes `cloud_notebook.ipynb`. The notebook carries its own copy of the code, so the private repo never has to be shared. Upload it to Kaggle (Accelerator: GPU T4 x2, Internet: on, notebook private), use Save Version, then Save & Run All. When it finishes, download `thin_gate_results.zip` from the Output tab and unzip it into this folder.

Kaggle sessions start empty and stop after 12 hours, so the work is split into sessions. Each new notebook carries the finished results of the earlier ones inside it, and the runner skips whatever is already done:

```bash
python make_cloud_notebook.py                          # session 1: speed check + Exp. A on all models
python make_cloud_notebook.py --run pilot              # session 2: Exp. B pilot, after TOKENS["S"] is fixed
python make_cloud_notebook.py --run main_S             # sessions 3-4: all arms, three seeds (finished runs are skipped)
python make_cloud_notebook.py --run heal,bench         # Exp. C and the speed/memory table
```

After every session: unzip, `git add results && git commit`, rebuild the notebook, upload, run. Rebuild after any code change too, since the copy inside an old notebook does not update.

A T4 has no native bfloat16, so Exp. A runs in float32 on every model and Exp. B and C use float16 with loss scaling. Each result file records its precision and GPU. All of Exp. B has to come from this one recipe; runs from different precisions or GPUs never share a table.

Check the baseline before anything else. The "baseline perplexity" printed for SmolLM2-135M on wikitext-2 should be in the tens, which is normal for a 135M model. Hundreds or thousands means the evaluation code is broken, and every later number depends on it.

Then work out the time budget. The throughput check prints tokens per second, and one size-S run takes `300e6 / tok_per_s` seconds. If that is more than about 1.5 hours, lower `TOKENS["S"]` in `grid.py` before starting the grid. Once a grid has started, the token budget stays the same for every arm.

### Go or no-go (decide by 6 pm Sunday Sep 27)

Look at `paper/figures/posthoc_*.pdf` and at the pilot runs. The noise floor is |S_dense_s0 − S_dense_s1| in final validation loss. A gap smaller than that is not a result.

- Go if, in most models, the gate curve sits clearly below the up and down curves at equal rank. The whitened SVD figure is the one that counts. Also go if thin_gate_r4 lands within the noise floor of dense while thin_up_r4 and thin_down_r4 are clearly worse, or if thin_gate_r4 beats shrunk_r4.
- Pivot if a different projection turns out to be the cheap one. The paper stays the same and the method gets renamed. The title and abstract stay true in that case.
- No-go if all three projections behave alike and low rank simply loses. Then withdraw or send the negative result to a workshop. Don't stretch the claims to keep the submission alive.

After the go, the remaining sessions are `main_S`, `main_M`, then `heal,bench`, and `lr` if quota allows. Zero-shot accuracy for a converted or truncated model is optional: see `lm_eval --help` for tasks such as `arc_easy,hellaswag,piqa`, and only add it once experiments A to C are finished. Experiments freeze on Nov 8. Then run `python plot.py` and write.

## Commands on a local GPU (reference only)

These run the same experiments directly on a Linux machine with a CUDA GPU. They are kept in case the Kaggle route stops working.

```bash
python tests.py
python posthoc_truncate.py --model HuggingFaceTB/SmolLM2-135M            # minutes; first real result
python data.py --num_train_shards 6                                       # ~1.4 GB of tokens
python train.py --name throughput_check --size S --tokens 20e6 --out_dir results/scratch   # note the tok/s it prints
python grid.py --stage pilot --run                                        # overnight, inside tmux
```

```bash
bash run_posthoc.sh            # plain SVD, all models
bash run_posthoc.sh --whiten   # activation-aware SVD, all models
python plot.py
```

```bash
python grid.py --stage main_S --run      # key arms first, 3 seeds
python grid.py --stage main_M --run
bash run_heal.sh HuggingFaceTB/SmolLM2-135M
bash run_heal.sh HuggingFaceTB/SmolLM2-360M --micro_bs 1
python bench.py --size S && python bench.py --size M
python grid.py --stage lr --run          # if time: shows the comparison doesn't hinge on one learning rate
```

## Rules that keep the paper honest

1. Every number in the paper comes from a JSON file in `results/` by way of `plot.py`. No number is typed in by hand.
2. Every arm gets the same data order, token budget, learning-rate schedule and seeds. The method never gets more tuning than the baseline.
3. Report mean ± std over seeds and compare each gap to the spread between dense seeds. A difference seen in one seed could be noise.
4. Report measured tokens/s and memory, even if the low-rank arms turn out no faster on this GPU.
5. Log failed and surprising runs in `notes.md` with the date. It makes reviewer questions much easier to answer.
6. State the scale limits in the abstract, the introduction and the limitations section. Don't extrapolate to large models.
7. Read every paper you cite, and export its BibTeX from the source page. Keep `related.md` current.
8. Keep names, usernames, repo URLs and machine names out of the PDF, the code zip and the result files. The JSONs record only the GPU model.
