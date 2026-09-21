# Thin-gate GLU: do GLU gates need full rank?

Code and paper for a submission to CPAL 2027 (Conference on Parsimony and Learning, Proceedings Track).

A SwiGLU block computes `down( silu(gate x) * (up x) )`. The gate decides which hidden units are active; the up and down projections carry the content that gets written back. The three matrices have the same shape, so the question "is selection cheaper than content?" has a controlled test: restrict one projection at a time to the same rank, so the parameter savings are identical, and measure which restriction the model tolerates best.

## Results so far (training-free truncation)

Each projection type was truncated to rank `r` in every layer of seven open pretrained models (SmolLM2 135M/360M/1.7B, Qwen2.5 0.5B/1.5B, TinyLlama 1.1B, OLMo-2 1B), at six rank fractions from `r/d = 3/4` down to `1/16`, using both plain SVD and activation-aware (whitened) SVD, and scored by WikiText-2 perplexity. Full tables are in `paper/tables/`, generated from `results/posthoc/*.json` by `plot.py`.

- **The gate is far more compressible than the up-projection.** Under whitened SVD, truncating the gate costs less perplexity than truncating the up-projection to the same rank in 41 of 42 model-rank settings. The one exception (SmolLM2-360M at `r/d = 1/4`) is within a fraction of a percent. Plain SVD gives the same ordering in 37 of 42.
- **The advantage depends on rank.** At `r/d = 3/4` the gate is the best projection to shrink in all 7 models, and at `r/d = 1/2` in 6 of 7 (in Qwen2.5-0.5B the down-projection is ahead by about 3%). Below `r/d = 1/4` the down-projection overtakes the gate in 5 of 7 models.
- **Plain SVD is not usable below `r/d = 3/4`** for any projection; the whitened variant is the one the rest of the paper builds on.

So the hypothesis holds in part: the gate is cheaper than the up-projection throughout, and cheaper than the down-projection at moderate rank. Pretraining from scratch (Exp. B) and post-truncation healing (Exp. C) are in progress.

## Setup

Code is edited and plotted on a laptop; every GPU experiment runs on Kaggle's free T4s (see "Running on Kaggle" below). Exp. A runs in float32; Exp. B and C use float16 with loss scaling. Each result file records its precision and GPU.

## Files

| File | Purpose |
|---|---|
| `posthoc_truncate.py`, `run_posthoc.sh` | Exp. A: truncate gate, up or down in pretrained models and compare perplexity, with no training |
| `model.py`, `train.py`, `data.py`, `grid.py` | Exp. B: pretrain small GPTs from scratch, all arms and seeds |
| `heal.py`, `run_heal.sh` | Exp. C: truncate a pretrained model, then train only the new factors |
| `bench.py` | measured tokens/s and memory for each arm |
| `plot.py` | turns results/*.json into `paper/figures/*.pdf` and `paper/tables/*.tex` |
| `tests.py` | 11 correctness tests (exact full-rank reconstruction, parameter/FLOP counts, causal masking) |
| `cloud_run.py`, `make_cloud_notebook.py` | run the experiments on a free cloud GPU from one self-contained notebook |
| `paper/` | the paper (`main.tex`), figures and tables; the ICLR style file is a placeholder until CPAL publishes its template |

Every script accepts `--help`. Each also accepts `--smoke`, which runs a toy check in under a minute and downloads nothing.

## Running on Kaggle

`python make_cloud_notebook.py` writes `cloud_notebook.ipynb`. The notebook carries its own copy of the code. Upload it to Kaggle (Accelerator: GPU T4 x2, Internet: on, notebook private), use Save Version, then Save & Run All. When it finishes, download `thin_gate_results.zip` from the Output tab and unzip it into this folder.

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
5. Log failed and surprising runs with the date, so reviewer questions are easy to answer.
6. State the scale limits in the abstract, the introduction and the limitations section. Don't extrapolate to large models.
7. Read every paper you cite, and export its BibTeX from the source page. Keep `related.md` current.
8. Keep names, usernames, repo URLs and machine names out of the PDF, the code zip and the result files. The JSONs record only the GPU model.
