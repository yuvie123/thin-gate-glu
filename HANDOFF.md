# Handoff notes for coding sessions in this repo

Read `README.md` (commands, go/no-go criteria, honesty rules) and `PLAN.md` (idea, abstract, titles; its
8-day ICLR schedule is history) before doing anything. `SETUP_PC.md` and `setup_pc.sh` cover the local
machine, which is no longer used for experiments. `related.md` is the reading list. `notes.md` is the dated
run log and the place where every decision is recorded.

## Who and what
- The author is a first-year CS undergraduate writing a first research paper, comfortable with Python and
  **new to PyTorch**. Explain ML-research steps (baselines, seeds, noise floor, out-of-memory fixes) instead
  of assuming them. Sole author; nobody available as a co-author or reciprocal reviewer.
- Project: one submission on **thin-gate GLU**: factorize only the gate projection of SwiGLU to low rank,
  keep up/down dense. **Target: CPAL 2027, Proceedings Track** (Conference on Parsimony and Learning, Tokyo,
  Mar 23-26, 2027). **Abstract registration Nov 23, 2026; paper Dec 5, 2026** (aim to upload Dec 3);
  notification Feb 1, 2027. 9 pages main text, double-blind, archival PMLR proceedings, arXiv preprint
  allowed. The ICLR 2027 abstract deadline (Sep 19) was missed; nothing was registered there.
- Fallbacks, in order: CPAL Recent Spotlight (non-archival, Jan 18, 2027), TMLR (rolling). AISTATS 2027 and
  the ARR October cycle are closed to a sole first-time author (reciprocal-reviewer / service-contributor
  rules; verified 2026-09-19, table in `notes.md`). MLSys 2027 (Oct 30) is not pursued.
- Milestones: **go/no-go Sun Sep 27, 6 pm EDT**; **experiments freeze Sun Nov 8**; abstract registration Nov 23.
- Ideas already ruled out by a novelty search (do not revisit): smaller Q/K than V dimension (published,
  arXiv 2603.04427); lossy LM-head vocabulary shortlisting; frequency-aware embedding compression.

## Machines
- **Every GPU experiment runs on Kaggle (2x Tesla T4, 15.6 GB each, no native bfloat16).** Exp. A in
  float32, Exp. B/C in float16 with loss scaling. Never pool results from another GPU or precision.
  `make_cloud_notebook.py --run <steps>` builds a self-contained notebook that carries the code and every
  finished result; the author uploads it, runs it, downloads `thin_gate_results.zip`, drops it in the repo
  root (gitignored). Unzip it with Python's `zipfile` into the repo root; files land in `results/...` and
  `logs/`. Kaggle's free quota was about 30 GPU-hours a week when last checked.
- The local Windows PC (Ryzen 7, RTX 3070, 8 GB) is **not** used for experiments by the author's decision.
  With eleven weeks it would lift the quota ceiling if the author changes their mind; nothing depends on it.
- This MacBook is for editing, the paper, plotting and `--smoke` checks. `.venv` here holds numpy, matplotlib
  and, since 2026-09-20, a CPU build of torch, so `python tests.py`, `python train.py --smoke ...` and
  `python grid.py --stage ...` all run here (use `.venv/bin/python`). The Kaggle check step still reruns
  them on the GPU, compiled, before any real run.
- Results travel as small JSON files through git (`results/posthoc`, `results/train`, `results/heal`,
  `results/bench`, `paper/figures`, `paper/tables`). `results/scratch` and `logs/` are gitignored: copy any
  number you need from them into `notes.md`. Never commit data, weights, checkpoints or the downloaded
  Kaggle notebook (`notebook*.ipynb`).
- The GitHub repo is **private and must stay private** until reviews finish (double-blind). Never change its
  visibility, never put names, usernames, repo URLs or machine names in the paper, code comments or result
  files.

## Status (update this section as work proceeds)
- 2026-09-17 (Day 0): all experiment code written; `tests.py` passes (6 tests). Cloud route built and chosen.
- **Novelty narrowed 2026-09-17:** WeLore (arXiv 2407.11239, ICML 2025) already reports that `gate_proj` is
  more low-rank than `up_proj`/`down_proj` in pretrained LLMs. Exp. A is therefore a controlled test of a
  known observation; the novel part is Exp. B (thin gate from scratch, with controls) and secondarily Exp. C.
  The intro must start from WeLore. Details in `related.md` (top entry) and `notes.md`.
- **2026-09-19: Kaggle session 1 taken in. First paper-grade result exists.** Exp. A finished on 7 models
  (SmolLM2-135M/360M/1.7B, Qwen2.5-0.5B/1.5B, TinyLlama v1.1, OLMo-2-1B; Llama-3.2-1B skipped, gated), T4,
  float32, full wikitext-2 test split; all checks ok; committed in `results/posthoc/` with figures and tables
  from `plot.py`. Whitened SVD: **gate beats up in 41 of 42 cells; gate is the most tolerant projection in
  every model at r/d = 0.75 and 0.5; below r/d = 0.25 down overtakes gate in 5 of 7 models.** Full reading in
  `notes.md`. This meets the README "go" criterion for gate vs up, and for gate vs down at the ranks Exp. B
  uses, with the crossover to be reported as a finding. Formal go/no-go waits for the pilot noise floor.
- **Throughput measured 2026-09-19:** 62,990 tokens/s for size S in float16 on one T4, peak memory 2.2 GB.
  One 300M-token size-S run is about 1.3 h, so **`TOKENS["S"] = 300e6` stays**. `TOKENS["M"]` is unmeasured:
  measure it before `main_M` starts, never during (rule 3).
- The detached CPU sweep of Exp. A (2026-09-17) was lost and is dropped; the T4 run supersedes it.
- Paper: title and the numbers-free abstract are written (`paper/main.tex`, `paper/openreview_abstract_form.md`,
  now a CPAL checklist); the AI-use, ethics and reproducibility statements are written; hardware wording now
  says T4. The paper still uses the **ICLR 2027 style file as a placeholder** because CPAL's template is not
  published; check cpal.cc/openreview/ and swap when it is (same 9-page limit). 12 `\todo`s remain.
  **`references.bib` is still empty and no paper in `related.md` has been read yet.**
- Open plotting decision: `plot.py` now writes both `posthoc_{whiten,plain}.pdf` (x = rank / full rank) and
  `posthoc_{whiten,plain}_params.pdf` (x = parameters of the factorized projection / dense, which differs
  between model families). Both are included in `main.tex` with a `\todo` to keep one in the main text.
- 2026-09-19: Setup (Exp. A part) and the Exp. A results subsection are drafted in `main.tex`, wired to the
  generated figure and `tables/posthoc_whiten.tex`; no number typed by hand. Two `\todo`s mark the cells
  to re-verify (the single gate-vs-up tie, the single gate-vs-down near-tie at r/d = 1/2). The paper
  compiles with `tectonic` on the Mac (empty bibliography, expected).
- **2026-09-20: Kaggle session 2 taken in. Exp. B pilot, 6 runs at size S, 300M tokens, float16, committed
  in `results/train/`.** Noise floor |dense s0 − dense s1| = 0.0033. At equal parameters the order is
  **dense 4.085 < shrunk 4.104 < thin up 4.117 < thin gate 4.125 < thin down 4.147**, stable from 27% of
  training onward. **Neither pilot "go" condition in `README.md` is met**: thin gate is worse than shrunk and
  than thin up, and far outside the dense noise floor. Post-hoc tolerance (Exp. A: up fragile, gate robust)
  does not predict the from-scratch outcome. One seed per non-dense arm. Full reading and the author's
  options (finish `main_S` key arms for seeds; add rank d/2 arms; reframe as a contrast result; no-go) are
  in `notes.md`. `plot.py` now also writes `figures/training.pdf` and `tables/training.tex`; the Exp. B
  results subsection stays a `\todo`.
- **2026-09-20, night: screening stage built (decision, author: screen new variants before finishing
  `main_S`; structural changes to the gate allowed).** Goal: a variant at the shrunk_r4 budget (921,600 MLP
  params/layer) whose seed-0 loss is at or below 4.100 (shrunk s0 = 4.1043). `grid.py --stage screen`, 13
  runs at size S seed 0 in priority order: Monarch gate (block-structured, full rank, same params as rank
  d/4), grouped gate (one gate value per 4 units, d_ff 1064), thin gate with spectral init and/or no factor
  decay, warm-started thin gate (dense gate for 10% / 25% of steps, then whitened SVD to rank d/4; reported
  separately, it costs about 1% more train compute), nonlinear bottleneck gate, plus grouped up/down
  controls and two fillers. New flags in `train.py` (`--gate_groups/--up_groups/--down_groups`,
  `--gate_monarch`, `--lowrank_init`, `--bottleneck`, `--factor_wd`, `--thin_at`); `tests.py` has 11 tests;
  every old arm is bit-for-bit unchanged (smoke curve identical to the previous commit). Decision rule and
  design in `notes.md` 2026-09-20 (screen entry). Session 4 = `--run screen,main_S`.
- **2026-09-20, evening: Kaggle session 3 taken in (main_S, 15 of 27 runs; the launch deadline worked).**
  Key arms at three seeds: dense 4.094 (range 0.029, one unlucky seed), shrunk r4 4.107, thin up r4 4.116,
  thin gate r4 4.118, thin down r4 4.144. Thin gate never beats shrunk (at d/2, d/4, d/8, seed 0), is
  indistinguishable from thin up, and only thin down and all-lowrank are clearly worse than dense.
  **Verdict firmer: the gate hypothesis as titled fails at size S.** The combined finding is that post-hoc
  rank tolerance (Exp. A) does not predict from-scratch trainability (Exp. B). Table, reading and the
  updated options are in `notes.md`. 12 `main_S` runs remain (seeds 1-2 of the six rest arms).

- **2026-09-21: Kaggle session 4 taken in (screen, 13 runs at seed 0; none of the 12 leftover `main_S` runs
  fit).** Decision rule applied: **all 13 variants dropped.** Best is thin gate r4 with spectral init and no
  factor decay at 4.1127 vs shrunk r4 s0 4.1043 (bar: 4.100); the 13-arm spread (0.019) is below the dense seed
  range (0.029). Optimizer fixes land inside the thin-gate seed range; grouped keeps gate < up < down, all above
  shrunk; Monarch is last; warm start ends worse than from-scratch thin gate. The "which structure on the gate"
  paper is off the table; only the contrast paper has support. Table and reading in `notes.md` 2026-09-21.
  `plot.py` now writes `tables/warm_start.tex` and keeps warm-started arms out of the matched table.
- **Repo state, 2026-09-21:** the remote's evening-of-Sep-20 README rewrite had deleted `notes.md`, `HANDOFF.md`
  and `PLAN.md`; they are restored (README kept). Work happens in `~/thin-gate-glu`; the other checkout,
  `~/Desktop/thin-gate-glu`, holds the `.venv` (numpy, matplotlib, CPU torch) but is behind: pull it or delete
  it. Until a `.venv` exists here, run `plot.py`, `tests.py` and `make_cloud_notebook.py` with
  `/Users/aman/Desktop/thin-gate-glu/.venv/bin/python`.

- **2026-09-22: Kaggle session 5 taken in. `main_S` is complete: 11 arms x 3 seeds (33 runs), all ok.**
  Thin gate loses to shrunk at every rank at every seed (nine of nine pairs; mean gaps 0.012 at d/2, 0.011 at
  d/4, 0.027 at d/8). Shrunk r2 matches dense on the mean with 10% fewer MLP parameters. Reinvest is within
  noise of dense. All-lowrank is worst. The titled hypothesis fails at size S at every rank; the contrast
  finding stands. Table in `notes.md` 2026-09-22. No notebook built: the next session's content depends on
  the Sep 27 decision.

- **2026-09-22, evening: screen 2 built (decision, author: one more attempt to beat shrunk, with the odds
  stated first).** Six arms at size S, seeds 0-2: tied gate (gate = up-projection, zero gate parameters,
  d_ff 1200), tied gate with relu (Primer's squared ReLU), thin gate + tied term (rank 96 plus a per-unit
  scale), a gate shared across all layers (d_ff 1104), and a starved-budget pair (shrunk 400 vs tied 600).
  New: `train.py --ref_json/--kill_steps` stops a run at step 2000 / 3000 if it is more than 0.015 / 0.010
  behind the same-seed shrunk curve (backtested on all 41 finished runs: zero false kills). `tests.py` has
  15 tests; the baseline smoke curve is unchanged. Priors, backtest and the promotion rule are in
  `notes.md` 2026-09-22 evening. **Kaggle quota is exhausted;** the notebook also runs on Colab.

- **2026-09-22, later: Exp. B written into `paper/main.tex`** (setup paragraph and the results subsection
  with five findings: noise floor, thin gate never beats shrunk, gate = up while down is worst, the dense
  block is wider than needed at S, screened alternatives; plus the Exp. A / B contrast). `plot.py` now also
  writes `tables/paired.tex` (seed-paired gaps), `tables/macros.tex` (every number the text uses, as LaTeX
  macros), and splits the training table/figure into `training` (planned grid), `training_screen`
  (variants at seed 0, shrunk r4 as reference) and, once run, `training_starved`. Compiles with tectonic;
  8 pages with an empty bibliography. Remaining `\todo`s need reading (citations), size M, Exp. C, bench.

### Next, in order
1. ~~Session 2: pilot~~, ~~Session 3: main_S key arms~~, ~~Session 4: screen~~ (all dropped),
   ~~Session 5: main_S rest~~ done 2026-09-22. `main_S` is complete.
2. **Session 6 (built 2026-09-22, upload next): `cloud_notebook.ipynb` from `make_cloud_notebook.py --run
   screen2`.** 18 runs with the early-kill rule; round 1 (seed 0 of all six) is about 4.2 h on two T4s if
   nothing is killed, the rest follows until the launch deadline. Take in as usual, then apply the promotion
   rule in `notes.md` 2026-09-22 evening (win: paired 3-seed mean at or below -0.005 with all pairs negative).
   Compute: Kaggle quota is used up this week; Colab runs the same notebook; the local RTX 3070 works if the
   shrunk references are rerun there first.
3. **Sun Sep 27, 6 pm: go / pivot / no-go.** The screen closed the "which structure on the gate" route. The
   data argue for the **contrast paper** (post-hoc tolerance does not predict trainability; Exp. A, B, C kept;
   retitle). If chosen, add `thin_up_r2/r8` and `thin_down_r2/r8` to `grid.py` (rerun `tests.py`) so Exp. B's
   headline figure is a rank sweep of all three projections at three seeds (12 runs, one session). Record the
   decision in `notes.md`.
4. Meanwhile the author reads WeLore (Sec. 2.1, 2.4, 3.1, Figs 1, 3, 7) first, then the other must-reads in
   `related.md`; BibTeX exported from each paper's page only after reading it.
5. **Sep 28 - Oct 11:** the rank-sweep session if the contrast paper is chosen (`make_cloud_notebook.py --run main_S` after the `grid.py` change; the runner skips the 33 finished runs). Related Work and Method drafts.
6. **Oct 12 - Oct 25:** measure size-M throughput, set `TOKENS["M"]`, rerun `tests.py`, then `--run main_M`
   (14 runs), then `--run heal,bench`. Setup and Exp. A results sections.
7. **Oct 26 - Nov 8:** `--stage lr` if quota allows; optional lm-eval zero-shot. Results for B and C;
   Limitations. **Freeze Nov 8.** If quota runs short, drop size M before dropping seeds at size S.
8. **Nov 9 - 22:** introduction (last), abstract with real numbers, appendix table (`grid.py --table`), swap
   in the CPAL template. **Nov 23: abstract registration** on OpenReview (`paper/openreview_abstract_form.md`).
9. **Nov 24 - Dec 3:** full read-through, anonymity check, PDF metadata, anonymized code zip. Upload by Dec 3
   (hard deadline Dec 5).

### Go or no-go (decide by 6 pm Sunday Sep 27)

This text lived in `README.md` until the 2026-09-20 rewrite removed it; it is kept here so the criteria on
record stay the ones written before the results came in.

Look at `paper/figures/posthoc_*.pdf` and at the training runs. The noise floor is the range of the dense
seeds in final validation loss (0.029 with three seeds, `notes.md` 2026-09-20 15:15; the original two-seed
definition |S_dense_s0 - S_dense_s1| gave 0.003). A gap smaller than that is not a result.

- Go if, in most models, the gate curve sits clearly below the up and down curves at equal rank. The whitened SVD figure is the one that counts. Also go if thin_gate_r4 lands within the noise floor of dense while thin_up_r4 and thin_down_r4 are clearly worse, or if thin_gate_r4 beats shrunk_r4.
- Pivot if a different projection turns out to be the cheap one. The paper stays the same and the method gets renamed. The title and abstract stay true in that case.
- No-go if all three projections behave alike and low rank simply loses. Then withdraw or send the negative result to a workshop. Don't stretch the claims to keep the submission alive.

After the go, the remaining sessions are `main_S`, `main_M`, then `heal,bench`, and `lr` if quota allows. Zero-shot accuracy for a converted or truncated model is optional: see `lm_eval --help` for tasks such as `arc_easy,hellaswag,piqa`, and only add it once experiments A to C are finished. Experiments freeze on Nov 8. Then run `python plot.py` and write.

## Rules for this project (non-negotiable)
1. **Never fabricate or guess** results, citations, or benchmark numbers. Every number in the paper comes from a JSON in `results/` via `plot.py`. `paper/references.bib` entries are exported by the author from the paper's own page after reading it, never written from memory.
2. Report failures and unflattering results plainly (e.g. low-rank arms not being faster, down beating gate at low rank). A negative result is acceptable; an unsupported claim is an ethics violation.
3. Fair comparisons: identical data order, token budget, schedule, learning rate and seeds across arms. Never tune the method more than the baseline. If `TOKENS` in `grid.py` must change for time, change it before a grid starts and rerun everything at that size.
4. Compare every gap to the seed-to-seed spread of the dense baseline before calling it an effect.
5. Run `python tests.py` after any change to `model.py`, `train.py`, `grid.py`, `posthoc_truncate.py` or `data.py` (on the Mac with the Desktop `.venv`, and again on Kaggle via the notebook's check step).
6. One Kaggle session at a time; rebuild the notebook after any code change, because the copy inside an old notebook does not update.
7. Keep a dated `notes.md` log of every run launched, failures, and decisions.
8. AI use is disclosed in the paper (`paper/main.tex`, AI use statement). Keep that statement true as the work evolves. The author must understand and be able to defend everything; explain what you do and why.
9. Commit with clear messages; push small result files so the Mac can plot and write.
