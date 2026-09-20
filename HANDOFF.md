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

### Next, in order
1. ~~Session 2: pilot~~ done 2026-09-20. ~~Session 3: main_S key arms~~ done 2026-09-20 (15 runs).
2. **Session 4 (built, upload next):** `python make_cloud_notebook.py --run screen,main_S`: the 13 screen
   runs first, then whatever of the 12 remaining `main_S` runs fit before the launch deadline. Take in as
   usual; then apply the decision rule in `notes.md` (promote a variant to seeds 1-2 plus its up/down
   controls if its final loss is at or below 4.100 and below shrunk at the last three evals; one more seed
   if between 4.100 and 4.1043; drop otherwise). Warm start is judged separately as the Exp. A/B bridge.
3. **Sun Sep 27, 6 pm: go / pivot / no-go.** If a screened variant is promoted and holds up with seeds, the
   paper is about that variant ("which structure on the gate", or "selection is cheap" if the grouped gate
   wins). If none does, the data argue for the **contrast paper** (post-hoc tolerance does not predict
   trainability; Exp. A, B, C kept; retitle), which then wants `thin_up_r2/r8` and `thin_down_r2/r8` added
   to `grid.py` so Exp. B's headline figure is a rank sweep of all three projections at three seeds (12
   runs, one session). Record the decision in `notes.md`.
4. Meanwhile the author reads WeLore (Sec. 2.1, 2.4, 3.1, Figs 1, 3, 7) first, then the other must-reads in
   `related.md`; BibTeX exported from each paper's page only after reading it.
4. **Sep 28 - Oct 11:** `--run main_S` over two sessions (33 runs x 1.3 h / 2 GPUs, about 22 h; key arms
   first, so a cut-off session still yields the 5 key arms x 3 seeds). Related Work and Method drafts.
5. **Oct 12 - Oct 25:** measure size-M throughput, set `TOKENS["M"]`, rerun `tests.py`, then `--run main_M`
   (14 runs), then `--run heal,bench`. Setup and Exp. A results sections.
6. **Oct 26 - Nov 8:** `--stage lr` if quota allows; optional lm-eval zero-shot. Results for B and C;
   Limitations. **Freeze Nov 8.** If quota runs short, drop size M before dropping seeds at size S.
7. **Nov 9 - 22:** introduction (last), abstract with real numbers, appendix table (`grid.py --table`), swap
   in the CPAL template. **Nov 23: abstract registration** on OpenReview (`paper/openreview_abstract_form.md`).
8. **Nov 24 - Dec 3:** full read-through, anonymity check, PDF metadata, anonymized code zip. Upload by Dec 3
   (hard deadline Dec 5).

## Rules for this project (non-negotiable)
1. **Never fabricate or guess** results, citations, or benchmark numbers. Every number in the paper comes from a JSON in `results/` via `plot.py`. `paper/references.bib` entries are exported by the author from the paper's own page after reading it, never written from memory.
2. Report failures and unflattering results plainly (e.g. low-rank arms not being faster, down beating gate at low rank). A negative result is acceptable; an unsupported claim is an ethics violation.
3. Fair comparisons: identical data order, token budget, schedule, learning rate and seeds across arms. Never tune the method more than the baseline. If `TOKENS` in `grid.py` must change for time, change it before a grid starts and rerun everything at that size.
4. Compare every gap to the seed-to-seed spread of the dense baseline before calling it an effect.
5. Run `python tests.py` after any change to `model.py`, `grid.py`, `posthoc_truncate.py` or `data.py` (on Kaggle, via the notebook's check step, since the Mac has no torch).
6. One Kaggle session at a time; rebuild the notebook after any code change, because the copy inside an old notebook does not update.
7. Keep a dated `notes.md` log of every run launched, failures, and decisions.
8. AI use is disclosed in the paper (`paper/main.tex`, AI use statement). Keep that statement true as the work evolves. The author must understand and be able to defend everything; explain what you do and why.
9. Commit with clear messages; push small result files so the Mac can plot and write.
