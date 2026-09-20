# Run log

Every run launched, every failure and every decision, with dates. Times are EDT.

## 2026-09-17 (Day 0)

**Environment audit on the GPU PC. Setup is blocked; no experiment has run.**

- GPU is visible from Windows: RTX 3070, 8192 MiB, driver 560.94. `nvidia-smi` works.
- **WSL2 is not installed.** `wsl --version`, `wsl --status` and `wsl -l -v` all print only the
  usage stub and exit 1, which is what `wsl.exe` does when the Windows feature is disabled and
  no distribution exists. So there is no Linux environment, no `.venv` and no `tmux` yet.
- The hardware supports it: virtualization is enabled in firmware and SLAT is present.
  184 GB free on C:. The hypervisor is not currently running, which is expected while the
  feature is off.
- Python on the Windows side is 3.8 only, plus the Store alias stubs. That is too old for a
  current PyTorch, so even a native-Windows fallback needs a newer Python installed first.
- `setup_pc.sh` still has never been executed. It is Linux-only (`apt-get`) and deliberately
  refuses to run from `/mnt/c`, so it cannot run until a distribution exists and the repo is
  cloned under `~/`.
- Consequence: `tests.py` on this machine, Exp. A, the throughput check and the pilot grid are
  all blocked. `results/` is still empty and no number of any kind exists yet.
- Decision pending, and it needs an elevated shell and a reboot, so it is the author's call:
  install WSL2 + Ubuntu as SETUP_PC.md describes, or fall back to native Windows with a fresh
  Python. `train.py` and `bench.py` both accept `--no_compile`, so the fallback needs no code
  change, but it gives up `tmux` and makes the measured tokens/s non-comparable to a Linux run.

### 2026-09-17, evening: CPU fallback so Exp. A can start before WSL2 exists

Decision: rather than idle until WSL2 is installed, run Exp. A on the CPU. Exp. A is
training-free (SVD plus a perplexity evaluation), so it does not need the GPU. Exp. B, Exp. C
and `bench.py` still do and remain blocked.

Environment built on the Windows side (interim, not the machine of record):
Python 3.12.10 in `.venv`, torch 2.14.0+cpu, transformers 5.17.0, datasets 5.0.1, numpy 2.5.3,
8 threads. `lm-eval` deliberately not installed; it is only needed for the optional Day 4-5
zero-shot evaluations and is the dependency most likely to fight `transformers`.

- `python tests.py`: **all 6 pass** on this machine.
- **Bug found and fixed** (`hf_utils.py`): `load_dataset("wikitext", ...)` fails under
  datasets>=4, which rejects a bare legacy dataset name and demands `namespace/name`. Changed to
  `Salesforce/wikitext`, verified on the Hub as carrying the `wikitext-2-raw-v1` config. This was
  blocking Exp. A completely and would have blocked it on the GPU too, so it is not a CPU artifact.
- **Change** (`posthoc_truncate.py`): the dtype is now chosen from the device, bfloat16 on CUDA and
  float32 on CPU, because Zen 2 has no native bfloat16 and emulates it far too slowly to evaluate.
  The dtype is now also recorded in the results JSON. CUDA behaviour is unchanged. `tests.py` rerun
  and still 6/6. Consequence to remember: CPU fp32 perplexities are not bit-identical to GPU bf16
  ones, so the two must not be mixed inside one figure.
- **Baseline sanity check: perplexity 20.074** for SmolLM2-135M on wikitext-2 (on 8 windows of
  1024 tokens, not the full split). Plausible tens for a 135M model, so the evaluation is sound
  and downstream numbers can be trusted. README's stop condition is not triggered.
- Plain SVD on `gate_proj`, all 30 layers, 8 windows, degradation is smoothly monotonic:
  rank 570 (0.99) x1.01, 518 (0.90) x1.11, 432 (0.75) x2.37, 288 (0.50) x81.6, 144 (0.25) x3531.
  Monotonic, so this is the method's real behaviour and not a bug. Plain SVD is simply destructive
  at low rank, which is consistent with why the activation-aware variant exists at all.
- Whitened SVD on the same ranks is far better: 432 x1.10 and 288 x1.63, against x2.37 and x81.6
  plain. Confirms whitening is the stronger baseline and the one the go/no-go should rest on.
- Measured cost: about 1.9 s per 1024-token window on 8 CPU threads, and the full wikitext-2 test
  split is 297 windows, so roughly 9.5 min per evaluation and about 3 h per 19-config sweep.
- **Launched**: full Exp. A on SmolLM2-135M, both variants in sequence, full test split, all three
  projection types, default rank fractions. Logs in `results/logs/`. Expected around 6 h total.
  Only the 135M model is CPU-feasible; 360M would be roughly 8 h per variant and 1.7B far beyond
  reach, so the larger models in the abstract still need the GPU.

### 2026-09-17, evening: ICLR submission requirements checked against the real form

Read the actual OpenReview submission form (screenshots, kept out of git: they carry the author name).
This corrected an earlier assumption taken from the Author Guidelines page, which says only the Abstract
field is mandatory. On the real form the PDF is indeed optional, but many other fields are required.

- Required at the abstract deadline (Sep 19, 07:59): Title, Authors, **Keywords**, Abstract,
  **Primary Area**, Code of Ethics, Paper Visibility, Submission Requirements, **Reciprocal Reviewing
  Author**, **Reciprocal Reviewing Exemption**, **AI Assistance**, License.
  Optional: TL;DR, PDF (max 50 MB), Supplementary Material (max 100 MB, zipped, must be anonymized),
  Recent Qualifying Paper, Ready For LLM Feedback.
- **The reciprocal-reviewing fields cannot be changed after the abstract deadline.** The form resolves
  the ambiguity found earlier: "If the submission has no eligible reciprocal reviewer, simply enter any
  author in this field", and there is a Reciprocal Reviewing Exemption dropdown whose stated example
  reason is exactly "no authors are qualified to be reviewers because they are first-time authors".
  So the first-time-author path is an explicit exemption request, not an automatic exemption. Getting
  this wrong is a desk rejection at the Program Chairs' discretion, and it is unfixable after Saturday.
- Paper Visibility acknowledgement is stricter than expected: submitted papers become public at the
  START of review, accepted and rejected papers are de-anonymized at the end, withdrawn papers are
  de-anonymized immediately on withdrawal, and papers cannot be deleted, hidden or retracted once
  reviewing begins. Decide before submitting, not after.
- AI Assistance is a required checkbox list. The truthful selections for this project are: aid or polish
  writing; retrieval and discovery; research ideation or execution; draft sections of the paper. Not
  ticked: generating synthetic datasets; proving mathematical claims.
- Written in `paper/main.tex`: the AI use statement now covers the full required-disclosure list split
  into used and not-used, a new Ethics statement section, and a real Reproducibility statement.
  `\todo` count is down from 18 to 12; the rest all need results or reading. A comment block above the
  AI-use review sentence lists the five things that must be TRUE before submitting, since that sentence
  currently claims verification the author has not yet done (notably: no cited paper has been read).
- Housekeeping: `paper/TEMPLATE_INSTRUCTIONS.tex` had been moved to the repo root at some point and was
  showing as a tracked deletion. Content was identical apart from line endings, so it was restored to
  `paper/`. Screenshots and `results/scratch|logs` added to `.gitignore`.
- Still blocked on the author: the Primary Area dropdown options and the License options were not
  captured in the screenshots, so neither can be chosen yet.

### 2026-09-17, evening: arXiv ids verified for the four must-read papers

License for the submission: **CC BY 4.0** (the only option the form offers).

Checked each must-read id against its arXiv page rather than trusting the search results that produced
`related.md`. All four ids are correct, and two gained venue information that changes how they must be
cited (`references.bib` says prefer the published version):

- 2406.16450 -- "Building on Efficient Foundations: Effectively Training LLMs with Structured Feedforward
  Layers", Wei/Moalla/Pascanu/Gulcehre, **NeurIPS 2024**.
- 2407.09835 -- "Investigating Low-Rank Training in Transformer Language Models", same authors,
  **ICML 2024 workshop** (not main-conference). Same group as the above; read the two together.
- 2603.04427 -- "Thin Keys, Full Values: Reducing KV Cache via Low-Dimensional Attention Selection",
  Yao/Chen/Murtadha/Wang, Feb 2026. Confirmed **attention only**, no MLP/GLU content.
- 2609.15037 -- "MoARa: Module-Aware Rank Allocation ...", Kim/Kwak, **EMNLP 2026 Main**.

**Novelty: not contradicted, not established.** None of the four factorizes only the gate, and the nearest
neighbour makes the selection-vs-content argument in attention rather than in the MLP. But this was an
abstract-level check: the per-matrix ablations in 2406.16450 and 2609.15037 could still contain the
gate/up/down result, and that can only be settled by reading their methods and appendices. The five
searches at the bottom of `related.md` are still unrun. No BibTeX was written, and no box was ticked.

### 2026-09-17, about 15:50: the CPU sweep was lost; WSL2 brought forward

- **Failure.** The Exp. A CPU sweep launched at 15:10 died at about 15:29. It was a child process of the
  terminal session that launched it, that session was closed by accident, and the sweep went with it.
  2 of 18 configs had finished: baseline 17.463, `gate_proj` rank 432 ppl 40.505 (x2.32), rank 288
  ppl 1011.630 (x57.93). `posthoc_truncate.py` writes its JSON only after the last config, so **no
  result file exists**. About 20 minutes of a ~6.5 h run were lost. The partial log stays in
  `results/logs/` (not in git). Those numbers are log lines only, agree with the 8-window probe above,
  and must not be used anywhere.
- Lesson: a long job must never be a child of an interactive session. Rule 6 already says `tmux`; the
  Windows side has no `tmux`, which is one more reason to move to WSL2 rather than patch around it.
- **Found while checking, fixed:** the PC was set to sleep after 30 minutes idle on AC power, which
  would have frozen any long run the first time the machine was left alone. Now set to never
  (`powercfg /change standby-timeout-ac 0`, verified 0). Still to do by hand: pause Windows Update for
  a week, since an automatic overnight restart kills a run just as surely.
- **Found while checking, avoided:** the sweep was writing to `results/posthoc/`. `run_posthoc.sh`
  skips a model whose JSON already exists there, and `plot.py` reads everything under `results/`
  except `smoke/` and `scratch/`. So the CPU fp32 result would later have blocked the GPU bf16 rerun
  of SmolLM2-135M and appeared in the paper figure next to GPU numbers, the exact mixing ruled out
  above. Any future CPU run must pass `--out_dir results/scratch/<something>`.
- **Decision (author): do not relaunch the CPU sweep; install WSL2 now.** With the sweep dead, the only
  reason to postpone the reboot is gone. The GPU repeats this sweep in minutes, for several models, in
  the dtype the paper will use, and the overnight pilot grid can start tonight instead of tomorrow.
  The pilot grid is on the critical path to the Sunday go/no-go; the CPU sweep never was, because the
  135M model had to be rerun on the GPU anyway.
- Nothing else was lost: the working tree was clean and in sync with the remote.
- Also decided: the five unrun searches at the bottom of `related.md` get run now, recorded with the
  same discipline as the id check (verified on the paper's own page, no BibTeX, no read box ticked).

### 2026-09-17, evening: the five novelty searches were run

All five queries from the bottom of `related.md` were run as general web searches (they index arXiv well).
Every hit that went into `related.md` was opened on its own arXiv page and its id, title, authors and
venue checked there. No BibTeX was written and no read box was ticked.

**Result: no paper found that factorizes only the gate.** Novelty is still not contradicted and still not
established. But two findings narrow what the paper may claim, and both need reading before Sep 25:

- **LASER, arXiv 2606.00573** (May 2026): its FFN scheme factorizes the gate AND up projections together on
  the same selected channels and leaves the down projection dense. That is an asymmetric FFN treatment
  already in print, split input-side vs output-side rather than gate vs content. The sentence in our
  abstract that existing methods "allocate them identical capacity" is therefore too strong as it stands.
  The abstract is editable until Sep 25, so this does not block Saturday's registration, but it must be
  fixed. `paper/main.tex` was not touched today.
- **FLRC, arXiv 2510.09332** (EMNLP 2025), Appendix A Figure 2: importance scores per projection type in
  Llama-3-8B, with `down_proj` named as the sensitive one. **WeLore, arXiv 2407.11239** (ICML 2025) says
  components differ in how low-rank they become. So per-projection sensitivity is not new in itself. If
  those figures already show gate below up, Exp. A is a confirmation at equal rank, not a discovery, and
  the paper must say so. Whichever way they point, they get cited next to Exp. A.

Also recorded: Masked GLU (2506.23225, gate and value share one matrix), NA-LoRA (2606.31717, adapters on
the gate), Spectron (2602.12429, ICML 2026, uniform low-rank pretraining and its instability). The five
"find the id" placeholders are resolved and verified: CoLA 2502.10940, ASVD 2312.05821, SVD-LLM 2403.07378
(ICLR 2025), Deja Vu 2310.17157 (ICML 2023), GLU variants 2002.05202.

Not covered, and said so in `related.md`: OpenReview's own search over ICLR 2026 submissions (its browser
check blocks automated reading, and one relevant-looking submission, `QSoc7HGc6Q`, could not be verified for
that reason) and Google Scholar "cited by". A Semantic Scholar citation lookup returned 4 citing papers for
2406.16450, which is implausibly few, so it is treated as incomplete rather than as evidence of absence.

### 2026-09-17, evening: abstract registration made ready

- `paper/openreview_abstract_form.md` now holds the value for every field of the OpenReview abstract form,
  in form order, built from the three form screenshots and the ICLR 2027 Author Guidelines page.
- Abstract changed in one place: "Existing architectures and low-rank methods nevertheless allocate them
  identical capacity" became "Standard architectures nevertheless give all three the same capacity". The
  old sentence was false after the searches above (LASER treats the FFN asymmetrically; FLRC, WeLore and
  MoARa allocate rank per module). The new one is true of every LLaMA-style block. Sheet and `main.tex`
  abstract verified identical.
- Checked on the guidelines page: title and abstract stay editable until the paper deadline; authors cannot
  be added or removed after the abstract deadline; a paper withdrawn BEFORE the paper deadline is deleted
  from OpenReview, so registering commits to nothing; first-time authors with no qualified reviewer are
  exempt from reciprocal reviewing, capped at one submission per author; placeholder abstracts are removed.
- Still open: the Primary Area dropdown was never captured. The call for papers lists the usual areas; the
  sheet gives a ranked choice. It only affects reviewer matching.
- The abstract still promises Exp. B, Exp. C and the throughput benchmark, which need a GPU. Whether and
  how the GPU gets used is undecided (author is reluctant to load it for ~100 h). Nothing GPU-related is
  needed for Saturday. Sentences for experiments that do not happen must be deleted before Sep 25.

### 2026-09-17, about 16:15: first probe that measures all three projections

Until now only `gate_proj` had ever been truncated, so the hypothesis itself (gate vs up vs down) had
no evidence either way. Ran a throwaway probe to get a first look before polishing the abstract.

Setup: SmolLM2-135M, wikitext-2 test, **only 8 windows of 1024 tokens**, CPU fp32, 4 threads (the PC was
in use, so the run was throttled), ranks 0.75 / 0.5 / 0.25 of full, whitening calibrated on 8 train
windows. Output in `results/scratch/probe8/` (gitignored, skipped by `plot.py`). Baseline 20.074,
identical to the earlier 8-window probe, so the evaluation is deterministic. A first attempt with 16
windows was killed because it was too slow under load; it produced nothing.

Perplexity as a multiple of the baseline (lower = that projection tolerates the rank cut better):

| rank / full | whitened: gate | up | down | plain: gate | up | down |
|---|---|---|---|---|---|---|
| 0.75 | **x1.10** | x1.22 | x1.18 | **x2.37** | x34.8 | x6.27 |
| 0.50 | **x1.63** | x2.54 | x2.02 | **x81.6** | x1082 | x6628 |
| 0.25 | x13.3 | x114 | **x9.09** | **x3532** | x480159 | x44172 |

- The gate is the most tolerant projection in 5 of the 6 cells. The exception is whitened SVD at the
  lowest rank, where down beats it (x9.1 vs x13.3).
- **Gate beats up in all 6 cells**, by x1.1 at the mildest setting and by more than x100 at the harshest.
  That is the cleanest comparison available, because gate and up have the same shape AND the same input.
  It also cuts against treating gate and up as interchangeable, which is what LASER (2606.00573) does.
- Gate vs down is NOT settled: gate wins at moderate rank, down wins at the extreme. The honest summary
  is "up is the fragile one; gate is robust; down is in between and crosses over", not "gate << content".
- Possible mechanism, untested: an error in the gate passes through SiLU, which flattens it on inactive
  units, while an error in up passes straight through. Worth one paragraph and one experiment later.

**What this is and is not.** It is one 135M model, 8k evaluation tokens, a non-standard calibration size
and CPU precision. It is a reason to keep going, consistent with the README "go" direction on this one
model. It is NOT a result: none of these numbers may appear in the paper or the abstract (rule 1), and the
README criterion needs "most models", which needs either the GPU or a cloud GPU for anything above 360M.
The abstract stays numbers-free and neutral about the direction.

### 2026-09-17, about 16:20: project review and abstract refinement

- Reviewed `posthoc_truncate.py`, `hf_utils.py`, `model.py`, `train.py`, `data.py`, `heal.py`, `bench.py`,
  `tests.py`, `grid.py`. No bug found that would invalidate a result. Equal rank really is equal parameter
  count for all three projections; arms are parameter-matched within 1.5% (tested); data order depends only
  on the seed. Two things to remember when writing: at rank 0.75 the factorized matrix has MORE parameters
  than the dense one (ratio 1.03; break-even is rank 0.727), so plots should use the recorded `param_ratio`,
  and `train.py` times throughput with evaluation pauses included, so `bench.py` is the number to report.
- Abstract rewritten for the registration (same content, no new claims): adds why the block matters (most
  of each block's parameters), presents selection-vs-content as the common description under test rather
  than as fact, and puts the design's strongest point up front: equal shapes make equal rank an
  equal-savings control. `main.tex` and `paper/openreview_abstract_form.md` verified identical. TL;DR and
  keywords sharpened. Title unchanged (a question, true under any outcome).
- Status against the original goal (a small but real contribution): not reached yet. There is a novel,
  well-controlled question, working code, and one encouraging probe. There is no paper-grade result, no
  read literature, no BibTeX, and the GPU question is open.

### 2026-09-17, 16:21: Exp. A CPU sweep relaunched, detached

- **Launched** (author asked for it): SmolLM2-135M, full wikitext-2 test split (297 windows), all three
  projection types, default rank fractions, CPU fp32. Whitened first, then plain, because the whitened
  figure is the one the go/no-go rests on and an interrupted run should lose the less important half.
- Differences from the run that was lost, all deliberate: (1) started outside the terminal session
  (`Win32_Process.Create`), so closing a window cannot kill it; a reboot still can. (2) Output goes to
  `results/scratch/posthoc_cpu/`, never `results/posthoc/`. (3) 4 threads at below-normal priority, because
  the PC is in use; measured in the probe at about 1.7 s per window, the same as 8 threads gave earlier.
  (4) Windows sleep on AC is off. Launcher: `results/scratch/run_cpu_sweep.cmd`. Progress:
  `results/logs/cpu_sweep_status.txt` and `results/logs/posthoc_135M_{whiten,plain}.log` (none in git).
- Expected: about 3 h per variant, about 6 h in total, longer while the machine is busy.
- These are an early look, not paper numbers. The 135M model gets rerun on the GPU in bf16 with the other
  models if the GPU route goes ahead; if it never does, moving these into `results/posthoc/` is a decision
  to take and log then, with the CPU/fp32 provenance stated in the paper.

### 2026-09-17, about 16:45: prior work already reports the gate asymmetry (WeLore, ICML 2025)

Looked inside the full texts of the must-read papers instead of only their abstracts. Result, stated plainly:

- **WeLore (arXiv 2407.11239, ICML 2025) already reports that `gate_proj` is more low-rank than `up_proj` and
  `down_proj`** in pretrained LLMs (Figure 1 caption, LLaMA2-7B; Sec. 2.4 "MLP Gate Projections"), explains
  it through the activation function, builds its method on it (gate, q, k, o treated as low-rank; up, down,
  v not), and draws the same attention analogy we do. Passages are quoted in `related.md`.
- So the central observation of Exp. A is **not new**. Our probe this afternoon agrees with it in direction,
  which makes Exp. A more likely to come out positive and less novel at the same time.
- What WeLore does not contain (searched for, not found): one projection type truncated at a time at equal
  rank with perplexity compared; any from-scratch training with a low-rank gate; symmetric controls.
- 2406.16450 and 2407.09835 turn out to use 2-matrix GELU FFNs, so they have no gate at all. MoARa is about
  gradient projection and gives only an ordering with `mlp.down` least sensitive.

**What this changes.**
1. The novelty of the paper now rests mainly on **Exp. B** (thin gate built from scratch, with thin-up and
   thin-down controls at matched parameters) and secondarily on Exp. C. Exp. A alone would be a controlled
   replication of a known observation on small models: fine for a workshop, thin for the main track.
   Exp. B needs a GPU, so the open GPU decision now matters more than it did this morning.
2. The introduction has to start from WeLore: prior work observed the asymmetry post hoc in pretrained
   models; we test it one projection at a time under equal-rank control, and ask whether it can be used as
   a design rule.
3. The registration abstract needs no change: it claims no novelty for the observation and every sentence in
   it is still true. It should credit the prior observation in the Sep 25 version, after the author has read
   WeLore (rule: nothing about a paper the author has not read).
4. This was found by a targeted look at HTML full texts through an automated reader. The author still has to
   read WeLore Sec. 2.1, 2.4, 3.1 and Figures 1, 3, 7 personally before any of this goes into the paper.

### 2026-09-17, about 17:00: cloud GPU route built (author does not want to load the local GPU)

- New: `cloud_run.py` (runner) and `make_cloud_notebook.py` (builds `cloud_notebook.ipynb`, git-ignored because
  generated). The notebook writes the code to the cloud machine itself, so the private repo is never shared.
  The runner gives each GPU the next unit of work (one model = whitened then plain), keeps going if a model
  fails, skips finished runs, and refreshes `thin_gate_results.zip` after every job. Gated Llama-3.2-1B is
  skipped unless an HF token is provided.
- **Code change, `posthoc_truncate.py`:** new `--dtype auto|float32|bfloat16`. `auto` now means bfloat16 only on
  GPUs with native support (compute capability 8 or higher) and float32 elsewhere. Before, any CUDA device got
  bfloat16, which a T4 only emulates. CPU and RTX 3070 behaviour is unchanged. The cloud run passes
  `--dtype float32` for EVERY model so that Exp. A has one precision throughout.
- **Code change, `train.py`:** new `--amp_dtype auto|bfloat16|float16`, with a `GradScaler` that is enabled only
  for float16. Verified that the default path is unchanged: the smoke run's full validation curve is identical,
  to the last digit, between the old and the new file. The float16 path was exercised on the CPU only; its
  first real test is the GPU smoke run in the notebook's check step, which takes under a minute.
- `python tests.py`: 6/6 after both changes. `posthoc_truncate.py --smoke` passes with `auto` and with
  `--dtype bfloat16`. Runner mechanics tested with fake jobs on two pretend GPUs: parallel dispatch, GPU
  pinning, a failing job, skip-if-done and the zip all behave.
- Not tested, because there is no CUDA here: anything on a real T4. Time estimate for Exp. A on two T4s is
  7 to 8 hours and is a guess from FLOP counts, not a measurement.
- Consequences to remember: (1) cloud Exp. A results are float32 on a T4, and a later 3070 run of the same
  model in bfloat16 would overwrite the file of the same name and must not be mixed into the same figure;
  (2) if the Exp. B pilot runs in the cloud in float16, then ALL of Exp. B has to run there.
- The CPU sweep that started at 16:21 keeps running. Its second half starts a new Python process that loads
  the edited `posthoc_truncate.py`; on a CPU `auto` still resolves to float32, so the run is unaffected.

### 2026-09-17, about 17:15: decision, ALL GPU experiments run on Kaggle

- **Decision (author):** the local RTX 3070 is not used for the experiments. Exp. A, B, C and the benchmark all
  run on Kaggle T4s. Consequence accepted: Exp. B and C run in float16 with loss scaling, Exp. A in float32,
  and the benchmark numbers are T4 numbers. The paper must say "T4", not "consumer GPU with 8 GB"; the ethics
  and reproducibility statements and the setup section in `paper/main.tex` still describe the 3070 and need
  changing once the first cloud results exist. WSL2 is no longer needed.
- `heal.py` and `bench.py` now take `--amp_dtype` like `train.py` (shared helper `pick_amp_dtype` in
  `train.py`), since both hard-coded bfloat16. Checked after the change: `tests.py` 6/6; the train smoke
  validation curve is still identical to the original file; `heal.py --smoke` passes in both precisions;
  `bench.py --smoke` passes.
- `cloud_run.py` gained `grid --stage`, `heal` and `bench`, writes its own console lines to `logs/_runner.log`,
  and packs `results/scratch` so the speed-check result comes back. `make_cloud_notebook.py` gained `--run`
  (the session's plan) and packs every finished result under `results/` INTO the notebook, because a Kaggle
  session starts empty and the runner can only skip what it can see. Tested with a fake result file.
- **Session plan (times are estimates, nothing on a T4 has been measured yet):** 1) speed check + Exp. A,
  7-8 h; then fix `TOKENS["S"]` from the measured tokens/s, before any grid run (rule 3); 2) pilot, 6 runs;
  3-4) `main_S`, 27 further runs; then `heal` and `bench`. Kaggle's free quota was about 30 GPU-hours a week
  when last checked, which would cover sessions 1-2 this week and `main_S`, `heal`, `bench` next week, but
  probably NOT size M. If the quota is what it was, Exp. B is a one-size experiment unless more GPU time is
  found; say so in the limitations rather than shrinking seeds to squeeze M in.
- Risk carried: the float16 training path has never run on a real GPU. Its first test is the smoke run in
  the notebook's check step, and its second is the 20M-token speed check, both in session 1 and both cheap.

### 2026-09-17, evening: where else this paper could go (deadlines checked on the venues' own pages today)

One archival venue at a time: a paper under review at ICLR cannot also be under review elsewhere. A paper
withdrawn from ICLR before its Sep 25 paper deadline is deleted, which frees it for any of these.

| Venue | Open now? | Deadline (checked 2026-09-17) | Fit |
|---|---|---|---|
| ICLR 2027 | yes | abstract Sep 18 AOE, paper Sep 25 AOE | current target; 8 days is very tight |
| AISTATS 2027 | yes | abstract Tue Sep 29, paper Tue Oct 6, 2026; 8 pages; double-blind | in scope ("Deep learning (theory, architectures, ...)") |
| ARR October 2026 cycle | yes | Oct 12, 2026; feeds NAACL 2027 and COLING 2027 (commit Dec 23) | good; needs complete OpenReview profile incl. ORCID; check reviewer rules for a sole first-time author |
| TMLR | always (rolling) | none | best fit after the WeLore finding: judged on correctness of claims, not novelty |
| CPAL 2027 (Tokyo, Mar 23-26) | yes | abstract Nov 23, paper Dec 5, 2026; spotlight track Jan 18, 2027 | best topical fit: the conference is about low-rank and sparse structure |
| NeurIPS 2026 workshops | effectively closed | most closed Aug 29 to Sep 11; the few still open are off-topic | none |

Not announced yet on their own sites: ICML 2027 (an aggregator says Jan 22, 2027; unofficial), ACL 2027 via the
ARR January cycle, COLM 2027, MLSys 2027, ICLR 2027 workshops. arXiv is open any time but a first submission
to cs.LG needs an endorser.

## 2026-09-19

### Kaggle session 1 taken in: Exp. A on 7 models, T4, float32 (first paper-grade result)

The author downloaded `thin_gate_results.zip` (84 KB) from the Kaggle Output tab; unzipped into the repo
root with Python's `zipfile`. Session ran 21:04 to 05:24 UTC on Sep 17-18, about 8.3 h, inside the 7-8 h
guess. `logs/_runner.log`: two Tesla T4 (15.6 GB, compute capability 7.5, so float32 for Exp. A and
float16 for Exp. B); `tests.py` 6/6, `posthoc smoke` ok, `train smoke` ok; torch 2.10.0+cu128,
transformers 5.0.0, datasets 5.0.0. No job failed. `meta-llama/Llama-3.2-1B` was skipped (gated, no token).

- **Throughput check** (size S, dense, 20M tokens, float16 with loss scaling, one T4, `torch.compile` on):
  **62,990 tokens/s**, peak memory 2.2 GB, val loss 5.66. One 300M-token size-S run is therefore about 1.3 h,
  under the 1.5 h rule, so `TOKENS["S"] = 300e6` stays. Runner's own estimate: 6 pilot runs on two T4s,
  about 4 h. These numbers live only in `results/scratch/throughput_check.json` and `logs/` (gitignored),
  hence recorded here. `TOKENS["M"]` is still unmeasured; measure before `main_M`, never during.
- **Exp. A, 14 result files** in `results/posthoc/` (7 models x whiten/plain), 18 settings each (3 projections
  x 6 rank fractions), full wikitext-2 test split, `dtype: float32`, `device: Tesla T4`. Baseline
  perplexities: SmolLM2-135M 17.493 (CPU full split gave 17.463; same evaluation, different precision),
  SmolLM2-360M 12.955, SmolLM2-1.7B 9.091, Qwen2.5-0.5B 14.651, Qwen2.5-1.5B 10.403, TinyLlama v1.1 8.586,
  OLMo-2-0425-1B 10.343. All plausible.
- Committed `results/posthoc/` and the outputs of `python plot.py` (run on the Mac in a fresh `.venv` with
  numpy and matplotlib only): `paper/figures/posthoc_{whiten,plain}.pdf`, `paper/tables/posthoc_*.tex`.

**What the whitened figure shows (perplexity / baseline; the go/no-go figure), stated plainly:**

- Over the 42 cells (7 models x 6 ranks), the gate is the most tolerant projection in 26, down in 16, up in 0.
- **Gate beats up in 41 of 42 cells.** The one exception is SmolLM2-360M at r/d = 0.25 (26.1 vs 26.0, a tie).
- At r/d = 0.75 and 0.5 the gate is the best projection in every one of the 7 models (Qwen2.5-0.5B at 0.5 is a
  near-tie with down, 1.76 vs 1.71).
- At r/d = 0.25 and below, down overtakes the gate in 5 of 7 models (SmolLM2-135M/360M/1.7B, Qwen2.5-1.5B,
  TinyLlama). Gate stays best at every rank only in OLMo-2-1B and Qwen2.5-0.5B. This is the same crossover
  the 8-window CPU probe showed on 135M, now seen across models.
- Plain SVD (42 cells): gate best in 34, up in 6 (SmolLM2-360M at 0.75/0.5/0.375/0.0625, SmolLM2-1.7B at
  0.75, Qwen2.5-1.5B at 0.0625), down in 2. Plain truncation is destructive everywhere below r/d = 0.75, as
  before, so it is the secondary figure.

Honest one-line summary: **up is the fragile projection in every model; the gate is the most tolerant at
moderate rank in every model; gate vs down crosses over at aggressive rank, in down's favour.** This is
consistent with WeLore's post-hoc observation (gate more low-rank than up/down) and refines it: the
advantage over up is universal, the advantage over down is rank-dependent. The paper must say both.

Against the README criterion ("go if in most models the gate curve sits clearly below up and down at
equal rank, whitened"): at the ranks Exp. B actually uses (d/2 and d/4, i.e. r/d = 0.5 and 0.25) the
criterion is met for gate vs up in all 7 models and for gate vs down at 0.5 in all 7, at 0.25 in 4 of 7.
That is a "go" for gate vs up and a qualified "go" for gate vs down; the formal decision waits for the
pilot's noise floor (Sun Sep 27, see the venue entry below). None of these numbers goes into the paper
by hand; `plot.py` regenerates the figure and table from the JSONs.

Open plotting item, not changed today: the figure's x-axis is rank / full rank. Because d_ff/d differs
between model families, the same rank fraction is a different parameter saving (recorded as
`param_ratio`; e.g. r/d = 0.75 is 1.03x dense for SmolLM2 but 0.88x for Qwen2.5-0.5B). Decide before the
Results section whether to plot against `param_ratio` instead, and say which in the caption.

The detached CPU sweep of Exp. A on SmolLM2-135M (started 2026-09-17 16:21) is **dropped**: its output could
not be retrieved, and the T4 float32 run of the same model supersedes it. Nothing from it was ever in
`results/posthoc/`.

### 2026-09-19: ICLR 2027 missed; new target is CPAL 2027 (decision, author)

The ICLR 2027 abstract registration was not filed before Sat Sep 19, 07:59 EDT. ICLR 2027 is therefore out.
Nothing was registered, so there is nothing to withdraw and no de-anonymization exposure. Deadlines were
re-checked today on the venues' own pages; this table supersedes the 2026-09-17 one.

| Venue | Deadline | Status for a sole first-time undergrad author |
|---|---|---|
| **CPAL 2027, Proceedings Track (chosen)** | abstract registration **Nov 23**, paper **Dec 5, 2026**; notification Feb 1; conference Mar 23-26, 2027 | 9 pages main text plus unlimited refs/appendix; double-blind; archival (PMLR); arXiv preprint allowed; no reciprocal-reviewing rule found; "sparsity, structured sparsity, low rank" is a listed topic; AI tools permitted with authors fully responsible. Template and OpenReview links go up on cpal.cc/openreview/ "as they become active": not up yet. |
| CPAL 2027 Recent Spotlight | Jan 18, 2027 | non-archival, single-blind, 250-word abstract plus material; the safety net if the proceedings paper is not ready or is rejected elsewhere |
| AISTATS 2027 | abstract Sep 29, paper Oct 6 | **out**: every submission must nominate a reciprocal reviewer with roughly 2nd-year-PhD experience and top-venue publications; a sole undergrad cannot, and the CFP says such submissions may be desk-rejected |
| ARR October 2026 (NAACL/COLING 2027) | Oct 12 | **out**: from this cycle, review is only guaranteed with a "qualified service contributor", otherwise a lottery |
| MLSys 2027 | Oct 30, 20:00 UTC | backup only: 10 pages, double-blind, no reciprocal rule, but a systems venue and only six weeks, which would mean writing while the grid runs |
| TMLR | rolling | fallback after a CPAL rejection: judged on whether the claims are supported |
| ICML 2027 | not announced (aggregators guess Jan 2027) | not planned for |

Consequences: the schedule stretches from 8 days to 11 weeks (see `HANDOFF.md`, "Next, in order"). Exp. B
at sizes S and M with three seeds, Exp. C and the benchmark become realistic on Kaggle's quota, and the
must-read papers can be read before any of them is cited. The go/no-go moves to **Sun Sep 27, 6 pm** (Exp.
A figure plus the pilot's noise floor); experiments freeze **Nov 8**; abstract registration Nov 23; aim to
upload Dec 3. The paper stays in the ICLR style file as a placeholder until the CPAL template is published
(both allow 9 pages of main text). The AI-use, ethics and reproducibility statements stay.

## 2026-09-20

### Kaggle session 2 taken in: Exp. B pilot, 6 runs at size S, T4, float16 (first from-scratch result)

The author downloaded the zip (34 KB, named `thin_gate_results (1).zip` by the browser) and dropped it in
the repo root; unzipped with Python's `zipfile`. Session ran 19:14 to 23:50 on Sep 19 (about 4.6 h, against
the runner's 4 h estimate). `logs/_runner.log`: two Tesla T4, `tests.py` 6/6, posthoc smoke ok, train smoke
ok, same library versions as session 1 (torch 2.10.0+cu128). No job failed; no nan, inf or overflow in any
training log. The 14 Exp. A files in the zip are byte-identical to the committed ones (the notebook carries
them so the runner can skip them). New: `results/train/S_*.json`, 6 files, committed.

Recipe, identical across arms (rule 3): size S (6 layers, 6 heads, d = 384, d_ff = 1024), 300M tokens,
4577 steps of 65,536 tokens, lr 1e-3 with 200 warmup steps and cosine to 1e-4, weight decay 0.1, grad clip
1.0, float16 with loss scaling, `torch.compile` on, one T4 per run, two runs at a time. Validation every 250
steps and at the end. All four rank-d/4 arms have exactly the same MLP parameter count (5,529,600 per
layer-set vs 7,077,888 dense); gate/up/down rank is 96 = d/4; shrunk uses d_ff = 800.

| Run | Final val loss | Gap to dense mean | Tokens/s | Wall |
|---|---|---|---|---|
| S_dense_s0 | 4.0867 | | 54,466 | 92 min |
| S_dense_s1 | 4.0834 | | 54,380 | 92 min |
| S_shrunk_r4_s0 | 4.1043 | +0.019 | 60,482 | 83 min |
| S_thin_up_r4_s0 | 4.1171 | +0.032 | 60,005 | 84 min |
| S_thin_gate_r4_s0 | 4.1250 | +0.040 | 59,893 | 84 min |
| S_thin_down_r4_s0 | 4.1470 | +0.062 | 56,171 | 89 min |

**Noise floor** |S_dense_s0 − S_dense_s1| = **0.0033** (dense mean 4.0850). The gaps above are 6x, 10x, 12x
and 19x that floor, so all four parameter-matched arms are clearly worse than dense, and the differences
between them are also outside the floor, with the caveat that each non-dense arm has one seed and the floor
itself comes from only two seeds.

**What the pilot shows, stated plainly:**

- At equal parameter count, the ordering is **dense < shrunk < thin up < thin gate < thin down**. The plain
  narrow SwiGLU (shrunk, d_ff = 800) is the best way to spend the budget; every factorized arm is worse.
- **Thin gate does not beat shrunk** (it is 0.021 worse, 6x the floor), and it does **not** sit within the
  noise floor of dense. Neither pilot "go" condition in `README.md` is met.
- Thin gate is **worse than thin up** by 0.008 (2.4x the floor, one seed each). This is the opposite of
  Exp. A, where the up-projection was the fragile one in 41 of 42 cells. Thin down is the worst arm, which
  matches Exp. A only in the sense that down was never the most fragile there either; at r/d = 0.25 Exp. A
  actually had down overtaking gate in 5 of 7 models. So **post-hoc rank tolerance in pretrained models does
  not predict which projection to factorize when training from scratch**, at least at this scale.
- The ordering is not a late fluctuation: from step 1250 (27% of training) to the end, every one of the 15
  evaluations has the same order dense < shrunk < thin up < thin gate < thin down.
- Speed: shrunk, thin up and thin gate are each 10-11% faster than dense in tokens/s; thin down only 3%
  (its extra matmul sits on the wide d_ff side). Absolute tokens/s (54-60k) is below session 1's 63k for
  the same dense recipe; the difference is probably the two concurrent jobs sharing the CPU data path
  (unverified). Only the ratios between arms are used.

**Against the README go/no-go, ahead of the Sun Sep 27 decision (the author's call, not made here):**
the Exp. A criterion is met (logged 2026-09-19); the pilot criterion is not. Read together: the gate is the
most rank-tolerant projection *after* training, but factorizing it *during* training is not the cheapest
way to save those parameters, and up is a (slightly) better choice than gate. That is a pivot or no-go
signal for the method as titled, but it is a clean, controlled finding in its own right. Options the author
can weigh, none decided:

1. Finish the key arms of `main_S` (dense, thin gate, thin up, thin down, shrunk; 9 more runs, about 7 h on
   two T4s) to put three seeds behind every number before deciding. The 0.008 gate-vs-up gap especially
   needs seeds. This is the planned next session anyway.
2. Add the rank d/2 arms (`thin_gate_r2`, `shrunk_r2`, and `thin_up_r2` if added to `grid.py`) since Exp. A
   showed the gate clearly best at r/d = 0.5 in all 7 models; if the from-scratch picture flips at milder
   rank, the paper is about the rank regime. `thin_up_r2` and `thin_down_r2` are not in the grid today.
3. Reframe as the contrast result: "post-hoc tolerance does not transfer to pretraining" with Exp. A, B
   and C all kept. Title and abstract would change; rule 2 says a negative result is fine, a stretched
   claim is not.
4. No-go: stop after `main_S` if three seeds confirm the pilot.

Nothing from this session goes into `paper/main.tex` yet: `plot.py` wrote `paper/figures/training.pdf` and
`paper/tables/training.tex` (a 5-row table with one seed per non-dense arm), and the Exp. B results
subsection stays a `\todo` until the seeds are in and the framing is decided. `plot.py` regenerated the
Exp. A tables byte-identical and the Exp. A PDFs identical except for their embedded timestamp; those PDFs
were restored rather than committed as noise.

Housekeeping: `.gitignore` now ignores `thin_gate_results*.zip` so a browser-renamed download does not show
as untracked. `logs/train__*.log` stay gitignored; every number needed from them is in this entry or in the
JSONs.

### 2026-09-20, 01:20: launch deadline added to the runner before session 3 (main_S)

`main_S` has 27 runs left after the pilot, about 19 h on two T4s at 1.5 h each, so it cannot fit one 12 h
Kaggle commit. A commit that overruns is killed and Kaggle keeps none of its output, so the notebook's
"a cut-off session still leaves usable results" was only true for an interactive session. `cloud_run.py`
now takes `--max_hours` and `--job_hours` (default 1.7): after `max_hours - job_hours` it starts no new job
and lets the running ones finish. `make_cloud_notebook.py` passes `--max_hours 11` to the `main_S`, `main_M`
and `lr` steps, so the last size-S run starts before 9.3 h and the commit ends inside 12 h with the zip
intact. Expected yield of session 3: about 14 of the 27 runs, the 9 key-arm runs first (three seeds for
dense, thin gate, thin up, thin down, shrunk), then 5 from the rest. Session 4 finishes the stage; the
runner skips what is done. `--job_hours` must be raised for size M once its run time is measured.

Tested on the Mac with fake jobs and a faked two-GPU count (no torch here): no cap runs everything, a
deadline in the past skips everything with a `[skip]` line in `_runner.log`, a deadline in the future runs
everything. `tests.py` is unaffected (rule 5 does not list `cloud_run.py`). Notebook rebuilt.

### 2026-09-20, 15:15: Kaggle session 3 taken in: main_S, 15 runs, key arms now at three seeds

Zip renamed `thin_gate_results_session3_mainS.zip` (browser name was `thin_gate_results (2).zip`). Session
ran 05:41 to 16:11 UTC, 10.5 h. Checks ok, same library versions. The launch deadline worked as designed:
after 9.3 h the runner logged 12 `[skip]` lines (seeds 1 and 2 of the six "rest" arms) and the commit
finished inside 12 h with the zip intact. No job failed; no nan, inf or overflow in any log. 15 new files
in `results/train/`, committed; the 6 pilot files and the 14 Exp. A files came back byte-identical.

Now in hand at size S, 300M tokens: dense, shrunk r4, thin gate r4, thin up r4, thin down r4 at seeds
0-2; thin gate r2/r8, shrunk r2/r8, reinvest r4, all-lowrank r4 at seed 0. Still to do for `main_S`:
those six arms at seeds 1-2 (12 runs, one more session).

| Arm | s0 | s1 | s2 | mean | range | MLP params |
|---|---|---|---|---|---|---|
| dense | 4.0867 | 4.0834 | 4.1125 | 4.0942 | 0.029 | 7,077,888 |
| shrunk r2 | 4.0969 | | | | | 6,359,040 |
| shrunk r8 | 4.1009 | | | | | 5,142,528 |
| reinvest r4 | 4.1039 | | | | | 7,064,064 |
| shrunk r4 | 4.1043 | 4.1043 | 4.1124 | 4.1070 | 0.008 | 5,529,600 |
| thin gate r2 | 4.1076 | | | | | 6,340,608 |
| thin up r4 | 4.1171 | 4.1145 | 4.1147 | 4.1155 | 0.003 | 5,529,600 |
| thin gate r4 | 4.1250 | 4.1151 | 4.1145 | 4.1182 | 0.011 | 5,529,600 |
| thin gate r8 | 4.1313 | | | | | 5,124,096 |
| thin down r4 | 4.1470 | 4.1476 | 4.1360 | 4.1435 | 0.012 | 5,529,600 |
| all lowrank r4 | 4.1682 | | | | | 5,474,304 |

**The noise floor moved.** Dense seed 2 finished at 4.1125, 0.029 above seeds 0 and 1, and it was
behind them at every evaluation from step 250 on (not a late spike; the largest jump between consecutive
train-loss prints is 0.15, normal). The seed sets both the init and the data order (`data.py`), and every
other seed-2 arm landed where its seeds 0 and 1 did (gate 4.1145, up 4.1147, shrunk 4.1124), so this is one
unlucky dense run, not a hard data order. The README floor is now the dense range, **0.029**, ten times the
pilot's 0.003. The pilot's "12x the noise floor" language is withdrawn.

**What three seeds say, stated plainly:**

- Mean order unchanged from the pilot: **dense < shrunk < thin up < thin gate < thin down**, with
  all-lowrank last.
- **Thin gate never beats shrunk.** Paired by seed, shrunk is better at s0 (by 0.021) and s1 (by 0.011)
  and tied at s2 (0.002). At seed 0 the same holds at rank d/2 (shrunk better by 0.011) and d/8 (by 0.031):
  the gap widens as the gate gets thinner.
- **Thin gate and thin up are indistinguishable**: means 0.003 apart, seed ranges 0.011 and 0.003, and the
  order flips between seeds (gate worse at s0, better at s1, tied at s2). The pilot's "up beats gate" is
  not supported; "gate beats up" is not either.
- **Thin down is clearly worst** of the three single-projection arms: 0.025 to 0.028 behind gate and up on
  the mean, worse at every seed, and the seed ranges do not overlap. Factorizing all three is worse still.
- Against dense: the four parameter-matched r4 arms are 0.013 (shrunk) to 0.049 (down) worse on the mean.
  With the dense range at 0.029, only thin down and all-lowrank are outside it; shrunk, thin up and thin
  gate sit inside the band of dense seeds. Reinvest (thin gate + wider d_ff, dense parameter count) is
  0.010 worse than dense on the mean at one seed, inside the band.
- Tokens/s in the table is not a speed measurement: the same arm ranged from 54k to 65k depending on which
  job shared the machine (dense s2 62k vs s0/s1 54k). Speed claims wait for `bench.py`.

**Against the go/no-go criteria, with seeds:** the pilot verdict stands and is now firmer. Thin gate r4 is
not within the (small) pilot floor of dense, does not beat shrunk at any of three ranks, and is not better
than thin up. The gate hypothesis as titled fails at this scale. What the two experiments show together is
sharper than a null: **the projection that tolerates truncation best after training (gate, Exp. A) is not
the one that trains best at low rank (gate = up, Exp. B), and the one that is most fragile after training
(up) is not the one that hurts most when trained thin (down).** Post-hoc rank tolerance does not predict
from-scratch trainability, in either direction. That is the contrast result (option 3 of 2026-09-20).

Options for the Sep 27 decision, updated (author's call):

1. **Contrast paper** (recommended by the assistant, not decided): keep Exp. A, B and C; retitle. Exp. B's
   headline figure becomes a rank sweep of all three projections from scratch, which needs `thin_up_r2/r8`
   and `thin_down_r2/r8` added to `grid.py` (a code change: rerun `tests.py` in the notebook's check step,
   rule 5) and run at three seeds: 12 runs, one session, on top of session 4's 12 remaining `main_S` runs.
2. Finish `main_S` as planned first (session 4, 12 runs, fits one commit), then decide.
3. No-go. Not recommended: the controlled negative is publishable under rule 2, and CPAL lists low-rank
   structure as a topic.

Size M is unchanged in the plan (measure throughput first). Nothing from session 3 is in `paper/main.tex`
yet; `figures/training.pdf` and `tables/training.tex` are regenerated (11 arms) and committed.

### 2026-09-20, night: screening stage for variants that could beat shrunk (decision, author)

The author asked for something new that beats `shrunk_r4` at the same parameter count, and chose to screen
before finishing `main_S`, with structural changes to the gate allowed. Reasoning recorded first, then the
design.

**Why the thin gate loses, as far as three seeds can tell.** A SwiGLU neuron computes silu(k_i·x)(u_i·x)v_i:
a product of two linear features of x, written along v_i. Rank-96 gate keys k_i all lie in one 96-dim
subspace, so every gating decision in the layer sees only a 96-dim projection of the residual stream. Thin
up restricts the other factor of the same product, and the silu is the only asymmetry between them, which is
why thin gate and thin up land together. Thin down puts the *output* directions v_i in a 96-dim subspace,
which is the harshest cut, and it is the worst arm. A narrower dense block (shrunk, 800 neurons with full
keys) keeps every direction and simply has fewer neurons; at this scale that is the better trade at every
rank tried. Exp. A's post-hoc tolerance of the gate is a property of trained gates (whitened, errors on
inactive neurons are cheap), not evidence that gates can be *trained* in a small subspace.

**Candidates, one sentence each, all at 921,600 MLP params/layer (±0.35%; checked by `tests.py`):**

1. *Optimization fixes* for the existing thin gate: spectral init (factors = top-96 SVD of a dense random
   init) and/or no weight decay on the factors (both factors decaying at 0.1 makes the product decay about
   twice as fast). Diagnostic: is the deficit an optimizer artifact? Expected gain small.
2. *Grouped gate* (`--gate_groups 4`, d_ff 1064): the gate computes 266 values, each gating 4 neighbouring
   units; keys stay full rank, there are just fewer of them. This is the original thesis restated:
   selection needs fewer *units*, not fewer *dimensions*. Grouped up / grouped down are the controls.
3. *Monarch gate* (`--gate_monarch 4`): two block-diagonal matrices with a fixed shuffle between them;
   exactly the parameter count of rank 96 (135,168) but full rank, every input reaches every output. The
   Monarch literature trains such matrices near dense quality; best prior of beating shrunk.
4. *Nonlinear bottleneck gate* (`--bottleneck silu` / `norm_silu`): B(silu(A x)), same parameters; the gate
   becomes a tiny two-layer network. Cheap; likely ≈ thin gate.
5. *Warm-started thin gate* (`--thin_at 0.25` / `0.1`): dense gate for the first 25% / 10% of steps, then
   each gate is replaced by its whitened rank-96 SVD (grams from 32 calibration windows drawn with a private
   RNG, so the training order is untouched; the optimizer keeps its state for every other parameter; the
   compiled model is rebuilt) and training continues on the same schedule. About 1.2% / 0.5% more train
   FLOPs, so it is **not** a matched from-scratch arm; it is the bridge between Exp. A and Exp. B and is
   reported separately, with the validation loss right before and after the swap.

**Stage `screen`** (`grid.py`, size S, seed 0, priority order; the launch deadline drops the tail):
monarch_gate_b4, grouped_gate_g4, thin_gate_r4_spectral_nowd, warm_gate_r4_f25, bottleneck_gate_r4,
thin_gate_r4_nowd, thin_gate_r4_spectral, warm_gate_r4_f10, bottleneck_norm_gate_r4, grouped_up_g4,
grouped_down_g4, thin_gate_r4_halfwd, monarch_gate_b2_narrow (13 runs, about 9-10 h on two T4s). Session 4
is `--run screen,main_S`, so leftover time goes to the 12 remaining `main_S` runs.

**Decision rule (seed 0; shrunk s0 = 4.1043, its last evals 4.1276 / 4.1150 / 4.1061; dense s0 4.0867):**
promote to seeds 1-2 plus the arm's up/down controls if final val ≤ 4.100 and below shrunk at each of the
last three evals; run seed 1 first if 4.100 < val ≤ 4.1043 (promote only if seed 1 is also ≥ 0.004 better
than shrunk s1 = 4.1043); drop otherwise. Warm start never enters the matched table. Rule 4 still applies:
a single-seed gap is a trigger, not an effect. Honest expectation: none is guaranteed; Monarch and grouped
have the best chance, the fixes are diagnostic, the bottleneck is a long shot.

**Fairness (rule 3):** lr, schedule, tokens, batch, data order and seed are untouched for every arm. The
init and decay variants are part of an arm's definition, not tuning; if one wins, the paper says the
baselines received no such variants.

**Verification done on the Mac** (a CPU torch was installed into `.venv` for this; the Kaggle check step
repeats everything compiled on the GPU): `tests.py` 11/11 (the 6 old tests plus spectral init, bottleneck,
grouped gate, Monarch, and the mid-training swap being exact at full rank while keeping Adam state);
`train.py --smoke` for the baseline and all five new paths; the baseline smoke's validation curve is
bit-identical to the previous commit's (`6.25308, 6.2524`), so every old arm is unchanged; the warm-start
smoke thins at step 10 with params 73,728 → 57,344 and the JSON reports the final model. Parameter matching
verified at S, M and L (table via `grid.py --table`).
