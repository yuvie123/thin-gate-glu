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
