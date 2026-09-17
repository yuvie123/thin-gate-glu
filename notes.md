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
