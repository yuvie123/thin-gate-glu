# Handoff notes for coding sessions in this repo

Read `README.md` (commands, go/no-go criteria, honesty rules) and `PLAN.md` (idea, abstract, titles, schedule) before doing anything. `SETUP_PC.md` and `setup_pc.sh` cover machine setup. `related.md` is the reading list.

## Who and what
- The author is a first-year CS undergraduate writing a first research paper, comfortable with Python and **new to PyTorch**. Explain ML-research steps (baselines, seeds, noise floor, out-of-memory fixes) instead of assuming them.
- Project: a single ICLR 2027 submission on **thin-gate GLU**: factorize only the gate projection of SwiGLU to low rank, keep up/down dense. Abstract due 2026-09-18 AOE (Sat Sep 19, 07:59 EDT); paper due 2026-09-25 AOE (Sat Sep 26, 07:59 EDT). Go/no-go decision on Sun 2026-09-20. Experiments freeze Wed Sep 23 noon.
- Ideas already ruled out by a novelty search (do not revisit): smaller Q/K than V dimension (published, arXiv 2603.04427); lossy LM-head vocabulary shortlisting; frequency-aware embedding compression.

## Machines
- **This GPU PC** (Ryzen 7, RTX 3070 **8 GB**, Windows + WSL2 Ubuntu) runs every real experiment. Size everything for 8 GB.
- A weak MacBook is used only for editing, paper writing and plotting. Results travel between machines as small JSON files through git (`results/`, `paper/figures`, `paper/tables`). Never commit data, weights or checkpoints.
- The GitHub repo is **private and must stay private** until reviews finish (double-blind). Never change its visibility, never put names, usernames, repo URLs or machine names in the paper, code comments or result files.

## Status (update this section as work proceeds)
- 2026-09-17 (Day 0): all experiment code written; `tests.py` passes (6 tests).
- **GPU route, decided 2026-09-17: everything runs on Kaggle T4s, not on the local RTX 3070** (the author does
  not want to load it). WSL2 is therefore not needed. `make_cloud_notebook.py --run <steps>` builds a
  self-contained notebook with the code and all finished results inside; the author uploads it, runs it,
  downloads `thin_gate_results.zip`, unzips it here. Exp. A is float32, Exp. B/C float16 with loss
  scaling, all on T4s; never pool with runs from another GPU or precision. Session plan in `README.md`
  and `notes.md`. Nothing has run on a T4 yet. Windows sleep on AC is set to never.
- **Novelty narrowed 2026-09-17:** WeLore (arXiv 2407.11239, ICML 2025) already reports that `gate_proj` is
  more low-rank than `up_proj`/`down_proj` in pretrained LLMs. Exp. A is therefore a controlled test of a
  known observation; the novel part is Exp. B (thin gate from scratch, with controls), which needs a GPU.
  The intro must start from WeLore. Details in `related.md` (top entry) and `notes.md`.
- **First evidence (probe only, not for the paper):** on SmolLM2-135M, 8 windows, the gate tolerated
  truncation best in 5 of 6 settings and beat the up-projection in all 6; gate vs down crosses over at
  the lowest rank under whitening. Table in `notes.md`. Encouraging, one model, proves nothing yet.
- **No paper-grade experiment result exists yet.** A CPU fallback sweep of Exp. A (SmolLM2-135M, fp32) was started
  and lost after 2 of 18 configs when the session that owned it was closed; it was deliberately not
  relaunched. What it did establish: the evaluation is sound (full-split baseline perplexity **17.463**,
  plausible tens), and two real bugs are fixed (`Salesforce/wikitext` dataset id under datasets>=4;
  fp32 instead of emulated bf16 on CPU). The Windows `.venv` (Python 3.12, torch CPU) is an interim
  tool, not the machine of record. Never write CPU results into `results/posthoc/`. See `notes.md`.
- Paper: title and the numbers-free abstract are submission-ready; the AI-use, ethics and reproducibility
  statements are written. 12 `\todo`s remain, all needing results or reading. **`references.bib` is
  still empty and no paper in `related.md` has been read yet** -- the biggest risk to Sep 25, ahead of the
  experiments. The four must-read arXiv ids are verified against their pages (ids only, not read).
- OpenReview form inspected 2026-09-17. The PDF is NOT required at the abstract deadline, but Title,
  Authors, Keywords, Abstract, Primary Area, Code of Ethics, Paper Visibility, Submission Requirements,
  Reciprocal Reviewing Author, Reciprocal Reviewing Exemption, AI Assistance and License all are.
  **The reciprocal-reviewing fields cannot be changed after the abstract deadline**, and a first-time
  author must claim the exemption there or risk desk rejection. License is CC BY 4.0 (the only option).
  Still missing: the Primary Area dropdown options.

### Next, in order
1. **File the abstract form by Friday evening** (hard limit Sat Sep 19, 7:59 AM EDT) from
   `paper/openreview_abstract_form.md`.
2. Kaggle session 1: upload `cloud_notebook.ipynb` (speed check + Exp. A). Check the log for the 135M
   baseline near 17.46. Unzip the results here, commit, run `python plot.py`.
3. Fix `TOKENS["S"]` in `grid.py` from the measured tokens/s, BEFORE any grid run; rerun `tests.py`.
4. Session 2: `--run pilot`. Sunday 6 pm: go/no-go on the Exp. A figure plus the pilot.
5. Sessions 3-4: `--run main_S`, then `--run heal,bench`. Experiments freeze Wed Sep 23 noon.
6. The author reads WeLore first, then the other must-reads (`related.md`); BibTeX only after reading.
7. Update `paper/main.tex` where it still says the experiments ran on one 8 GB consumer GPU.

## Rules for this project (non-negotiable)
1. **Never fabricate or guess** results, citations, or benchmark numbers. Every number in the paper comes from a JSON in `results/` via `plot.py`. `paper/references.bib` entries are exported by the author from the paper's own page after reading it, never written from memory.
2. Report failures and unflattering results plainly (e.g. low-rank arms not being faster). A negative result is acceptable; an unsupported claim is an ethics violation at ICLR.
3. Fair comparisons: identical data order, token budget, schedule, learning rate and seeds across arms. Never tune the method more than the baseline. If `TOKENS` in `grid.py` must change for time, change it before a grid starts and rerun everything at that size.
4. Compare every gap to the seed-to-seed spread of the dense baseline before calling it an effect.
5. Run `python tests.py` after any change to `model.py`, `grid.py`, `posthoc_truncate.py` or `data.py`.
6. Long jobs run inside `tmux`. Don't start a second GPU job while one is running (8 GB).
7. Keep a dated `notes.md` log of every run launched, failures, and decisions.
8. AI use is disclosed in the paper (`paper/main.tex`, AI use statement). Keep that statement true as the work evolves. The author must understand and be able to defend everything; explain what you do and why.
9. Commit with clear messages; push small result files so the Mac can plot and write.
