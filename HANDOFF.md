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
- 2026-09-17: all code written and smoke-tested on CPU only. `tests.py` passes (6 tests). **No real experiment has run; `results/` is empty.** `setup_pc.sh` has only been syntax-checked, never executed; expect to debug it.
- Next: finish `setup_pc.sh`, then the Day 1 commands in `README.md` (Exp. A on SmolLM2-135M, data download, throughput check, overnight pilot grid in tmux).

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
