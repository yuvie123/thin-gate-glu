# ICLR 2027 first-paper plan: idea, abstract, titles, 8-day schedule

## Context

You are a first-year undergrad, new to PyTorch, with **one RTX 3070 (8 GB)**, one allowed ICLR 2027 submission, the abstract due **Sep 18 AOE (= Sat Sep 19, 07:59 EDT)** and the paper due **Sep 25 AOE (= Sat Sep 26, 07:59 EDT)**. Today is Thu Sep 17. I ran live novelty checks while planning; two of my first-choice ideas were already taken, which changed the recommendation (see "Killed ideas").

Things I verified on iclr.cc / the ICLR blog that you must act on **before the abstract deadline**:
- **The author list freezes at the abstract deadline.** No one can be added or removed after Sep 18 AOE. If your colleague, a TA, or a professor might co-author, add them (with an OpenReview profile) *now*. A co-author who has published at a major venue also matters for the reciprocal-reviewer rule, which is the source of your 1-paper limit.
- Title and abstract **can** be revised until Sep 25, but the final paper "shouldn't read like a different paper." So submit a conservative, numbers-free abstract tomorrow and refine it later.
- 9 pages main text; a **mandatory AI-use statement** section (not counted toward pages); reproducibility statement recommended; style files: `https://media.iclr.cc/Conferences/ICLR2027/iclr-2027-style-files.zip`.
- **All submissions are de-anonymized at the end of review, including rejected ones.** Your name will be publicly attached to whatever you submit. This is the strongest reason to follow your own rule: modest, fully supported claims.
- Primary area: your screenshot shows the OpenReview venue list, not the area dropdown. In the submission form, pick the area that mentions foundation models/LLMs or, failing that, infrastructure/systems/efficiency. Paste me the dropdown and I'll choose with you.

### Two-machine workflow (weak MacBook here; Ryzen 7 + RTX 3070 PC elsewhere)
**Nothing heavy runs on this Mac.** No training, no model evaluation and no large downloads happen here.
- **Mac (this session):** writing code, the paper (Overleaf runs in the browser), literature notes, and plotting from small JSON logs. Optional smoke tests use a toy model (2 layers, d=64, a few hundred random tokens, under 1 minute on CPU) only to catch typos and shape bugs. They need no HF model downloads.
- **PC (3070):** every real experiment (A, B, C, benchmarks), all datasets and all model weights.
- **Bridge, in order of preference:**
  1. **Work directly inside WSL2 on the PC** with the same repo checked out there. Runs can then be launched, monitored and debugged on the GPU machine itself. For a PyTorch beginner on a 7-day clock this is the biggest time-saver, so set it up on Day 0.
  2. Enable SSH into the PC's WSL2 from the Mac (the same network or Tailscale). I run commands over `ssh` from here, with long jobs inside `tmux` so they survive disconnects.
  3. Fallback: a private GitHub repo. I write code here, you `git pull` on the PC, run `bash run_grid.sh`, and push `results/*.json` and the logs back. I read the logs here to debug and plot.
- Every script writes small JSON or CSV results, never big artifacts, to `results/`, so moving results between machines takes seconds. Checkpoints stay on the PC.
- Keep the PC awake (disable sleep/hibernate, and set Windows Update active hours) so overnight runs aren't killed.

Honest odds: a 7-day first project on an 8 GB GPU is a long shot for the main track (overall acceptance is roughly 30%, and reviewers will push on scale). The plan below maximizes the chance of a *complete, correct, honest* paper; the reviews alone are valuable, and the same work can go to an ICLR 2027 workshop (deadlines usually ~Feb) if rejected.

---

<ideas>

### Idea 1: Thin-gate GLU: factorize only the gate of SwiGLU (MLP block) — RECOMMENDED
- **Core idea.** A SwiGLU block computes `W_down( silu(W_gate x) ⊙ (W_up x) )` with three equal-size matrices. The gate only *selects* which hidden units fire; up/down carry the *content*. Replace `W_gate` (d×d_ff) by a rank-r product `B·A` (r ≪ d), keep up/down dense, and either bank the ~20-25% MLP parameter/FLOP saving or reinvest it in width.
- **Why it might work.** (a) Activation-sparsity work shows tiny low-rank predictors can forecast which MLP neurons fire, which suggests the gating decision lives in a low-dimensional subspace. (b) A March 2026 paper showed the analogous asymmetry in attention (selection via Q/K needs far fewer dims than value transfer). Whether the same holds for the MLP gate appears to be open.
- **Closest work to check.** (1) Low-rank/structured FFN training from scratch, which low-ranks *all* FFN matrices uniformly: arXiv 2406.16450 (NeurIPS 2024), 2407.09835, CoLA, LOST (2508.02668), NOBLE (2603.06492). (2) "Thin Keys, Full Values" (2603.04427), the conceptual sibling in attention. (3) SVD-compression papers with per-module rank allocation (ASVD, SVD-LLM, Zero-Sum SVD 2602.02848, SigmaScale 2606.07098); their appendices may already report gate/up/down sensitivity. (4) Contextual-sparsity predictors (Deja Vu line). My searches found **no paper that factorizes only the gate**, but four searches do not amount to a literature review. Search arXiv, Semantic Scholar and OpenReview (including ICLR 2026 submissions) for: "low-rank gate GLU", "factorized gate SwiGLU", "asymmetric GLU", "narrow gate feed-forward", and check "cited by" on 2406.16450.
- **Minimal experiments.**
  - **A. Training-free asymmetry study (cheap, broad).** On 6-8 open GLU models ≤1.7B (SmolLM2-135M/360M/1.7B, Qwen2.5-0.5B/1.5B, Llama-3.2-1B, TinyLlama-1.1B; OLMo-1B if time), SVD-truncate gate vs up vs down to the same rank (same shapes ⇒ same parameter savings) and measure WikiText-2 / FineWeb-val perplexity vs rank. If time allows, add an activation-aware (whitened) SVD variant.
  - **B. From-scratch pretraining (small, controlled).** GPT-style models (RMSNorm, RoPE, SwiGLU) at ~3 sizes (≈10M / 25M / stretch 85M non-embedding params) on a FineWeb-Edu subset, fixed token budget, 3 seeds at the two small sizes. Arms: dense SwiGLU; dense with smaller d_ff at matched params (the "just shrink it" baseline); thin-gate at r ∈ {d/2, d/4, d/8}; **controls** thin-up and thin-down at the same rank; all-three-low-rank at matched params; thin-gate with the savings reinvested in d_ff (iso-param).
  - **C. Conversion ("healing").** Truncate gates of a pretrained 135M-360M model, freeze everything else, train only the new factors for a few tens of millions of tokens (fits 8 GB because optimizer state is tiny), and compare with the same procedure applied to up/down.
  - **Metrics.** Val loss/perplexity (mean ± std over seeds), params, FLOPs, *measured* tokens/s and peak memory; a few lm-eval-harness zero-shot tasks for A and C.
- **Main risk.** The gate is *not* more compressible than up/down, or from-scratch differences fall inside seed noise. A useful partial result is still a clean asymmetry map of the three GLU matrices across models, whichever direction it points (if up or down is the cheap one, the method becomes "thin-up" or "thin-down"). An all-symmetric result is a negative finding suited to a workshop, not the main track.

### Idea 2: Shared memories, private gates — cross-layer sharing of up/down, per-layer low-rank gates (MLP block)
- **Core idea.** Adjacent layers share the big content matrices (`W_up`, `W_down`) while each layer keeps its own cheap low-rank gate, so each layer reads a shared "neuron bank" differently. This cuts MLP parameters ~2× per shared pair and reduces weight memory traffic at decode time.
- **Why.** It follows the key-value-memory view of FFNs, and adjacent layers are known to be similar. The per-layer gate restores the layer specificity that plain sharing loses.
- **Closest work.** ALBERT-style sharing, immediate block-wise sharing in on-device LLMs (the MobileLLM line), recursive transformers with per-layer LoRA deltas, Basis Sharing (2410.03765), merging FFN sublayers (2501.06126). This area is crowded, and the delta (share content, privatize selection) has to be checked carefully.
- **Experiments.** Same from-scratch harness as Idea 1, with arms {no sharing, full MLP sharing, shared + private gate, shared + per-layer LoRA}.
- **Risk.** The benefits of sharing show up mainly at scales you can't reach on a 3070, and it depends on Idea 1 being true. It is better kept as a follow-up paper or an ablation.

### Idea 3: Certified-exact LM-head pruning for greedy decoding (embeddings / output head)
- **Core idea.** In small LMs with 128k-262k vocabularies the output head is a large fraction of the weights read per decoded token. Compute a low-rank preview of the logits plus a precomputed per-token bound on the residual, then evaluate exact logits only for tokens whose upper bound beats the current best. The argmax is **provably identical** to full decoding, with no draft model and no verification pass.
- **Why.** Decoding is memory-bound, and skipping most head rows gives real wall-clock gains in plain PyTorch (row gather + small matmul). It is training-free and fits in 8 GB.
- **Closest work.** This area is crowded. Drafter-side vocabulary pruning for speculative decoding (FR-Spec 2502.14856, VocabTrim 2506.22694, SlimSpec 2605.10453, SpecVocab 2602.13836, NanoSpec 2605.26444), dynamic vocabulary selection for small LMs (VocabTailor 2508.15229), older SVD-softmax / learning-to-screen, and classical exact-MIPS bounds. The only fresh angle is the *exactness certificate* on the target model.
- **Risk.** Bounds are loose in ~1-2k dimensions, so little gets pruned, and the novelty is "classical MIPS applied to LLM heads". The partial result would be a characterization of how often certificates succeed across models.

### Killed ideas (from my novelty checks; don't spend time on these)
- Smaller Q/K dimension than V ("thin keys") is **published**, arXiv 2603.04427 (Mar 2026).
- Lossy LM-head shortlisting has at least six recent papers (listed under Idea 3).
- Frequency-aware embedding compression for small LMs is covered by GroupReduce (1806.06950), TensorGPT and TensorSLM (2506.13514).

</ideas>

<recommendation>

**Idea 1 (thin-gate GLU).** Reasons, given a 3070, 7 days and a PyTorch beginner:
1. **It survived the novelty check.** The other two attention/embedding candidates did not.
2. **Two of three experiments need no training.** Experiment A is about 100 lines with HuggingFace, runs in hours on 8 GB, and covers 6-8 public models, which partly offsets the "tiny scale" objection to Experiment B.
3. **The go/no-go is fast and cheap.** By day 3 you know from A plus one pair of B runs whether the asymmetry exists.
4. **It has built-in controls** (thin-up and thin-down at equal rank), so the claim is a falsifiable asymmetry and not just "our layer is good".
5. **The change is about 15 lines in one module**, which a newcomer can implement and fully understand. That matters because you are responsible for every claim.
6. **It fails gracefully**, because the asymmetry map is reportable either way.

Claim discipline: say "at ≤[P]M parameters and [T] tokens". Never extrapolate to LLM scale. Report measured speed even if small low-rank matmuls turn out to be no faster on a 3070.

</recommendation>

<abstract>

**Full version (fill in the brackets by Sep 25; delete any sentence whose result you don't have):**

Gated linear units (GLUs) are the standard feed-forward block in modern Transformers and spend three equally sized projections per block: a gate, an up-projection, and a down-projection. These play different roles: the gate selects which hidden units are active, while the up- and down-projections carry the content written back to the residual stream. Existing architectures and low-rank methods nevertheless allocate them identical capacity. We ask whether selection is cheaper than content. First, in a training-free study of [N] open pretrained language models ([A]M–[B]B parameters), we truncate each projection to the same rank at matched parameter savings and find that perplexity degrades [X]× more slowly for the gate than for the up- or down-projection. Motivated by this asymmetry, we propose the thin-gate GLU, which factorizes only the gate through a rank-r bottleneck and leaves the content path dense. In from-scratch language-model pretraining at [P1]–[P2]M parameters on [DATASET], thin-gate GLUs with r = d/[k] remove [Y]% of feed-forward parameters and FLOPs while changing validation loss by [Z] (mean over [S] seeds), whereas equally sized bottlenecks on the up- or down-projection cost [Z′]; reinvesting the savings in width at matched parameters yields [W]. Pretrained models can also be converted by truncating the gate and training only the new factors for [T]M tokens, recovering [V]% of the perplexity gap. We report measured throughput and memory, ablate the rank, and discuss the limited scale of our experiments.

**Sep 18 version (no results exist yet):** keep sentences 1-4; replace every result clause with what you *do*, e.g. "…we compare truncating each projection at matched parameter savings across open pretrained models", "…we evaluate thin-gate GLUs against dense and uniformly low-rank baselines in small-scale pretraining", "…and test whether pretrained models can be converted by training only the new factors." That version claims nothing unverified and is consistent with any outcome, including a pivot to thin-up or thin-down.

</abstract>

<titles>

1. Selection Is Cheap: Low-Rank Gates for Gated Feed-Forward Layers
2. Do GLU Gates Need Full Rank? An Asymmetry Study of Transformer Feed-Forward Blocks
3. Thin-Gate GLU: Factorizing Only the Gate in Transformer Feed-Forward Layers

For tomorrow's submission use #2. It is a question, so it stays accurate whatever the results are, and you can switch to #1 or #3 by Sep 25.

</titles>

<plan>

All clock times are EDT. The GPU should be running something every night, and you write while it runs.

**Day 0, Thu Sep 17 (today). Decide, register, set up.**
- 2 h novelty search (queries under Idea 1). Skim abstracts + method sections of 2406.16450 and 2603.04427. Keep a `related.md` with one line per paper; it becomes your Related Work.
- Settle the author list (it freezes tomorrow). Every author needs an OpenReview profile.
- On the 3070 PC: install **WSL2 + Ubuntu** (`torch.compile`/Triton are unreliable on native Windows), then the NVIDIA driver, `uv` or conda, PyTorch (CUDA build), `transformers`, `datasets`, `lm-eval`. Sanity check: `torch.cuda.is_available()` and a bf16 matmul.
- Create a **private** GitHub repo, cloned on both machines. Work in WSL2 on the PC (preferred) or enable SSH from the Mac; see "Two-machine workflow". The Mac is used only for editing, writing and toy smoke tests.
- Download the ICLR 2027 style zip. Create an Overleaf project from it and compile the unmodified template once.

**Day 1, Fri Sep 18. Submit the abstract; first numbers.**
- Submit title #2 + the numbers-free abstract on OpenReview **by Friday evening**, not at 07:00 Saturday. Choose the primary area, add keywords, fill in the AI-use field in the form.
- Run Experiment A on SmolLM2-135M first (minutes), then 360M and Qwen2.5-0.5B. "Setting up a baseline" means first reproducing the *untruncated* model's known perplexity with your own eval code. If that number is off, everything downstream is wrong.
- Get the from-scratch trainer running with the dense baseline at the smallest size. Measure tokens/s, then set the token budget so one run is ≤ ~1.5 h. Launch the dense baseline overnight.

**Day 2, Sat Sep 19.** Experiment A on all models. Thin-gate r = d/4 vs dense vs the shrunk-d_ff baseline, one seed each at the smallest size. Overnight: a second seed of the dense baseline. The gap between two seeds of the *same* config is your noise floor; any effect smaller than that is not an effect.

**Day 3, Sun Sep 20. GO / NO-GO (decide by 6 pm).**
- **GO** if (i) in most models gate truncation hurts clearly less than up/down at equal rank, **or** (ii) from scratch, thin-gate r = d/4 is within the seed noise of dense while thin-up/thin-down are not, or it beats the shrunk-d_ff baseline at matched params.
- **PIVOT** if the asymmetry exists but points at up or down: same paper, renamed method, title #2. The Sep 18 abstract stays truthful.
- **NO-GO** if everything is symmetric and low-rank simply loses. Options: withdraw (check the withdrawal and de-anonymization rules first) or retarget a workshop with the negative result. Don't stretch the claims to survive.

**Day 4-5, Mon-Tue Sep 21-22. Full grid.** Two sizes × 3 seeds × {dense, shrunk, thin-gate r/2, r/4, r/8, thin-up, thin-down, all-low-rank, iso-param reinvest} is ~40 short runs, so drop arms before dropping seeds. Run the 85M size with one seed only if throughput allows. Run Experiment C (healing) on SmolLM2-135M/360M, the latency/memory benchmark, and lm-eval zero-shot on A/C. While the GPU runs, write the Method, Experimental Setup and Related Work sections.

**Day 6, Wed Sep 23. Freeze experiments at noon.** Every figure and table is generated by a script from logged JSON, never hand-typed. Write the Results section. Every number in the text must trace to a log file.

**Day 7, Thu Sep 24.** Full draft: Intro (written last, from the results), Limitations (scale, single GPU, short training), the AI-use statement (say concretely that an LLM assistant helped with ideation, the literature search, code and drafting, and that the authors ran and verified all experiments), the Reproducibility statement, and an appendix with hyperparameters. Finalize the abstract and replace every bracket with a real logged number.

**Day 8, Fri Sep 25. Polish and submit by ~8 pm** (hard deadline Sat 07:59).
- ≤ 9 pages, template margins untouched.
- Anonymity: no names, no GitHub URLs, no acknowledgements. Strip PDF metadata. Include an anonymized code zip as supplementary material.
- Check every citation against the actual paper (title, authors, year). Never cite from memory or from an LLM.
- Re-read the abstract against the tables, because ICLR requires the two to match.

**What was built (this repository):**
- `posthoc_truncate.py`: Experiment A (HF model → SVD-truncate gate/up/down at rank r → perplexity → JSON).
- `model.py` / `train.py` / `data.py`: a minimal GPT (RMSNorm, RoPE, SwiGLU with `gate_rank`, `up_rank`, `down_rank`, `d_ff` options; bf16; SDPA flash attention; gradient accumulation for 8 GB). Data comes from pre-tokenized FineWeb shards (I'll verify the HF dataset id before relying on it).
- `heal.py` (Experiment C), `bench.py` (tokens/s, peak memory), `plot.py` (figures from JSON logs), `configs/` for every arm and `run_grid.sh`.
- `paper/`: the ICLR 2027 template pre-filled with the section skeleton, the AI-use and reproducibility sections, and `related.md`.
- Every script gets a `--smoke` flag (toy random-weight model, random tokens, no downloads) that finishes in <1 min on the Mac CPU. Real models and data are only ever touched on the PC.
- `SETUP_PC.md`: step-by-step setup for WSL2, the NVIDIA driver, the Python environment, `tmux`, and disabling sleep on the 3070 machine.

**Verification:** toy smoke tests pass on the Mac (shape and typo checks only); on the PC, (1) the untruncated perplexity from `posthoc_truncate.py` matches a plain HF evaluation of the same model, (2) thin-gate with r = full rank, initialized from an SVD of a dense gate, reproduces the dense forward pass to numerical tolerance (unit test), (3) parameter and FLOP counts printed by `model.py` match hand calculations for each arm, (4) two seeds of the dense baseline give the noise floor before any comparison is interpreted.

</plan>

## Sources (from my searches; read them yourself before citing)
- Thin Keys, Full Values: https://arxiv.org/pdf/2603.04427
- Structured feedforward layers (NeurIPS 2024): https://arxiv.org/abs/2406.16450
- Low-rank training scaling analysis: https://arxiv.org/html/2407.09835v1
- LOST: https://arxiv.org/pdf/2508.02668 · NOBLE: https://arxiv.org/pdf/2603.06492 · MoARa: https://arxiv.org/abs/2609.15037
- SigmaScale: https://arxiv.org/pdf/2606.07098 · Zero Sum SVD: https://arxiv.org/pdf/2602.02848
- Basis Sharing: https://arxiv.org/html/2410.03765v1 · Merging FF sublayers: https://arxiv.org/pdf/2501.06126
- FR-Spec: https://arxiv.org/pdf/2502.14856 · VocabTrim: https://arxiv.org/abs/2506.22694 · SlimSpec: https://arxiv.org/abs/2605.10453 · SpecVocab: https://arxiv.org/html/2602.13836 · NanoSpec: https://arxiv.org/html/2605.26444 · VocabTailor: https://arxiv.org/html/2508.15229
- GroupReduce: https://arxiv.org/pdf/1806.06950 · TensorSLM: https://arxiv.org/abs/2506.13514
- ICLR 2027: https://iclr.cc/Conferences/2027/AuthorGuidelines · https://blog.iclr.cc/2026/09/02/submission-policies-for-iclr-2027/ · https://iclr.cc/Conferences/2027/CallForPapers · https://iclr.cc/Conferences/2027/AIPolicyForAuthors
