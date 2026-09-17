# Related work reading list

One line per paper once you've read it: *what they do / how we differ*. This file becomes Section 2.
The links came from web searches on 2026-09-17. **Verify titles, authors and venues on the page itself,
and read each paper before citing it.**

## Must read first (decides novelty)

`[id verified]` means the arXiv id, title, authors and venue were checked against the arXiv page on
2026-09-17. It does NOT mean the paper has been read: the `[ ]` box is what tracks reading, and the
per-matrix ablation question below can only be answered from the methods and appendix, not the abstract.

- [ ] **arXiv 2406.16450** `[id verified]` -- "Building on Efficient Foundations: Effectively Training
  LLMs with Structured Feedforward Layers", Wei, Moalla, Pascanu, Gulcehre. **NeurIPS 2024** (cite the
  proceedings version, not the preprint). Three structured low-rank / block-diagonal parameterizations of
  the FFN, trained from scratch up to 1.3B, plus a self-guided training scheme. Abstract shows the
  structure applied to the FFN as a whole with no gate/up/down split. *Still to check in the paper: is
  there any per-matrix ablation isolating one of gate, up or down?*
- [ ] **arXiv 2407.09835** `[id verified]` -- "Investigating Low-Rank Training in Transformer Language
  Models: Efficiency and Scaling Analysis", Wei, Moalla, Pascanu, Gulcehre. **ICML 2024 workshop** (Next
  Generation of Sequence Modeling Architectures), so a workshop paper, not main-conference. Low-rank FFN
  training to 1.3B on RefinedWeb; reports 2.6x FFN speedup at 32% of the parameters. Same authors as
  2406.16450, so read these two together. *Same per-matrix check.*
- [ ] **arXiv 2603.04427** `[id verified]` -- "Thin Keys, Full Values: Reducing KV Cache via
  Low-Dimensional Attention Selection", Yao, Chen, Murtadha, Wang (Feb 2026, rev. Mar 2026; no venue
  listed). SVD-factorizes the KEY projection to shrink the KV cache while values stay full-dimensional,
  on the argument that "queries and keys produce scalar attention weights (selection), while values carry
  rich representations (value transfer)". **Confirmed to be about attention only -- it does not touch
  MLP/GLU layers.** Our conceptual sibling: cite it prominently, state that it motivated our question,
  and be explicit that we ask it of the MLP gate instead.
- [ ] **arXiv 2609.15037** `[id verified]` -- "MoARa: Module-Aware Rank Allocation and Structure-Preserving
  Decomposition for Low-Rank LLM Pre-training", Kim, Kwak (14 Sep 2026). **EMNLP 2026 Main Conference.**
  Allocates low-rank *gradient* projection ranks per module by sensitivity instead of uniformly, with
  block-wise magnitude-direction decomposition. Abstract reports module-level sensitivity but no
  within-GLU comparison of gate vs up vs down, and it does not factorize only the gate. *Still to check:
  does any table or appendix break sensitivity down to the gate/up/down level?*

**Novelty status after id verification (not after reading):** none of these four factorizes only the gate.
The closest, 2603.04427, makes the same selection-vs-content argument but in attention. This does not
establish novelty -- the per-matrix ablations in 2406.16450 and 2609.15037 could still contain the result,
and the searches at the bottom of this file are still unrun.

## Low-rank pretraining (others)
- [ ] arXiv 2508.02668: LOST (low-rank + sparse pretraining)
- [ ] arXiv 2603.06492: NOBLE (nonlinear low-rank branches)
- [ ] CoLA (bottleneck low-rank with nonlinearity): find the arXiv id

## Post-hoc SVD compression (Experiment A baselines; look in their appendices for gate/up/down sensitivity)
- [ ] ASVD, SVD-LLM (activation-aware / whitened SVD): find the ids; we use the whitening trick
- [ ] arXiv 2602.02848: Zero Sum SVD
- [ ] arXiv 2606.07098: SigmaScale
- [ ] arXiv 2510.05544: activation-informed Pareto-guided low-rank compression

## Motivation for "selection is cheap"
- [ ] Contextual sparsity with low-rank predictors of active MLP neurons (Deja Vu and follow-ups): find the ids
- [ ] GLU variants paper (origin of SwiGLU): find the id

## Found by the searches of 2026-09-17 (ids verified on the arXiv page; NONE read yet)

Same convention as above: `[id verified]` = id, title, authors and venue checked on the page; `[ ]` = unread.
What is said about each paper below comes from its abstract or from one targeted look at its HTML, so
treat it as a pointer to where to read, not as a finding you can cite.

**Closest neighbours (read right after the four must-reads):**
- [ ] **arXiv 2606.00573** `[id verified]` -- "LASER: Loss-Aware Singular-value Decomposition and Rank
  Allocation for Efficient Low-Precision Vision-Language Models", Wang, Wang, Li, Ren, Zhang (30 May 2026,
  no venue listed). Post-hoc compression of VLMs. **Its FFN scheme is asymmetric: Sec. 3.5 / Algorithm 1
  apply SVD to the gate AND up projections together, on the same selected hidden channels, keep the
  remaining channels dense, and leave the down projection dense (quantized).** So it is input-side vs
  output-side, not gate vs content, and it is not gate-only. No ablation comparing gate vs up vs down was
  found. *Read Sec. 3.5: why do they spare down? That reasoning is the nearest thing to ours.*
- [ ] **arXiv 2510.09332** `[id verified]` -- "FLRC: Fine-grained Low-Rank Compressor for Efficient LLM
  Inference", Lu, Chen, Chang, Hu, Wu. **EMNLP 2025.** Fisher-based per-projection rank allocation.
  **Appendix A, Figure 2 plots importance scores for every projection type in Llama-3-8B by layer**; the
  caption names `down_proj` as high-importance (compress less). The text gives no gate-vs-up ordering and
  no functional explanation, but the figure may show one. *Look at that figure: where do the gate_proj
  points sit relative to up_proj? This is prior per-matrix evidence and must be cited next to Exp. A
  whichever way it points.*
- [ ] **arXiv 2506.23225** `[id verified]` -- "Masked Gated Linear Unit", Tajima, Inoue, Sekikawa, Sato,
  Yokota (29 Jun 2025, no venue listed). Removes the separate gate matrix altogether: gate and value
  streams share ONE weight matrix through learned binary masks (MoEG), to cut memory reads. No low-rank
  gate. A different answer to the same question ("does the gate deserve a full matrix of its own?"), so
  it belongs in Related Work and possibly as a baseline to discuss.

**SVD-compression papers surfaced by the same searches (ids NOT yet verified; check appendices for
per-projection results):** arXiv 2605.15626 (IO-SVD), arXiv 2602.03051 (SAES-SVD), arXiv 2601.07839
(hierarchical sparse plus low-rank), arXiv 2505.21732 (LaX, low-rank training).

## Searches still to run (arXiv, Semantic Scholar, OpenReview incl. ICLR 2026 submissions, Google Scholar "cited by" on 2406.16450)
- [ran 2026-09-17, web search over arXiv] "low-rank gate" GLU / SwiGLU -- no gate-only paper found
- [ran 2026-09-17, web search over arXiv] "factorized gate" feed-forward -- no gate-only paper found
- [ran 2026-09-17, web search] "asymmetric GLU" / "narrow gate" / "thin gate" -- nothing relevant at all
- "gate projection" rank / compressibility / sensitivity
- "which FFN matrix" low-rank ablation

If you find a paper that factorizes only the gate: tell me immediately. We then either pivot the
framing (asymmetry study + conversion) or drop the novelty claim; do not ignore it.
