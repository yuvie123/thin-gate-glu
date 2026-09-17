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

## Searches still to run (arXiv, Semantic Scholar, OpenReview incl. ICLR 2026 submissions, Google Scholar "cited by" on 2406.16450)
- "low-rank gate" GLU / SwiGLU
- "factorized gate" feed-forward
- "asymmetric GLU" / "narrow gate" / "thin gate"
- "gate projection" rank / compressibility / sensitivity
- "which FFN matrix" low-rank ablation

If you find a paper that factorizes only the gate: tell me immediately. We then either pivot the
framing (asymmetry study + conversion) or drop the novelty claim; do not ignore it.
