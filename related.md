# Related work reading list

One line per paper once you've read it: *what they do / how we differ*. This file becomes Section 2.
The links came from web searches on 2026-09-17. **Verify titles, authors and venues on the page itself,
and read each paper before citing it.**

## Must read first (decides novelty)
- [ ] arXiv 2406.16450: structured (low-rank / block) feed-forward layers trained from scratch, NeurIPS 2024. They factorize all FFN matrices. *Check: do they ever factorize only one of gate/up/down? Any per-matrix ablation?*
- [ ] arXiv 2407.09835: low-rank training of Transformer LMs, efficiency and scaling analysis. *Same check.*
- [ ] arXiv 2603.04427: "Thin Keys, Full Values". Selection needs fewer dims than value transfer, in attention. Our conceptual sibling; we must cite it and say clearly that we study the MLP gate instead.
- [ ] arXiv 2609.15037: MoARa, module-aware rank allocation for low-rank gradient projection (Sept 2026). *Check whether their sensitivity profile already ranks gate vs up vs down.*

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
