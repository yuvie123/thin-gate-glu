# Related work reading list

One line per paper once you've read it: *what they do / how we differ*. This file becomes Section 2.
The links came from web searches on 2026-09-17. **Verify titles, authors and venues on the page itself,
and read each paper before citing it.**

## READ THIS ONE FIRST: prior work already reports the asymmetry (found 2026-09-17, evening)

- [ ] **arXiv 2407.11239** `[id verified; key passages checked in the HTML full text, v2]` -- WeLore, "From Low
  Rank Gradient Subspace Stabilization to Low-Rank Weights: Observations, Theories, and Applications",
  Jaiswal, Wang, Yin, Liu, Chen, Zhao, Grama, Tian, Wang. **ICML 2025.**
  **It already says the gate is the low-rank one.** Figure 1 caption (LLaMA2-7B), verbatim: "mlp.up_proj,
  mlp.down_proj, and self_attn.v_proj exhibit less pronounced Hessian gaps compared to self_attn.k_proj,
  self_attn.q_proj, self_attn.o_proj, and mlp.gate_proj ... components with a pronounced Hessian gap ...
  tend to be more low-rank". Its Sec. 2.4 has a paragraph titled "MLP Gate Projections" and explains the
  effect by the activation concentrating gradients in a few directions. Its method then treats `gate_proj`
  (with q, k, o) as low-rank components and `up_proj`, `down_proj`, `v_proj` as not. It even has our
  attention analogy: q/k low-rank, v dense. Models: LLaMA-130M, LLaMA-2 7B and 13B, Mistral-7B.
  **What it does NOT do** (checked, "not found" in the full text): compress one projection type at a time
  at equal rank and compare perplexity; train anything from scratch with a low-rank gate; use controls.
  Everything is post-hoc compression and fine-tuning of pretrained checkpoints.
  **Consequence for us:** "the gate is more compressible than up and down in pretrained models" is NOT our
  discovery. Exp. A becomes a controlled test of a published observation (one projection at a time, equal
  rank, whitened SVD, several small models). The part nobody seems to have done is Exp. B: building the
  gate thin from the start, against thin-up and thin-down controls. The introduction must open from
  WeLore, not around it. *Read Sec. 2.1, 2.4, 3.1 and Figures 1, 3 and 7 before writing a word of the intro.*

## Must read first (decides novelty)

`[id verified]` means the arXiv id, title, authors and venue were checked against the arXiv page on
2026-09-17. It does NOT mean the paper has been read: the `[ ]` box is what tracks reading, and the
per-matrix ablation question below can only be answered from the methods and appendix, not the abstract.

- [ ] **arXiv 2406.16450** `[id verified]` -- "Building on Efficient Foundations: Effectively Training
  LLMs with Structured Feedforward Layers", Wei, Moalla, Pascanu, Gulcehre. **NeurIPS 2024** (cite the
  proceedings version, not the preprint). Three structured low-rank / block-diagonal parameterizations of
  the FFN, trained from scratch up to 1.3B, plus a self-guided training scheme. **Checked in the HTML full text: their
  FFN is the 2-matrix GELU kind (Sec. 4.1: "two linear layers and a GeLU activation"), so there is no
  gate in their models at all**, and no per-matrix ablation or sensitivity comparison was found. They are the
  uniform-low-rank baseline for NON-gated FFNs; say exactly that. *Read Sec. 2.1, 2.3 and Appendix B.*
- [ ] **arXiv 2407.09835** `[id verified]` -- "Investigating Low-Rank Training in Transformer Language
  Models: Efficiency and Scaling Analysis", Wei, Moalla, Pascanu, Gulcehre. **ICML 2024 workshop** (Next
  Generation of Sequence Modeling Architectures), so a workshop paper, not main-conference. Low-rank FFN
  training to 1.3B on RefinedWeb; reports 2.6x FFN speedup at 32% of the parameters. Same authors as
  2406.16450, so read these two together. **Checked: also a 2-matrix GELU FFN (Sec. 3.1), no gate, no
  per-matrix ablation; they keep the first FFN dense.**
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
  within-GLU comparison of gate vs up vs down, and it does not factorize only the gate. **Checked in the HTML full text:
  this is about low-rank GRADIENT projection (optimizer state), not low-rank weights. Appendix C.3 / Table 6
  gives an ordering only: `mlp.down` least sensitive by a wide margin, `mlp.up` and `mlp.gate` in the
  middle, no numbers separating gate from up. It calls attn q/k the "routing pair" and v/o "content",
  language close to ours.** Not gate-only, not weights; cite as related evidence on per-module sensitivity.

**Novelty status after id verification (not after reading):** none of these four factorizes only the gate.
The closest, 2603.04427, makes the same selection-vs-content argument but in attention. This does not
establish novelty -- the per-matrix ablations in 2406.16450 and 2609.15037 could still contain the result.

**Update after the five searches (2026-09-17, evening): still no paper that factorizes only the gate, and
the claim now needs narrowing.** Two findings change what we may say. (1) LASER (2606.00573) already treats
the FFN asymmetrically: gate and up low-rank together, down dense. So "existing methods allocate the three
projections identical capacity" is too strong as written in the abstract and introduction; the accurate
version is that nobody separates the GATE from the two content projections. (2) FLRC (2510.09332) and
WeLore (2407.11239) publish per-projection importance or rank profiles, so per-matrix sensitivity is not
new in itself; what would be new is the controlled equal-rank comparison, the selection-vs-content
reading of it, and the from-scratch architecture. None of this is settled until those figures are read.

## Novelty search of 2026-09-24 for the screen-2 win and the screen-3 candidates (abstracts read online by
## the assistant; the author has read NONE of these yet; cite only after reading)
- **arXiv 2109.08668, So et al. 2021, "Primer".** MUST READ. States verbatim: "Squared ReLU does have
  significant overlap with ReGLU and in fact is equivalent when ReGLU's U and V weight matrices are the same
  and squared ReLU is immediately preceded by a linear transformation with weight matrix U", and "squared
  ReLUs capture the benefits of these GLU variants, while being simpler, without additional parameters, and
  delivering better quality" (110M, C4, 525K steps: squared ReLU beats ReGLU and SwiGLU). **Our relu-tied
  gate is this; the screen-2 win is a reproduction at 28M, not a new result.**
- **arXiv 2411.13010, Huang and Schlag 2024/25, "Deriving Activation Functions Using Integration" (xIELU).**
  Repeats the equivalence ("equivalent to ReGLU when the U and V weight matrices are identical") and, at 1.1B
  and 3B Llama on 125B FineWeb-Edu tokens, reports SwiGLU 2.353 > ReLU^2 2.337 > xIELU 2.323 (loss). So the
  ordering we see at 28M holds at 1.1B, and a trainable activation beats ReLU^2 there.
- **arXiv 2506.23225, Tajima et al. 2025, "Masked Gated Linear Unit" (MGLU).** One shared weight matrix
  produces both gate and value streams through learned element-wise binary masks ("mixture of element-wise
  gating"); SwiMGLU matches or beats SwiGLU with 47% less memory. Reports that naive weight sharing in SwiGLU
  degrades perplexity (23.6 -> 27.0), which matches our silu-tie kill. **Our partner-gated units are a fixed
  pairing mask on a shared matrix, a special case of this family; dropped as a contribution.**
- arXiv 2002.05202, Shazeer 2020, "GLU Variants Improve Transformer". GLU at 2/3 width vs plain FFN at
  equal parameters; the baseline result every GLU paper cites.
- arXiv 1710.05941, Ramachandran et al. 2017, "Searching for Activation Functions". Swish as self-gating.
- arXiv 2405.20768, Huang 2024, "Expanded Gating Ranges Improve Activation Functions". Per-block trainable
  scalars on self-gated activations; related in spirit to the per-unit scale in thin+tied.
- arXiv 2605.03667, 2026, "ELAS". Low-rank pretraining with squared ReLU and 2:4 activation sparsity; close
  in ingredients (low rank + ReLU^2) but about sparsity, not gate structure. Read to be sure.
- arXiv 2605.26647, 2026, "More Expressive Feedforward Layers: Token-Adaptive Mixing of Activations" (MoA);
  arXiv 2603.13347, 2026, "PolyGLU"; arXiv 2608.07323, 2026, "MemGLU". Recent gate/activation designs for
  the FFN at 9M-600M; none tie the gate, add a low-rank term to a self-gate, or pair units, from the
  abstracts. Read MoA and PolyGLU for the related-work paragraph.
- **Not found anywhere (searched: low-rank gate + self term, input-dependent / low-rank threshold for
  ReLU^2, GLU low-rank gate plus identity):** context-thresholded self-gating, h = relu(z + BAx) z. This is
  the candidate contribution. The author must repeat the search on Semantic Scholar and OpenReview before
  the paper calls it new.

## Low-rank pretraining (others)
- [ ] arXiv 2508.02668: LOST (low-rank + sparse pretraining)
- [ ] arXiv 2603.06492: NOBLE (nonlinear low-rank branches)
- [ ] **arXiv 2502.10940** `[id verified]` -- "CoLA: Compute-Efficient Pre-Training of LLMs via Low-Rank
  Activation", Liu, Zhang, Wang, Yan, Yang, Hovland, Nicolae, Cappello, Tang, Zhang (Feb 2025). The page
  says only "camera-ready"; **find the venue on the page before citing.** Replaces full-size MLPs and
  attention projections with low-rank auto-encoders; the abstract does not separate gate/up/down.
- [ ] **arXiv 2602.12429** `[id verified]` -- "Stabilizing Native Low-Rank LLM Pretraining" (Spectron),
  Janson, Oyallon, Belilovsky. **ICML 2026.** Low-rank factors for ALL non-embedding matrices, with a
  spectral renormalization against loss spikes. Uniform, no per-matrix split in the abstract. *Relevant to
  Exp. B twice over: as the uniform baseline, and because it says low-rank pretraining can be unstable.*

## Post-hoc SVD compression (Experiment A baselines; look in their appendices for gate/up/down sensitivity)
- [ ] **arXiv 2312.05821** `[id verified]` -- "ASVD: Activation-aware Singular Value Decomposition for
  Compressing Large Language Models", Yuan, Shang, Song, Yang, Wu, Yan, Sun (Dec 2023; no venue on the page).
- [ ] **arXiv 2403.07378** `[id verified]` -- "SVD-LLM: Truncation-aware Singular Value Decomposition for
  Large Language Model Compression", Wang, Zheng, Wan, Zhang. **ICLR 2025.** Source of the truncation-aware
  whitening that `posthoc_truncate.py --whiten` follows; *confirm our Cholesky construction matches theirs.*
- [ ] arXiv 2602.02848: Zero Sum SVD
- [ ] arXiv 2606.07098: SigmaScale
- [ ] arXiv 2510.05544: activation-informed Pareto-guided low-rank compression

## Motivation for "selection is cheap"
- [ ] **arXiv 2310.17157** `[id verified]` -- "Deja Vu: Contextual Sparsity for Efficient LLMs at Inference
  Time", Liu, Wang, Dao, Zhou, Yuan, Song, Shrivastava, Zhang, Tian, Re, Chen. **ICML 2023.** Low-cost
  predictors of which MLP neurons and attention heads are active. Follow-ups: still to find.
- [ ] **arXiv 2002.05202** `[id verified]` -- "GLU Variants Improve Transformer", Shazeer (Feb 2020; arXiv
  only, no venue on the page).

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

**Further neighbours (second batch of searches):**
- [ ] **arXiv 2407.11239** `[id verified]` -- "From Low Rank Gradient Subspace Stabilization to Low-Rank
  Weights: Observations, Theories, and Applications" (WeLore), Jaiswal, Wang, Yin, Liu, Chen, Zhao, Grama,
  Tian, Wang. **ICML 2025.** Abstract: "different LLM components exhibit varying levels of converged
  low-rank structures, necessitating variable rank reduction across them". It does not name the gate.
  **Checked: yes, and decisively. See the entry at the very top of this file.**
- [ ] **arXiv 2606.31717** `[id verified]` -- "Nonlinearity-Aware LoRA: Structured Gate Adaptation under
  Low-Rank Constraints", Yuan, Cai, Chen, Zheng, Xiao, Onizuka, Mao (30 Jun 2026; page says under review).
  Fine-tuning ADAPTERS on the gate of gated FFNs, not a low-rank replacement of the gate itself. Relevant
  because it argues a low-rank update to the gate changes the nonlinear selection, not just the features.
- [ ] OpenReview `QSoc7HGc6Q` **[NOT verified: OpenReview's browser check blocks automated reading]** --
  search results titled it "Low Rank Experts Enable Specialization in Dense ..." and quoted it as attaching
  its low-rank experts ONLY to the up-projection because the down-projection gave a weaker trade-off. An
  asymmetric choice among FFN matrices, so open it in a browser and check title, status and that claim.

**Surfaced by the same searches, ids NOT yet verified (check appendices for per-projection results):**
arXiv 2605.15626 (IO-SVD), arXiv 2602.03051 (SAES-SVD), arXiv 2601.07839 (hierarchical sparse plus
low-rank), arXiv 2505.21732 (LaX, low-rank training), arXiv 2406.02214 (SLTrain), arXiv 2505.12781
(Low-Rank Clone; its ablation drops FFN gate / up / down loss terms separately), arXiv 2505.17936
(gated neurons and their input-output functionality), arXiv 2606.22172 (gated MLPs as rank-1 bilinear
attention).

## Searches still to run (arXiv, Semantic Scholar, OpenReview incl. ICLR 2026 submissions, Google Scholar "cited by" on 2406.16450)
- [ran 2026-09-17, web search over arXiv] "low-rank gate" GLU / SwiGLU -- no gate-only paper found
- [ran 2026-09-17, web search over arXiv] "factorized gate" feed-forward -- no gate-only paper found
- [ran 2026-09-17, web search] "asymmetric GLU" / "narrow gate" / "thin gate" -- nothing relevant at all
- [ran 2026-09-17, web search over arXiv] "gate projection" rank / compressibility / sensitivity -- found FLRC,
  WeLore and LASER above; no gate-only paper
- [ran 2026-09-17, web search over arXiv] "which FFN matrix" low-rank ablation -- found MoARa (already listed),
  Low-Rank Clone and the low-rank-experts submission above; no gate-only paper

**Coverage, stated honestly.** These were general web searches, which index arXiv well and OpenReview badly.
Still NOT done, and only doable in a browser: (1) OpenReview's own search over ICLR 2026 submissions, which
blocks automated reading; (2) Google Scholar "cited by" on 2406.16450. A Semantic Scholar citation lookup
returned only 4 citing papers for 2406.16450 and 2 for 2603.04427, none about GLU gates or FFN structure; 4 is
implausibly few for a NeurIPS 2024 paper, so treat that list as incomplete, not as a clean bill.

If you find a paper that factorizes only the gate: tell me immediately. We then either pivot the
framing (asymmetry study + conversion) or drop the novelty claim; do not ignore it.
