# CPAL 2027 abstract registration: what to enter, field by field

Target: **CPAL 2027, Proceedings Track** (OpenReview). **Abstract registration Nov 23, 2026; paper Dec 5,
2026** (aim to upload Dec 3). Verified on cpal.cc on 2026-09-19: 9 pages main text plus unlimited
references and appendix, double-blind, archival PMLR proceedings, arXiv preprint allowed, AI tools allowed
with authors fully responsible. **No reciprocal-reviewing requirement was found**, unlike ICLR and AISTATS.

The CPAL OpenReview form was not live when this was written (links appear on cpal.cc/openreview/ "as they
become active"). The fields below are the ones every OpenReview venue has; add any CPAL-specific ones
(subject area, code of conduct, AI-use tick boxes) to this file when the form is visible, and check
whether the abstract can still change between Nov 23 and Dec 5.

The ICLR 2027 version of this checklist (Sep 2026) was never filed; its content is kept where it still applies.

## Title *
```
Do GLU Gates Need Full Rank? An Asymmetry Study of Transformer Feed-Forward Blocks
```

## Authors *
Sole author. Make sure the OpenReview profile is complete (affiliation, email) before registering.

## Keywords *
```
gated linear units, SwiGLU, feed-forward layers, low-rank factorization, efficient Transformers, language models, parameter efficiency, singular value decomposition, model compression
```

## TL;DR (optional, but fill it in)
```
GLU feed-forward blocks give their gate, up- and down-projections equal size; using equal-rank controls, we test whether the gate, which only selects, can be much lower rank than the projections that carry content.
```

## Abstract *
Paste as one paragraph. It contains no TeX and no numbers. It matches `paper/main.tex` word for word.
```
Gated linear units (GLUs) are the standard feed-forward block of modern Transformer language models and hold most of each block's parameters in three equally sized projections: a gate, an up-projection, and a down-projection. These are commonly described as doing different jobs: the gate selects which hidden units are active, while the up- and down-projections carry the content written back to the residual stream. Standard architectures nevertheless give all three the same capacity. We ask whether selection is cheaper than content. Because the three matrices share a shape, the question admits a controlled test: restrict one projection at a time to the same rank, so that parameter savings are identical, and measure which restriction the model tolerates best. First, in open pretrained language models and without any training, we truncate each projection with plain and activation-aware singular value decompositions and compare perplexity. Second, in small-scale pretraining from scratch with multiple seeds, we compare the thin-gate GLU, which factorizes only the gate and keeps the content path dense, against dense, width-reduced, and uniformly low-rank baselines at matched parameter counts, and against the symmetric controls that factorize only the up- or only the down-projection. Third, we test whether pretrained models can be converted by truncating one projection and training only the new factors. We report measured throughput and memory alongside parameter and FLOP counts, and state the scale limits of our experiments.
```
By Nov 23 the Exp. A numbers exist and most of Exp. B will; the registration abstract may still be this
numbers-free version, and the Dec 5 version replaces each "we compare"/"we test" clause with the logged
result (see the bracketed full version in `PLAN.md`). Delete the sentence for any experiment that did not
happen. Never register a placeholder abstract.

## PDF
Required Dec 5, probably optional Nov 23 (confirm on the form). Never upload a draft with `\todo`s in it.

## Supplementary Material
Anonymized code zip (no names, usernames, URLs, machine names) with the Dec 5 upload.

## Primary / subject area
CPAL lists "sparsity, structured sparsity, low rank" and "interpretable and structured neural
architectures" among its topics; pick the low-rank one if the form has such a field.

## Anonymity
Double-blind: no names, affiliations, acknowledgements or repo links in the PDF, the code zip or the
PDF metadata. A public arXiv preprint is allowed by CPAL's policy but is not planned.

## AI assistance
If the form has AI-use tick boxes, tick the ones matching the AI use statement in `paper/main.tex`
(writing polish, retrieval and discovery, research ideation and execution, drafting sections). If you
change one, change the other.

## Then
Press **Submit**, then open the submission from the OpenReview author console and check that the title,
abstract and every field saved as intended.
