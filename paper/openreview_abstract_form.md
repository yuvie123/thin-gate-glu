# OpenReview abstract form: what to enter, field by field

Deadline shown on the form: **Sat Sep 19, 2026, 7:59 AM** (your local time; = Sep 18 AOE).
Aim for Friday evening. After submitting you can still edit the title, abstract, keywords and TL;DR
until the full-paper deadline (Sep 25 AOE). Three things can NOT be changed after Saturday 7:59 AM:
the author list, the Reciprocal Reviewing Author, and the Reciprocal Reviewing Exemption.

Fields are in the order they appear on the form. `*` = required.

## Title *
```
Do GLU Gates Need Full Rank? An Asymmetry Study of Transformer Feed-Forward Blocks
```

## Authors *
Your own profile is already there with one institution. Nothing to type.
**Last chance to add anyone.** If a professor, TA or friend might end up as a co-author, they must be
added before the deadline and need their own OpenReview profile. If you are the only author, leave it.

## Keywords *
```
gated linear units, SwiGLU, low-rank factorization, feed-forward layers, Transformer language models, parameter efficiency, singular value decomposition, model compression
```

## TL;DR (optional, but fill it in)
```
We ask whether the gate of a GLU feed-forward block needs less rank than its up- and down-projections, by compressing each one to the same rank and comparing the damage.
```

## Abstract *
Paste as one paragraph. It contains no TeX and no numbers. It matches `paper/main.tex` word for word.
```
Gated linear units (GLUs) are the standard feed-forward block in modern Transformers and spend three equally sized projections per block: a gate, an up-projection, and a down-projection. These play different roles: the gate selects which hidden units are active, while the up- and down-projections carry the content written back to the residual stream. Standard architectures nevertheless give all three the same capacity. We ask whether selection is cheaper than content. First, in a training-free study of open pretrained language models, we compare truncating each projection to the same rank at matched parameter savings, using both plain and activation-aware singular value decompositions. Second, we study the thin-gate GLU, which factorizes only the gate through a low-rank bottleneck and leaves the content path dense, and evaluate it against dense, width-reduced, and uniformly low-rank baselines, as well as the symmetric controls that factorize only the up- or only the down-projection, in small-scale language-model pretraining with multiple seeds. Third, we test whether pretrained models can be converted by truncating one projection and training only the new factors. We report measured throughput and memory alongside parameter and FLOP counts, and discuss the limited scale of our experiments.
```
Every sentence from "Second" onwards is a promise about an experiment that needs a GPU. If one of
them does not happen, delete its sentence before Sep 25. The guidelines allow that ("titles and
abstracts may be edited before the submission deadline"); what they forbid is a placeholder abstract,
and this is not one.

## PDF -- leave empty
Not required at the abstract deadline. Do not upload the skeleton with TODOs in it.

## Supplementary Material -- leave empty

## Primary Area *
The dropdown was not in the screenshots, so pick the FIRST of these that exists in the list:
1. anything naming **foundation models / frontier models / LLMs**
2. **representation learning for computer vision, audio, language, and other modalities**
3. **general machine learning (i.e., none of the above)**

Not "infrastructure, software libraries, hardware": the paper is about model architecture, not systems.
This field only steers which reviewers see the paper.

## Code Of Ethics * -- tick
Read https://iclr.cc/public/CodeOfEthics first; ticking says you did. The part that bites for this
project: no fabricated or unsupported results, and honest AI disclosure.

## Paper Visibility * -- tick, after reading this
You are agreeing that once **reviewing begins** the paper is public, it is de-anonymized at the end
even if rejected, and it cannot be deleted. Your exit stays open until then: the guidelines say a
paper withdrawn **before the paper submission deadline is deleted** from OpenReview. So registering
on Saturday commits you to nothing; the real point of no return is the Sep 25 upload.

## Submission Requirements * -- tick

## Reciprocal Reviewing Author *  (LOCKED after the deadline)
Select **yourself**. The form says: "If the submission has no eligible reciprocal reviewer, simply
enter any author in this field."
Optional 1-minute check first: open the Reciprocal_Reviewing_Check link on the form and confirm it
says you are not eligible.

## Recent Qualifying Paper -- leave UNTICKED

## Reciprocal Reviewing Exemption *  (LOCKED after the deadline)
Choose the option that says no author is qualified to review / first-time authors. Do NOT choose
"We do not need an exemption": with no registered reviewer and no exemption the paper can be
desk-rejected. The guidelines back this: "If none of the authors are qualified under this definition,
then they are exempt from this requirement", capped at one submission per author.

## Reciprocal Reviewing Exemption Reason  (LOCKED after the deadline)
Leave empty if the dropdown had a first-time-author option. Only if it did not, choose its "other"
option and paste:
```
The sole author is an undergraduate student and a first-time author with no prior publications, and therefore does not meet the reviewer qualification criteria.
```

## AI Assistance * -- tick exactly these four
- [x] Yes, to aid or polish writing.
- [x] Yes, for retrieval and discovery (e.g., finding related work).
- [x] Yes, for research ideation or execution.
- [x] Yes, to draft sections of the paper.
- [ ] No, not at all.
- [ ] Yes, for generating synthetic datasets.
- [ ] Yes, for proving mathematical claims.
- [ ] Yes, but for none of the above purposes.

These four match the AI use statement in `paper/main.tex`. If you change one, change the other.

## Ready For LLM Feedback -- leave UNTICKED
It needs a PDF and spends your single feedback voucher.

## License *
```
CC BY 4.0
```

## Then
Press **Submit**. Afterwards, open the submission from your OpenReview author console and check that
the title, abstract, both reciprocal-reviewing fields and the four AI boxes saved as intended. Fix
anything wrong before Saturday 7:59 AM.
