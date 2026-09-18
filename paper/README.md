# Paper

**Retrieval Succeeds, Reasoning Fails: Disentangling Bottlenecks in Multi-Hop Knowledge Graph Question Answering with Small Language Models**

A short paper based on this repository's benchmark results, written up in standard two-column academic format. [Read the compiled PDF](paper.pdf).

## Before you submit or share this anywhere

- **Verify every citation.** The reference list (Lewis et al. 2020 on RAG, Edge et al. 2024 on GraphRAG, Zhang et al. 2018 on MetaQA, Yang et al. 2018 on HotpotQA, Yao et al. 2023 on ReAct, Wei et al. 2022 on chain-of-thought, Chung et al. 2022 on FLAN-T5, Reimers & Gurevych 2019 on Sentence-BERT) is written from best recollection of well-known papers. The titles, authors, and venues should be correct, but double-check each one on Google Scholar or Semantic Scholar before submitting anywhere - citation accuracy matters and this was not verified against a live database.
- **This is a preprint-quality writeup, not a peer-reviewed result.** It's honest, it's based on real experiments with real numbers (nothing in it is fabricated), and it's well-suited for a portfolio, a Master's application writing sample, or an arXiv preprint. It has not been through peer review and should not be described as "published in" any venue unless it actually goes through one.

## If you want to post it on arXiv

1. First-time submitters typically need an **endorsement** from an existing arXiv author in the target category (likely `cs.CL` or `cs.AI`) - check [arxiv.org/help/endorsement](https://arxiv.org/help/endorsement) for the current process. This is a real hurdle that isn't automatic.
2. Upload `paper.tex`, `accuracy_by_hop.pdf`, and the bibliography (already inline in this version - no separate `.bib` file needed) as a single submission.
3. arXiv compiles LaTeX server-side, so no local LaTeX install is required for the submission itself - but compiling it yourself first (see below) is the only way to catch errors before you submit.

## Recompiling locally

Requires a LaTeX distribution (e.g., [MiKTeX](https://miktex.org/) on Windows, [TeX Live](https://tug.org/texlive/) on Linux/macOS) or use [Overleaf](https://overleaf.com) (free, no install, upload `paper.tex` and `accuracy_by_hop.pdf` directly).

```bash
pdflatex paper.tex
pdflatex paper.tex   # run twice - the first pass resolves citations/references, the second renders them
```

## Regenerating the figure

`accuracy_by_hop.pdf` is generated from the real benchmark numbers in [`../benchmark_results/results.json`](../benchmark_results/results.json). See the plotting script in the project history, or regenerate manually with `matplotlib` using the values in Table 1 of the paper.
