# Paper

**Retrieval Succeeds, Reasoning Fails: Disentangling Bottlenecks in Multi-Hop Knowledge Graph Question Answering with Small Language Models**

A 9-page paper (single-column, 11pt) based on this repository's benchmark results. [Read the compiled PDF](paper.pdf).

**Published as a preprint on Zenodo** (CC BY 4.0, Version 2.0): [doi.org/10.5281/zenodo.22848733](https://doi.org/10.5281/zenodo.22848733)

## Notes on citing and sharing

- **Citations were checked, but re-check before reusing.** All 8 references (Lewis et al. 2020 on RAG, Edge et al. 2024 on GraphRAG, Zhang et al. 2018 on MetaQA, Yang et al. 2018 on HotpotQA, Yao et al. 2023 on ReAct, Wei et al. 2022 on chain-of-thought, Chung et al. 2022 on FLAN-T5, Reimers & Gurevych 2019 on Sentence-BERT) were verified against arXiv, the ACL Anthology, or the NeurIPS/AAAI/ICLR proceedings for title, authors, venue, and year. Author lists for the two arXiv-only papers with many authors (Edge et al., Chung et al.) are truncated with "et al." Spot-check them yourself against Google Scholar before submitting anywhere.
- **This is a preprint-quality writeup, not a peer-reviewed result.** It's based on real experiments with real numbers, and it has a permanent DOI via Zenodo, but it has not been through peer review and should be described as a "preprint on Zenodo", not as "published in" a journal or conference.

## Optional: also post it on arXiv

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
