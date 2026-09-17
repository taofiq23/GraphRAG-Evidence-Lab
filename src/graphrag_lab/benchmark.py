"""End-to-end benchmark: vector RAG vs fixed 1-hop GraphRAG vs agentic
GraphRAG, evaluated on the generated QA set and broken down by hop count.

Usage:
    python -m src.graphrag_lab.benchmark
"""

import json
from pathlib import Path

from .agent import Agent
from .evaluation import evaluate_system, summarize_by_hop
from .qa_generator import generate_qa_examples
from .vector_retriever import VectorRetriever

RESULTS_DIR = Path(__file__).resolve().parent.parent.parent / "benchmark_results"


def run_benchmark() -> dict:
    examples = generate_qa_examples()
    vector_retriever = VectorRetriever()
    agent = Agent(vector_retriever)

    systems = {
        "vector_rag": lambda q: agent.answer_vector_rag(q),
        "fixed_1hop_graph_rag": lambda q: agent.answer_fixed_graph_rag(q, hops=1),
        "agentic_graph_rag": lambda q: agent.answer_agentic_graph_rag(q),
    }

    report: dict = {"n_examples": len(examples), "systems": {}}
    for name, answer_fn in systems.items():
        results = evaluate_system(examples, answer_fn)
        report["systems"][name] = summarize_by_hop(results)

    return report


def render_markdown(report: dict) -> str:
    lines = ["# GraphRAG Evidence Lab Benchmark Results", "", f"{report['n_examples']} generated QA examples.", ""]
    hop_counts = sorted({int(hop) for system in report["systems"].values() for hop in system})

    for system_name, by_hop in report["systems"].items():
        lines.append(f"## {system_name}")
        lines.append("")
        lines.append("| Hops | N | Accuracy | Retrieval precision | Retrieval recall | Avg hops used |")
        lines.append("|---|---|---|---|---|---|")
        for hop in hop_counts:
            m = by_hop.get(hop) or by_hop.get(str(hop))
            if m is None:
                continue
            lines.append(
                f"| {hop} | {m['n']} | {m['accuracy']:.0%} | {m['retrieval_precision']:.0%} | "
                f"{m['retrieval_recall']:.0%} | {m['avg_hops_used']:.1f} |"
            )
        lines.append("")
    return "\n".join(lines)


def main():
    report = run_benchmark()
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (RESULTS_DIR / "results.json").write_text(json.dumps(report, indent=2))
    markdown = render_markdown(report)
    (RESULTS_DIR / "RESULTS.md").write_text(markdown)
    print(markdown)


if __name__ == "__main__":
    main()
