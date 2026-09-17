"""Gradio demo: ask a question and compare vector RAG against agentic
GraphRAG side by side - answer, retrieved evidence, and (for GraphRAG)
how many hops the agent decided it needed.
"""

import io

import gradio as gr
import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from src.graphrag_lab.agent import Agent
from src.graphrag_lab.knowledge_graph import build_graph, display_name
from src.graphrag_lab.qa_generator import generate_qa_examples
from src.graphrag_lab.vector_retriever import VectorRetriever

_vector_retriever = VectorRetriever()
_agent = Agent(_vector_retriever)
_graph = build_graph()
_examples = generate_qa_examples()
_example_questions = [e.question for e in _examples]


def _render_evidence_graph(evidence) -> Image.Image:
    fig, ax = plt.subplots(figsize=(5, 4))
    if not evidence:
        ax.text(0.5, 0.5, "No graph evidence retrieved", ha="center", va="center")
        ax.axis("off")
    else:
        sub = nx.DiGraph()
        for s, r, o in evidence:
            sub.add_edge(display_name(s), display_name(o), relation=r)
        pos = nx.spring_layout(sub, seed=0)
        nx.draw(sub, pos, ax=ax, with_labels=True, node_color="#7b93ff", node_size=1800, font_size=8)
        edge_labels = {(u, v): d["relation"] for u, v, d in sub.edges(data=True)}
        nx.draw_networkx_edge_labels(sub, pos, edge_labels=edge_labels, ax=ax, font_size=7)
        ax.axis("off")

    buf = io.BytesIO()
    fig.tight_layout()
    fig.savefig(buf, format="png", dpi=120)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf)


def _format_evidence(evidence) -> str:
    from src.graphrag_lab.knowledge_graph import triple_to_sentence

    if not evidence:
        return "(no evidence retrieved)"
    return "\n".join(f"- {triple_to_sentence(t)}" for t in evidence)


def answer_question(question: str):
    if not question or not question.strip():
        return "Enter a question.", "", "", "", None

    vector_result = _agent.answer_vector_rag(question)
    graph_result = _agent.answer_agentic_graph_rag(question)

    graph_image = _render_evidence_graph(graph_result.evidence)
    hops_text = f"Agent used {graph_result.hops_used} hop(s) of graph traversal."

    return (
        vector_result.answer,
        _format_evidence(vector_result.evidence),
        graph_result.answer,
        _format_evidence(graph_result.evidence) + f"\n\n{hops_text}",
        graph_image,
    )


with gr.Blocks(title="GraphRAG Evidence Lab") as demo:
    gr.Markdown(
        "# GraphRAG Evidence Lab\n"
        "Ask a question about the synthetic company/product knowledge graph and compare "
        "**vector RAG** (embedding similarity) against **agentic GraphRAG** (the agent decides "
        "for itself how many hops of graph traversal it needs before answering)."
    )

    question_box = gr.Dropdown(
        choices=_example_questions,
        label="Example questions (or type your own below)",
        allow_custom_value=True,
    )
    submit_btn = gr.Button("Ask", variant="primary")

    with gr.Row():
        with gr.Column():
            gr.Markdown("### Vector RAG")
            vector_answer = gr.Textbox(label="Answer")
            vector_evidence = gr.Textbox(label="Retrieved evidence", lines=5)
        with gr.Column():
            gr.Markdown("### Agentic GraphRAG")
            graph_answer = gr.Textbox(label="Answer")
            graph_evidence = gr.Textbox(label="Retrieved evidence", lines=5)

    graph_plot = gr.Image(label="Graph evidence visualization")

    submit_btn.click(
        answer_question,
        inputs=[question_box],
        outputs=[vector_answer, vector_evidence, graph_answer, graph_evidence, graph_plot],
    )

if __name__ == "__main__":
    demo.launch()
