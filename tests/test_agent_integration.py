from src.graphrag_lab.agent import Agent
from src.graphrag_lab.vector_retriever import VectorRetriever


def test_all_three_answer_strategies_run_end_to_end():
    agent = Agent(VectorRetriever())
    question = "Who founded the company that created Aurora?"

    vector_result = agent.answer_vector_rag(question)
    fixed_result = agent.answer_fixed_graph_rag(question, hops=1)
    agentic_result = agent.answer_agentic_graph_rag(question)

    for result in (vector_result, fixed_result, agentic_result):
        assert isinstance(result.answer, str) and result.answer
        assert isinstance(result.evidence, list)

    assert agentic_result.hops_used >= 1
