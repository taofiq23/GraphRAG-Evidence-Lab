from src.graphrag_lab.vector_retriever import VectorRetriever


def test_vector_retriever_finds_relevant_triple_for_easy_question():
    retriever = VectorRetriever()
    results = retriever.retrieve("Who founded NovaTech?", k=3)
    assert ("Alice", "founded", "NovaTech") in results
