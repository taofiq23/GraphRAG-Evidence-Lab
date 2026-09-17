from src.graphrag_lab.knowledge_graph import (
    TRIPLES,
    build_graph,
    display_name,
    triple_to_sentence,
)


def test_display_name_splits_camel_case():
    assert display_name("SolsticeRobotics") == "Solstice Robotics"
    assert display_name("Alice") == "Alice"


def test_triple_to_sentence():
    # NovaTech is rendered as "Nova Tech" for readability, consistent with
    # display_name() - see test_display_name_splits_camel_case.
    sentence = triple_to_sentence(("Alice", "founded", "NovaTech"))
    assert sentence == "Alice founded Nova Tech."


def test_build_graph_has_all_entities_and_edges():
    graph = build_graph()
    assert graph.number_of_edges() == len(TRIPLES)
    assert "Alice" in graph.nodes
    assert "NovaTech" in graph.nodes
