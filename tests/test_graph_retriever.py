from src.graphrag_lab.graph_retriever import find_seed_entities, k_hop_evidence
from src.graphrag_lab.knowledge_graph import build_graph


def test_find_seed_entities_matches_known_entity():
    # Questions are generated via display_name(), so "NovaTech" appears as
    # the spaced "Nova Tech" in real question text - matching must use the
    # same rendering, which is exactly what this test checks.
    entities = ["NovaTech", "Aurora", "Berlin"]
    found = find_seed_entities("Who founded Nova Tech?", entities)
    assert found == ["NovaTech"]


def test_find_seed_entities_no_match():
    entities = ["NovaTech", "Aurora"]
    assert find_seed_entities("What is the weather today?", entities) == []


def test_k_hop_evidence_one_hop_contains_direct_facts():
    graph = build_graph()
    evidence = k_hop_evidence(graph, ["Aurora"], hops=1)
    assert ("NovaTech", "created", "Aurora") in evidence
    assert ("Aurora", "released_in", "2020") in evidence
    # The founder is two hops away (Aurora -> NovaTech -> Alice), not one.
    assert ("Alice", "founded", "NovaTech") not in evidence


def test_k_hop_evidence_two_hops_reaches_founder():
    graph = build_graph()
    evidence = k_hop_evidence(graph, ["Aurora"], hops=2)
    assert ("Alice", "founded", "NovaTech") in evidence
