from src.graphrag_lab.knowledge_graph import TRIPLES
from src.graphrag_lab.qa_generator import generate_qa_examples


def test_generates_examples_at_each_hop_count():
    examples = generate_qa_examples()
    hop_counts = {e.hop_count for e in examples}
    assert hop_counts == {1, 2, 3}
    assert len(examples) > 20


def test_gold_supporting_triples_are_real_edges():
    examples = generate_qa_examples()
    triple_set = set(TRIPLES)
    for example in examples:
        for triple in example.gold_supporting_triples:
            assert triple in triple_set, f"{triple} not a real edge (question: {example.question!r})"


def test_hop_count_matches_supporting_triple_count():
    examples = generate_qa_examples()
    for example in examples:
        assert len(example.gold_supporting_triples) == example.hop_count
