"""Generates QA pairs directly from the knowledge graph's structure, using
a fixed set of relation-chain patterns (1, 2, and 3 hops). Because each
question is derived from real graph paths, every example carries a known
gold answer and the exact gold supporting triples - the ground truth the
retrieval benchmark measures against.
"""

from dataclasses import dataclass
from typing import Dict, List, Tuple

from .knowledge_graph import TRIPLES, Triple, display_name


@dataclass
class QAExample:
    question: str
    gold_answer: str
    gold_supporting_triples: List[Triple]
    hop_count: int
    seed_entities: List[str]


def _index_by_relation(triples: List[Triple]) -> Dict[str, List[Tuple[str, str]]]:
    index: Dict[str, List[Tuple[str, str]]] = {}
    for s, r, o in triples:
        index.setdefault(r, []).append((s, o))
    return index


def generate_qa_examples(triples: List[Triple] = TRIPLES) -> List[QAExample]:
    by_relation = _index_by_relation(triples)
    founded = {c: p for p, c in by_relation.get("founded", [])}
    ceo_of = {c: p for p, c in by_relation.get("ceo_of", [])}
    based_in = dict(by_relation.get("based_in", []))
    created = {prod: comp for comp, prod in by_relation.get("created", [])}
    acquired_by = dict(by_relation.get("acquired_by", []))
    released_in = dict(by_relation.get("released_in", []))

    d = display_name
    examples: List[QAExample] = []

    # --- 1-hop ---
    for company, person in founded.items():
        examples.append(
            QAExample(
                f"Who founded {d(company)}?",
                d(person),
                [(person, "founded", company)],
                1,
                [company],
            )
        )
    for company, city in based_in.items():
        examples.append(
            QAExample(
                f"Where is {d(company)} based?",
                d(city),
                [(company, "based_in", city)],
                1,
                [company],
            )
        )
    for product, company in created.items():
        examples.append(
            QAExample(
                f"Which company created {d(product)}?",
                d(company),
                [(company, "created", product)],
                1,
                [product],
            )
        )
    for company, acquirer in acquired_by.items():
        examples.append(
            QAExample(
                f"Which company acquired {d(company)}?",
                d(acquirer),
                [(company, "acquired_by", acquirer)],
                1,
                [company],
            )
        )
    for company, person in ceo_of.items():
        examples.append(
            QAExample(
                f"Who is the CEO of {d(company)}?",
                d(person),
                [(person, "ceo_of", company)],
                1,
                [company],
            )
        )
    for product, year in released_in.items():
        examples.append(
            QAExample(
                f"When was {d(product)} released?",
                year,
                [(product, "released_in", year)],
                1,
                [product],
            )
        )

    # --- 2-hop ---
    for product, company in created.items():
        if company in founded:
            person = founded[company]
            examples.append(
                QAExample(
                    f"Who founded the company that created {d(product)}?",
                    d(person),
                    [(company, "created", product), (person, "founded", company)],
                    2,
                    [product],
                )
            )
        if company in based_in:
            city = based_in[company]
            examples.append(
                QAExample(
                    f"Where is the company based that created {d(product)}?",
                    d(city),
                    [(company, "created", product), (company, "based_in", city)],
                    2,
                    [product],
                )
            )
        if company in ceo_of:
            person = ceo_of[company]
            examples.append(
                QAExample(
                    f"Who is the CEO of the company that created {d(product)}?",
                    d(person),
                    [(company, "created", product), (person, "ceo_of", company)],
                    2,
                    [product],
                )
            )

    for company, acquirer in acquired_by.items():
        if acquirer in based_in:
            city = based_in[acquirer]
            examples.append(
                QAExample(
                    f"Where is the company based that acquired {d(company)}?",
                    d(city),
                    [(company, "acquired_by", acquirer), (acquirer, "based_in", city)],
                    2,
                    [company],
                )
            )
        if company in founded:
            person = founded[company]
            examples.append(
                QAExample(
                    f"Which company acquired the company founded by {d(person)}?",
                    d(acquirer),
                    [(person, "founded", company), (company, "acquired_by", acquirer)],
                    2,
                    [person],
                )
            )

    # --- 3-hop ---
    for product, company in created.items():
        if company in acquired_by:
            acquirer = acquired_by[company]
            if acquirer in based_in:
                city = based_in[acquirer]
                examples.append(
                    QAExample(
                        f"Where is the company based that acquired the company that created {d(product)}?",
                        d(city),
                        [
                            (company, "created", product),
                            (company, "acquired_by", acquirer),
                            (acquirer, "based_in", city),
                        ],
                        3,
                        [product],
                    )
                )
            if acquirer in ceo_of:
                person = ceo_of[acquirer]
                examples.append(
                    QAExample(
                        f"Who is the CEO of the company that acquired the company that created {d(product)}?",
                        d(person),
                        [
                            (company, "created", product),
                            (company, "acquired_by", acquirer),
                            (person, "ceo_of", acquirer),
                        ],
                        3,
                        [product],
                    )
                )

    for company, person in founded.items():
        if company in acquired_by:
            acquirer = acquired_by[company]
            if acquirer in based_in:
                city = based_in[acquirer]
                examples.append(
                    QAExample(
                        f"Where is the company based that acquired the company founded by {d(person)}?",
                        d(city),
                        [
                            (person, "founded", company),
                            (company, "acquired_by", acquirer),
                            (acquirer, "based_in", city),
                        ],
                        3,
                        [person],
                    )
                )

    return examples
