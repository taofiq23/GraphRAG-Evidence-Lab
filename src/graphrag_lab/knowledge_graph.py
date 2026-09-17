"""A small, hand-authored synthetic knowledge graph (people, companies,
products, cities). It's synthetic so the QA generator below can produce
questions with known-correct gold answers and gold supporting facts -
that ground truth is what makes the retrieval benchmark measurable.
"""

from typing import List, Tuple

import networkx as nx

Triple = Tuple[str, str, str]

# (subject, relation, object)
TRIPLES: List[Triple] = [
    # Founder and CEO are deliberately different people at every company:
    # if they were the same, a graph-traversal agent could satisfy a
    # "who is the CEO" stopping check one hop too early, at an intermediate
    # company in the chain rather than the actual target company.
    ("Alice", "founded", "NovaTech"),
    ("Bob", "ceo_of", "NovaTech"),
    ("Carol", "works_at", "NovaTech"),
    ("NovaTech", "based_in", "Berlin"),
    ("NovaTech", "created", "Aurora"),
    ("NovaTech", "created", "Nimbus"),
    ("NovaTech", "acquired_by", "MegaCorp"),
    ("Dave", "founded", "ByteWorks"),
    ("Erin", "ceo_of", "ByteWorks"),
    ("Jack", "works_at", "ByteWorks"),
    ("ByteWorks", "based_in", "Austin"),
    ("ByteWorks", "created", "CodeStream"),
    ("ByteWorks", "acquired_by", "MegaCorp"),
    ("Frank", "founded", "SolsticeRobotics"),
    ("Karen", "ceo_of", "SolsticeRobotics"),
    ("SolsticeRobotics", "based_in", "Toronto"),
    ("SolsticeRobotics", "created", "RoboArmX"),
    ("Grace", "founded", "MegaCorp"),
    ("Leo", "ceo_of", "MegaCorp"),
    ("MegaCorp", "based_in", "Singapore"),
    ("MegaCorp", "created", "MegaSuite"),
    ("Heidi", "founded", "PixelForge"),
    ("Nina", "ceo_of", "PixelForge"),
    ("PixelForge", "based_in", "Lisbon"),
    ("PixelForge", "created", "PixelStudio"),
    ("PixelForge", "acquired_by", "SolsticeRobotics"),
    ("Ivan", "founded", "QuantumLeap"),
    ("Oscar", "ceo_of", "QuantumLeap"),
    ("QuantumLeap", "based_in", "Austin"),
    ("QuantumLeap", "created", "QLeapEngine"),
    ("Aurora", "released_in", "2020"),
    ("Nimbus", "released_in", "2022"),
    ("CodeStream", "released_in", "2019"),
    ("RoboArmX", "released_in", "2021"),
    ("MegaSuite", "released_in", "2018"),
    ("PixelStudio", "released_in", "2023"),
    ("QLeapEngine", "released_in", "2021"),
]

# Natural-language templates for rendering a triple as evidence text, keyed
# by relation. {s} and {o} are the (human-readable) subject/object.
_SENTENCE_TEMPLATES = {
    "founded": "{s} founded {o}.",
    "ceo_of": "{s} is the CEO of {o}.",
    "based_in": "{s} is based in {o}.",
    "created": "{s} created {o}.",
    "works_at": "{s} works at {o}.",
    "acquired_by": "{s} was acquired by {o}.",
    "released_in": "{s} was released in {o}.",
}


def display_name(entity: str) -> str:
    # Splits CamelCase entity ids into readable names: "SolsticeRobotics" -> "Solstice Robotics".
    out = []
    for i, ch in enumerate(entity):
        if i > 0 and ch.isupper() and not entity[i - 1].isupper():
            out.append(" ")
        out.append(ch)
    return "".join(out)


def triple_to_sentence(triple: Triple) -> str:
    s, r, o = triple
    template = _SENTENCE_TEMPLATES[r]
    return template.format(s=display_name(s), o=display_name(o))


def build_graph(triples: List[Triple] = TRIPLES) -> nx.MultiDiGraph:
    graph = nx.MultiDiGraph()
    for s, r, o in triples:
        graph.add_edge(s, o, relation=r)
    return graph


def all_entities(triples: List[Triple] = TRIPLES) -> List[str]:
    entities = set()
    for s, _, o in triples:
        entities.add(s)
        entities.add(o)
    return sorted(entities)
