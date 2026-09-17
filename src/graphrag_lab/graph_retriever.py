"""Graph-based retrieval: find entities mentioned in the question, then
pull the k-hop neighborhood around them as evidence. This is the
"GraphRAG" half of the comparison - traversal instead of embedding
similarity.
"""

from typing import List, Set

import networkx as nx

from .knowledge_graph import Triple


def find_seed_entities(question: str, entity_names: List[str]) -> List[str]:
    """Exact-substring entity linking against the known entity vocabulary.
    Longest names are checked first so e.g. "SolsticeRobotics" isn't missed
    in favor of a shorter false match.
    """
    found = []
    for entity in sorted(entity_names, key=len, reverse=True):
        readable = _spaced(entity)
        if readable in question:
            found.append(entity)
    return found


def _spaced(entity: str) -> str:
    from .knowledge_graph import display_name

    return display_name(entity)


def k_hop_evidence(graph: nx.MultiDiGraph, seed_entities: List[str], hops: int) -> List[Triple]:
    """Undirected BFS from each seed entity out to `hops` steps, returning
    every triple (edge) touched. Direction is ignored when traversing since
    evidence-gathering should find a fact regardless of which way it reads,
    even though the fact is stored and rendered with its real direction.
    """
    undirected = graph.to_undirected(as_view=True)
    visited_nodes: Set[str] = set(seed_entities)
    frontier: Set[str] = set(seed_entities)

    for _ in range(hops):
        next_frontier: Set[str] = set()
        for node in frontier:
            if node not in undirected:
                continue
            next_frontier.update(undirected.neighbors(node))
        next_frontier -= visited_nodes
        visited_nodes.update(next_frontier)
        frontier = next_frontier
        if not frontier:
            break

    triples: List[Triple] = []
    seen = set()
    for u, v, data in graph.edges(data=True):
        if u in visited_nodes and v in visited_nodes:
            triple = (u, data["relation"], v)
            if triple not in seen:
                seen.add(triple)
                triples.append(triple)
    return triples
