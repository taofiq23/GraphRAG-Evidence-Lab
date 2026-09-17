"""Scoring: answer correctness (lenient containment match, standard for
generative QA with small LLMs) and retrieval precision/recall against the
gold supporting triples, aggregated by hop count.
"""

from dataclasses import dataclass
from typing import Callable, Dict, List, Tuple

from .agent import AnswerResult
from .knowledge_graph import Triple
from .qa_generator import QAExample


@dataclass
class QAResult:
    question: str
    hop_count: int
    gold_answer: str
    predicted_answer: str
    correct: bool
    retrieval_precision: float
    retrieval_recall: float
    hops_used: int


def _answer_correct(gold_answer: str, predicted_answer: str) -> bool:
    return gold_answer.strip().lower() in predicted_answer.strip().lower()


def _retrieval_scores(gold: List[Triple], retrieved: List[Triple]) -> Tuple[float, float]:
    gold_set, retrieved_set = set(gold), set(retrieved)
    precision = len(gold_set & retrieved_set) / len(retrieved_set) if retrieved_set else 0.0
    recall = len(gold_set & retrieved_set) / len(gold_set) if gold_set else 1.0
    return precision, recall


def evaluate_system(examples: List[QAExample], answer_fn: Callable[[str], AnswerResult]) -> List[QAResult]:
    results = []
    for example in examples:
        result = answer_fn(example.question)
        precision, recall = _retrieval_scores(example.gold_supporting_triples, result.evidence)
        results.append(
            QAResult(
                question=example.question,
                hop_count=example.hop_count,
                gold_answer=example.gold_answer,
                predicted_answer=result.answer,
                correct=_answer_correct(example.gold_answer, result.answer),
                retrieval_precision=precision,
                retrieval_recall=recall,
                hops_used=result.hops_used,
            )
        )
    return results


def summarize_by_hop(results: List[QAResult]) -> Dict[int, Dict[str, float]]:
    by_hop: Dict[int, List[QAResult]] = {}
    for r in results:
        by_hop.setdefault(r.hop_count, []).append(r)

    summary = {}
    for hop_count, group in sorted(by_hop.items()):
        n = len(group)
        summary[hop_count] = {
            "n": n,
            "accuracy": sum(r.correct for r in group) / n,
            "retrieval_precision": sum(r.retrieval_precision for r in group) / n,
            "retrieval_recall": sum(r.retrieval_recall for r in group) / n,
            "avg_hops_used": sum(r.hops_used for r in group) / n,
        }
    return summary
