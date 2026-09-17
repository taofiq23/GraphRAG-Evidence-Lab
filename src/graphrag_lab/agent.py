"""The LLM-based agent: generates answers from retrieved evidence, and -
for the agentic GraphRAG strategy - decides for itself whether it has
enough evidence yet or needs to expand its graph search further, instead
of using a fixed hop count.
"""

from dataclasses import dataclass
from typing import List

from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import LLM_MODEL_NAME, MAX_AGENTIC_HOPS, VECTOR_RAG_TOP_K
from .graph_retriever import find_seed_entities, k_hop_evidence
from .knowledge_graph import Triple, build_graph, triple_to_sentence
from .vector_retriever import VectorRetriever


@dataclass
class AnswerResult:
    answer: str
    evidence: List[Triple]
    hops_used: int  # 0 for vector RAG (hop count isn't meaningful there)


class Agent:
    def __init__(self, vector_retriever: VectorRetriever, model_name: str = LLM_MODEL_NAME):
        self.graph = build_graph()
        self.entity_names = list(self.graph.nodes)
        self.vector_retriever = vector_retriever
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    def _generate(self, prompt: str, max_new_tokens: int = 32) -> str:
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=max_new_tokens)
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()

    def _evidence_to_context(self, evidence: List[Triple]) -> str:
        return " ".join(triple_to_sentence(t) for t in evidence)

    def _answer_from_evidence(self, question: str, evidence: List[Triple]) -> str:
        context = self._evidence_to_context(evidence)
        prompt = f"Context: {context}\nQuestion: {question}\nAnswer with just the name, in a few words:"
        return self._generate(prompt)

    def _is_evidence_sufficient(self, question: str, evidence: List[Triple]) -> bool:
        context = self._evidence_to_context(evidence)
        prompt = (
            f"Context: {context}\nQuestion: {question}\n"
            "Does the context above contain enough information to answer the question? Answer yes or no."
        )
        response = self._generate(prompt, max_new_tokens=5).lower()
        return response.startswith("yes")

    def answer_vector_rag(self, question: str, k: int = VECTOR_RAG_TOP_K) -> AnswerResult:
        evidence = self.vector_retriever.retrieve(question, k)
        answer = self._answer_from_evidence(question, evidence)
        return AnswerResult(answer, evidence, hops_used=0)

    def answer_fixed_graph_rag(self, question: str, hops: int = 1) -> AnswerResult:
        seeds = find_seed_entities(question, self.entity_names)
        evidence = k_hop_evidence(self.graph, seeds, hops) if seeds else []
        answer = self._answer_from_evidence(question, evidence)
        return AnswerResult(answer, evidence, hops_used=hops)

    def answer_agentic_graph_rag(self, question: str, max_hops: int = MAX_AGENTIC_HOPS) -> AnswerResult:
        seeds = find_seed_entities(question, self.entity_names)
        if not seeds:
            return AnswerResult(self._answer_from_evidence(question, []), [], hops_used=0)

        evidence: List[Triple] = []
        hops_used = 0
        for hops in range(1, max_hops + 1):
            evidence = k_hop_evidence(self.graph, seeds, hops)
            hops_used = hops
            if self._is_evidence_sufficient(question, evidence):
                break

        answer = self._answer_from_evidence(question, evidence)
        return AnswerResult(answer, evidence, hops_used=hops_used)
