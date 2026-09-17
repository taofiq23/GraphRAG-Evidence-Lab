"""Embedding-similarity retrieval - the standard RAG baseline being
compared against graph traversal. Every triple in the graph is rendered
as one sentence and embedded once; retrieval is top-k cosine similarity
against the question.
"""

from typing import List

import numpy as np
from sentence_transformers import SentenceTransformer

from .config import EMBEDDING_MODEL_NAME
from .knowledge_graph import TRIPLES, Triple, triple_to_sentence


class VectorRetriever:
    def __init__(self, triples: List[Triple] = TRIPLES, model: SentenceTransformer = None):
        self.triples = list(triples)
        self.sentences = [triple_to_sentence(t) for t in self.triples]
        self.model = model or SentenceTransformer(EMBEDDING_MODEL_NAME)
        self.embeddings = self.model.encode(self.sentences, normalize_embeddings=True)

    def retrieve(self, question: str, k: int) -> List[Triple]:
        query_embedding = self.model.encode([question], normalize_embeddings=True)[0]
        scores = self.embeddings @ query_embedding
        top_k_idx = np.argsort(-scores)[:k]
        return [self.triples[i] for i in top_k_idx]
