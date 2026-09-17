# GraphRAG Evidence Lab Benchmark Results

73 generated QA examples.

## vector_rag

| Hops | N | Accuracy | Retrieval precision | Retrieval recall | Avg hops used |
|---|---|---|---|---|---|
| 1 | 35 | 100% | 33% | 100% | 0.0 |
| 2 | 27 | 44% | 47% | 70% | 0.0 |
| 3 | 11 | 9% | 36% | 36% | 0.0 |

## fixed_1hop_graph_rag

| Hops | N | Accuracy | Retrieval precision | Retrieval recall | Avg hops used |
|---|---|---|---|---|---|
| 1 | 35 | 100% | 31% | 100% | 1.0 |
| 2 | 27 | 0% | 52% | 50% | 1.0 |
| 3 | 11 | 0% | 64% | 33% | 1.0 |

## agentic_graph_rag

| Hops | N | Accuracy | Retrieval precision | Retrieval recall | Avg hops used |
|---|---|---|---|---|---|
| 1 | 35 | 100% | 28% | 100% | 1.4 |
| 2 | 27 | 74% | 17% | 100% | 2.9 |
| 3 | 11 | 9% | 25% | 94% | 2.8 |
