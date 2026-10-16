# RAG Chatbot for the EDAB 6808 Learner Guide

**An AI assistant that answers from one document only, cites the exact page for every answer, and says "I don't know" when the document doesn't say.**

*EDAB 6808 Assignment 5 · B.Com Business Analytics (Hons), University of the Free State · Group project (three members), 2026*

---

## Problem
Build a retrieval-augmented generation (RAG) chatbot that answers questions about the EDAB 6808 Learner Guide **accurately and transparently**. Every answer must show its source, and questions the guide doesn't support must be refused. Everything runs locally in Google Colab with **open-weight models** and no hosted LLM API.

## Data
| Source | What | Size |
|---|---|---|
| EDAB 6808 Learner Guide (module document) | The only knowledge source | 16 pages → 58 chunks |

The learner guide is a module document, so it is not redistributed in this repo. To run the notebook, supply your own copy.

## Tools
Python · Google Colab (T4 GPU) · `PyMuPDF` · `sentence-transformers` (BAAI/bge-small-en-v1.5) · `qdrant-client` (Qdrant Local) · `rank-bm25` · cross-encoder reranker · `transformers` (Qwen2.5-3B-Instruct) · `gradio`

## Pipeline
```
Learner Guide PDF
  → PyMuPDF extraction (reading order, tables row by row) → remove footers and artefacts → preservation checks
  → section-aware chunks of ≤ 200 tokens, with stable IDs and page and section metadata
  → BGE embeddings (384-d) → Qdrant Local collection on Google Drive (rerun-safe)
  → question → dense top-20 + BM25 top-20 → Reciprocal Rank Fusion → cross-encoder rerank → top k = 4
  → relevance gate (abstain if weak) + strongest-passage filter
  → Qwen2.5-3B-Instruct, evidence-only prompt, greedy decoding
  → citations built from stored metadata and verified per statement + faithfulness check
  → Gradio chat interface
```

## Method
1. **Extraction:** PyMuPDF with block sorting for reading order and `find_tables()` for the tables. Footers and look-alike Cyrillic characters were removed. **310/310 sentence units and 99.8% of words were preserved.**
2. **Chunking:** chunks of ≤ 200 tokens (counted with the embedding model's own tokenizer) that never cross a section, sub-section or table boundary, with a one-sentence overlap. 200 tokens was chosen by testing 64, 128, 200 and 320 for retrieval hit rate, context length and citation precision.
3. **Embeddings and storage:** `BAAI/bge-small-en-v1.5` (384 dimensions, cosine similarity) and Qdrant Local persisted on Google Drive. Deterministic `uuid5` point IDs and a content fingerprint make re-runs safe (58 vectors = 58 chunks, no duplicates).
4. **Retrieval:** dense search plus BM25, merged with Reciprocal Rank Fusion, then a cross-encoder reranks the results and **k = 4** passages are kept.
5. **Abstention gate:** calibrated on a *separate* development set. The lowest in-scope score was 0.739, the highest out-of-scope score was 0.158, and the gate was set at 0.342.
6. **Generation:** Qwen2.5-3B-Instruct sees only passages labelled [E1] to [E4], with no page numbers, and must answer from them or reply `INSUFFICIENT_EVIDENCE`.
7. **Citations:** page, section and chunk ID are attached from Qdrant metadata, never written by the model. Each statement's citation is verified against the evidence and corrected if needed.

## Evaluation
20 test questions (14 factual, 3 multi-part and 3 out-of-scope), covering 9 sections of the guide and kept separate from the development set.

| Measure | Result |
|---|---|
| Hit@4 | **1.000** |
| MRR | **1.000** |
| Context recall | 1.000 |
| Context precision | 0.294 |
| Answer accuracy | **1.000** |
| Citation accuracy (verified) | **0.941** |
| Citation accuracy (model labels only) | 0.882 |
| Faithfulness (proxy) | 0.735 |
| Abstention success rate | **1.000** |
| False abstention rate | **0.000** |
| Mean / P95 response time | 3.1 s / 5.2 s |

**Ablation: what each retrieval stage adds**
| Retrieval | Hit@4 | MRR |
|---|---|---|
| Dense only | 0.941 | 0.853 |
| Dense + BM25 (RRF) | 1.000 | 0.941 |
| Dense + BM25 + RRF + reranker | 1.000 | **1.000** |

## Key findings
- **Hybrid retrieval beats dense search alone.** BM25 caught the exact terms the embeddings missed, and the reranker put the right passage first every time.
- **Don't trust an LLM's citations blindly.** Verification corrected the model's labels in 11.8% of answers, raising citation accuracy from 0.882 to 0.941.
- **A calibrated gate refuses unanswerable questions without refusing real ones**, including near-misses such as deadlines, fees and room numbers.
- **The faithfulness proxy is conservative.** It flagged some correct short answers (for example "Assignment 5 contributes 20%"), so it should be read alongside the answers themselves.

![Gradio interface with an answer, citations and evidence panel](images/gradio_demo.png)

## How to run
1. Open the notebook in Google Colab: [Open in Colab](COLAB_LINK_HERE)
2. **Runtime → Change runtime type → T4 GPU.**
3. Put the learner guide PDF in `MyDrive/EDAB6808_RAG/`, or upload it when prompted.
4. **Runtime → Run all.** The first run builds the Qdrant collection on Drive, and later runs reuse it.
5. The last cell launches the Gradio app and prints a public link (valid for one week). A demo video is linked below.

## Demo
- [Demo video](DEMO_VIDEO_LINK_HERE): a factual question, a multi-part question and an out-of-scope refusal in one session.

## Repository structure
```
rag-chatbot-learner-guide/
├── README.md
├── edab6808_rag_chatbot.ipynb
└── images/
    ├── pipeline.png
    ├── chunk_size_tradeoff.png
    ├── gradio_demo.png
    └── gradio_refusal.png
```

## Limitations
- Reranker scores measure relevance, not truth.
- 20 test questions give a useful but small estimate of reliability.
- The faithfulness measure is a proxy and cannot prove that a statement means the same as its source.
- Qwen2.5-3B-Instruct is used as released, without fine-tuning on the guide.

## Links
- [Portfolio page](EPORTFOLIO_LINK_HERE)

## References
- Cormack, G. V., Clarke, C. L. A., & Büttcher, S. (2009). Reciprocal rank fusion outperforms Condorcet and individual rank learning methods. *Proceedings of SIGIR '09*, 758–759.
- Es, S., James, J., Espinosa-Anke, L., & Schockaert, S. (2023). RAGAS: Automated evaluation of retrieval augmented generation. *arXiv:2309.15217*.
- Lewis, P., et al. (2020). Retrieval-augmented generation for knowledge-intensive NLP tasks. *Advances in NeurIPS 33*.
- Robertson, S., & Zaragoza, H. (2009). The probabilistic relevance framework: BM25 and beyond. *Foundations and Trends in Information Retrieval, 3*(4), 333–389.
