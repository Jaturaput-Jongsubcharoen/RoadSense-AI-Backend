# Offline RAG Prompt Comparison

- Model: `llama3.1`
- Baseline: `production-rag-v1`
- Candidate: `candidate-context-focused-v1`
- Generation settings: `{"temperature": 0.1}`
- Test questions: 10
- Retrieval index: `FAISS IndexFlatL2; unnormalized; all bundled PDFs`
- Embedding model path: `C:\Users\kulis\.cache\roadsense\all-MiniLM-L6-v2`

No prompt is automatically selected or copied into production.
Manual accuracy and grounding remain unscored until answers are reviewed.

## Retrieval metrics from the shared context run

| Metric | Measured value |
|---|---:|
| Hit@1 | 1.0 |
| Hit@3 | 1.0 |
| Hit@5 | 1.0 |
| MRR@5 | 1.0 |

| Metric | Baseline | Candidate |
|---|---:|---:|
| Successful requests | 10 | 10 |
| Failed requests | 0 | 0 |
| Expected-fact coverage (not accuracy) | 0.8333333333333333 | 0.8999999999999998 |
| Mean actual provider tokens | 1095.3 | 1117.2 |
| Mean actual-or-estimated tokens | 1095.3 | 1117.2 |
| Mean successful response latency (ms) | 4654.7 | 3931.2 |
| Mean failed request duration (ms) | unavailable | unavailable |
| Human strict accuracy (%) | unavailable | unavailable |
| Human partial-credit (%) | unavailable | unavailable |
| Human grounding supported (%) | unavailable | unavailable |

## Per-question outcomes

| Question ID | Baseline | Candidate | Fact coverage (B/C) |
|---|---|---|---:|
| bc-road-length | success | success | 1.0 / 1.0 |
| bc-unpaved-share | success | success | 1.0 / 1.0 |
| bc-roadless-definition | success | success | 1.0 / 1.0 |
| ontario-fatalities-change | success | success | 1.0 / 1.0 |
| ontario-fatality-rate | success | success | 1.0 / 1.0 |
| ontario-inattentive-driving | success | success | 1.0 / 1.0 |
| canada-strategy-slogan | success | success | 1.0 / 1.0 |
| canada-occupant-protection | success | success | 0.3333333333333333 / 0.3333333333333333 |
| canada-seat-belt-benefits | success | success | 1.0 / 1.0 |
| transport-2025-source | success | success | 0.0 / 0.6666666666666666 |

## Prompt versions

### Baseline

Uses the unchanged production RAG prompt.

### Candidate

```text
You are a careful road-safety assistant.
Answer the question directly using only the supplied context.
Keep relevant dates, numbers, units, and attribution accurate.
Do not add facts that are not stated in the context.
If the context does not contain enough information, reply exactly:
"I don't have enough information in the uploaded document to answer that."

CONTEXT:
{context}

QUESTION:
{question}
```

## Limitations

- Manual accuracy and grounding are unscored until a reviewer supplies labels.
- Expected-fact coverage is literal phrase matching, not factual accuracy.
- Token estimates use whitespace-separated words when provider counts are absent.
- Latency is measured sequentially and can vary with model/server load.
