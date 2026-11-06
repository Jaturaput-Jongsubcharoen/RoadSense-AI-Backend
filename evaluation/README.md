# Phase B: offline RAG evaluation

## Source inspection and boundaries

Inspected before implementation: `services/rag_service.py`, bundled PDF pages,
`examples/documents/metadata.json`, and the existing lazy-loading regression test.
Production uses PyPDF2 extraction, word-based approximately 800-character chunks,
`all-MiniLM-L6-v2` embeddings and FAISS `IndexFlatL2`, without normalization.
Uploads replace process-local `DOCS`/`INDEX`; bundled PDFs are not loaded at startup.
This evaluation will use a **separate index over all four bundled PDFs**. It is not
a measurement of the upload route's single-document index.

References were checked against locally extracted PDF text, not filenames or
metadata suggestions. PDF pages are one-based; printed page numbers differ.
The 2025 PDF's extractable text on page 1 is only its source line, so its question
is limited to attribution. No chart values were inferred. Dataset reviewer notes
describe source verification, **not review of a model's answer**.

## Metrics

- Hit@K: 1 if any of the first K retrieved chunks belongs to an expected document.
- MRR@K: reciprocal rank of the first such chunk, averaged over labeled questions.
  Ranks are chunk ranks, not deduplicated document ranks. This is a coarse
  document-level metric, not a claim that an answer-bearing chunk was retrieved.
- Expected-fact coverage: fraction of facts with at least one listed phrase in
  the answer (case/punctuation normalized). This is **not correctness**: negation,
  wrong attribution, or copying facts into a false answer can still match.
- Manual accuracy: `Correct`, `Partially correct`, `Incorrect`,
  `Not enough evidence`, or `Unreviewed`.
  Strict accuracy = percentage Correct among reviewed answers; partial-credit
  score = percentage with weights 1, 0.5, and 0 for the other reviewed labels.
- Separate grounding: `Supported`, `Partially supported`, `Unsupported`,
  or `Unreviewed`.
  Grounding percentage = percentage Supported among grounding-reviewed answers.
  Review every material claim against the **retrieved context**, not just the
  reference. A correct answer can be unsupported by its retrieved context.
- Missing labels, answers, facts, and unreviewed records are excluded from their
  own denominators. Empty retrieval with known labels scores zero. Empty metric
  populations return `null`, never a fabricated zero or perfect score.

## Run tests (no models or servers)

From the backend directory:

```text
.venv\Scripts\python.exe -m unittest discover -s tests -p test_rag_evaluation.py -v
```

The PDF evidence test uses the already installed PyPDF2; it skips explicitly if
that dependency is absent. All other tests use standard-library fakes.

## Download the embedding model and run retrieval

The evaluation runner never downloads a model. Download and save the same model
used by the application once, then run the offline evaluator:

```powershell
.\.venv\Scripts\python.exe -c "import os; from sentence_transformers import SentenceTransformer; path=os.path.join(os.environ['USERPROFILE'],'.cache','roadsense','all-MiniLM-L6-v2'); SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save(path); print(path)"
.\.venv\Scripts\python.exe -m evaluation.offline --model-path "$env:USERPROFILE\.cache\roadsense\all-MiniLM-L6-v2" --output local-run.json
```

The first command requires internet access to download the model. The second
command requires no model download: it forces Hugging Face/Transformers offline
mode and accepts only an existing local model directory.
The run records the caller-supplied directory and marks model identity as unverified.
The caller must provide the actual `all-MiniLM-L6-v2` snapshot for production-model
parity; the presence of `modules.json` alone does not establish model identity.
It does not call Ollama or generate answers. Optional `--answers answers.json`
takes a JSON object mapping question IDs to saved answer strings. Without an
answer, coverage and answer reviews remain unscored. The Python `run_evaluation`
function also accepts an answer callback for caller-owned offline generation.
No production prompts or request contracts are changed.

## Compare the production and candidate RAG prompts

After creating a retrieval run, compare both prompt versions against its exact
saved questions and retrieved contexts. This uses the configured Ollama URL and
model, with the same temperature and timeout for each prompt:

```powershell
.\.venv\Scripts\python.exe -m evaluation.prompt_comparison --retrieval-run evaluation\retrieval-results.json --output evaluation\prompt-comparison.json --markdown evaluation\prompt-comparison.md
```

The baseline wording is shared with production; the candidate is experimental.
The runner records each result, token source, latency, fact coverage and an
unreviewed manual-review template. Optional `--reviews review-input.json` accepts
per-question `baseline` and `candidate` labels using the Phase B review categories.
Copy `manual_review_template` from the JSON report into a review file, fill in
the reviewer, notes, accuracy and grounding for each answer, then pass that file
with `--reviews`. Keep the exact answer and retrieved chunk IDs from the template;
the runner rejects stale reviews. An unreachable Ollama service produces
failed-request records and null answer-quality/token metrics; it does not invent
answers or reviewer scores.
The current human-readable blank rating checklist is
[`prompt-review-template.md`](./prompt-review-template.md); it is not a scored
evaluation.

Each run contains questions, references, facts, retrieved texts/IDs and an
unreviewed `review_template`. Copy that list to `reviews.json`. For each answer:

1. Compare to the reference and facts; set `accuracy`.
2. Independently check all claims against the saved retrieved context; set
   `grounding`.
3. Fill in `reviewer` and `notes`; retain the exact `answer` and ordered
   `retrieved_chunk_ids` so reviews cannot silently attach to a different run.
4. Score saved results without loading models:

```text
.venv\Scripts\python.exe -m evaluation.offline --score local-run.json --reviews reviews.json --output scored.json
```

Unreviewed dimensions stay excluded even if the other dimension was reviewed.
Review labels must use the spellings above. Unknown IDs, duplicates, stale
reviews, missing reviewer/notes, or invalid labels fail rather than inflate scores.

Document IDs are actual bundled filenames; chunk IDs are
`filename::chunk-0000` (zero-based production chunk order). Dataset SHA-256 values
pin document bytes; changed documents require re-verifying references and IDs.
The runner checks hashes before extraction. Chunk IDs are evaluation-only.

## Render safety

Nothing imports this package from Flask, and no service, route, dependency,
deployment configuration or environment file is modified. Run evaluation locally,
not during Render startup or on a memory-constrained web worker: it builds a
second index and loads an embedding model explicitly. Saved answers/reviews may
contain sensitive user text; keep outputs local and do not commit secrets.
This is a small sanity dataset, not production accuracy certification.
