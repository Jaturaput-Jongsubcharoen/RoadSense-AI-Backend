"""Isolated FAISS evaluation; all heavyweight imports are opt-in."""

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path

from evaluation.metrics import (
    expected_fact_coverage, hit_at_k, mean_score, reciprocal_rank,
    score_reviews, validate_k,
)

ROOT = Path(__file__).resolve().parents[1]
DATASET = Path(__file__).with_name("dataset.json")


def load_dataset(path=DATASET, documents_dir=ROOT / "examples" / "documents"):
    dataset = json.loads(Path(path).read_text(encoding="utf-8"))
    for document_id, source in dataset["documents"].items():
        _validate_document_id(document_id)
        actual = hashlib.sha256((Path(documents_dir) / document_id).read_bytes()).hexdigest()
        if actual != source["sha256"]:
            raise ValueError(f"Source changed; re-verify dataset: {document_id}")
    seen = set()
    for item in dataset["questions"]:
        if item["id"] in seen:
            raise ValueError("Duplicate question ID")
        seen.add(item["id"])
        if not item["expected_document_ids"] or any(
                name not in dataset["documents"] for name in item["expected_document_ids"]):
            raise ValueError("Question must reference actual bundled document IDs")
        if not item["sources"] or not item["reviewer_note"]:
            raise ValueError("Question needs source verification notes")
        for source in item["sources"]:
            if (source["document_id"] not in item["expected_document_ids"]
                    or type(source["pdf_page"]) is not int or source["pdf_page"] < 1):
                raise ValueError("Invalid question source/page")
        if expected_fact_coverage(item["reference_answer"], item["expected_facts"]) != 1:
            raise ValueError("Reference answer must contain all annotated facts")
    return dataset


def _validate_document_id(document_id):
    if (not isinstance(document_id, str) or "/" in document_id or "\\" in document_id
            or Path(document_id).suffix.lower() != ".pdf"):
        raise ValueError("Document IDs must be bundled PDF filenames")


@dataclass(frozen=True)
class Chunk:
    document_id: str
    chunk_id: str
    text: str


def build_chunks(documents_dir, document_ids, parser=None, chunker=None):
    """Use production extraction/chunk boundaries, never its DOCS/INDEX."""
    if parser is None or chunker is None:
        from services.rag_service import _chunk_text, _extract_text_from_file
        parser = parser or _extract_text_from_file
        chunker = chunker or _chunk_text
    chunks = []
    for document_id in document_ids:
        _validate_document_id(document_id)
        path = Path(documents_dir) / document_id
        text = parser(str(path))
        for number, text_chunk in enumerate(chunker(text)):
            chunks.append(Chunk(document_id, f"{document_id}::chunk-{number:04d}",
                                text_chunk))
    return chunks


class OfflineRetriever:
    """Same unnormalized embedding/L2 search as production, with bounded K."""

    def __init__(self, chunks, embedder, index_factory=None, array=None):
        self.chunks = list(chunks)
        self.embedder = embedder
        self.index = None
        self.array = array
        if not self.chunks:
            return
        if index_factory is None:
            import faiss
            index_factory = faiss.IndexFlatL2
        if array is None:
            import numpy as np
            self.array = np.array
        vectors = embedder.encode([chunk.text for chunk in self.chunks])
        self.index = index_factory(vectors.shape[1])
        self.index.add(self.array(vectors))

    def retrieve(self, question, k=5):
        validate_k(k)
        if self.index is None or k == 0:
            return []
        vectors = self.embedder.encode([question])
        _, indices = self.index.search(self.array(vectors), min(k, len(self.chunks)))
        return [self.chunks[int(i)] for i in indices[0] if 0 <= i < len(self.chunks)]


def load_local_embedder(model_path):
    """Refuse Hub IDs/downloads; load only a caller-supplied local directory."""
    path = Path(model_path).resolve()
    if not path.is_dir() or not (path / "modules.json").is_file():
        raise ValueError("Provide an existing complete local SentenceTransformer directory")
    # Set before importing transformers. This CLI is separate from Flask.
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    os.environ["USE_TF"] = "0"
    os.environ["TRANSFORMERS_NO_TF"] = "1"
    os.environ["TRANSFORMERS_NO_FLAX"] = "1"
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(str(path), local_files_only=True)


def summarize(results, reviews=()):
    """Match human records to the exact saved answer and ordered context IDs."""
    by_id = {result["question_id"]: result for result in results}
    if len(by_id) != len(results):
        raise ValueError("Duplicate result question ID")
    seen = set()
    for review in reviews:
        if not review:
            continue
        question_id = review.get("question_id")
        if question_id not in by_id or question_id in seen:
            raise ValueError("Unknown or duplicate review question ID")
        seen.add(question_id)
        result = by_id[question_id]
        ids = [chunk["chunk_id"] for chunk in result["retrieved"]]
        if review.get("answer") != result["answer"] or review.get("retrieved_chunk_ids") != ids:
            raise ValueError("Stale review: answer or retrieved context changed")
        if result["answer"] is None and any(
                review.get(key) not in (None, "Unreviewed")
                for key in ("accuracy", "grounding")):
            raise ValueError("Cannot review a missing answer")
    summary = {
        "questions": len(results),
        "retrieval_scored": sum(r["hit_at_k"] is not None for r in results),
        "hit_at_k": mean_score(r["hit_at_k"] for r in results),
        "mrr_at_k": mean_score(r["reciprocal_rank_at_k"] for r in results),
        "coverage_scored": sum(r["expected_fact_coverage"] is not None for r in results),
        "expected_fact_coverage": mean_score(r["expected_fact_coverage"] for r in results),
    }
    for k in (1, 3, 5):
        hits = []
        ranks = []
        for result in results:
            expected = result.get("expected_document_ids")
            documents = [chunk["document_id"] for chunk in result.get("retrieved", [])]
            hits.append(hit_at_k(documents, expected, k))
            if k == 5:
                ranks.append(reciprocal_rank(documents, expected, k))
        summary[f"hit_at_{k}"] = mean_score(hits)
        if k == 5:
            summary["mrr_at_5"] = mean_score(ranks)
    summary.update(score_reviews(reviews))
    return summary


def run_evaluation(questions, retriever, k=5, answerer=None):
    """Optional answerer(question, context) is caller-owned; no LLM calls here."""
    validate_k(k)
    results, template = [], []
    for item in questions:
        chunks = retriever.retrieve(item["question"], k)
        context = "\n\n".join(chunk.text for chunk in chunks)
        answer = answerer(item["question"], context) if answerer else None
        documents = [chunk.document_id for chunk in chunks]
        results.append({
            "question_id": item["id"], "question": item["question"],
            "expected_document_ids": item.get("expected_document_ids"),
            "reference_answer": item["reference_answer"],
            "expected_facts": item.get("expected_facts"),
            "sources": item.get("sources", []), "retrieved": [asdict(c) for c in chunks],
            "answer": answer,
            "hit_at_k": hit_at_k(documents, item.get("expected_document_ids"), k),
            "reciprocal_rank_at_k": reciprocal_rank(
                documents, item.get("expected_document_ids"), k),
            "expected_fact_coverage": expected_fact_coverage(answer, item.get("expected_facts")),
        })
        template.append({
            "question_id": item["id"], "answer": answer,
            "retrieved_chunk_ids": [chunk.chunk_id for chunk in chunks],
            "accuracy": "Unreviewed", "grounding": "Unreviewed",
            "reviewer": "", "notes": "",
        })
    return {"k": k, "results": results, "review_template": template,
            "summary": summarize(results)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-path", type=Path,
                        help="Local SentenceTransformer; use all-MiniLM-L6-v2 for parity")
    parser.add_argument("--answers", type=Path, help="Saved answer strings keyed by question ID")
    parser.add_argument("--score", type=Path, help="Score a saved run without loading models")
    parser.add_argument("--reviews", type=Path, help="JSON list of manual review records")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--k", type=int, default=5)
    args = parser.parse_args(argv)
    try:
        validate_k(args.k)
        if args.score:
            if args.model_path or args.answers:
                parser.error("--score cannot be combined with retrieval options")
            run = json.loads(args.score.read_text(encoding="utf-8"))
            reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else []
            run["summary"] = summarize(run["results"], reviews)
            run["reviews"] = reviews
        else:
            if args.model_path is None or args.reviews:
                parser.error("Retrieval requires --model-path; reviews require --score")
            dataset = load_dataset()
            answers = json.loads(args.answers.read_text(encoding="utf-8")) if args.answers else {}
            known_ids = {item["id"] for item in dataset["questions"]}
            if (not isinstance(answers, dict) or set(answers) - known_ids
                    or any(not isinstance(answer, str) for answer in answers.values())):
                raise ValueError("Answers must map known question IDs to strings")
            embedder = load_local_embedder(args.model_path)
            chunks = build_chunks(ROOT / "examples" / "documents", dataset["documents"])
            retriever = OfflineRetriever(chunks, embedder)
            # Reuse the callback path while looking up saved answers by question ID.
            by_question = {item["question"]: answers.get(item["id"])
                           for item in dataset["questions"]}
            run = run_evaluation(dataset["questions"], retriever, args.k,
                                 lambda question, context: by_question[question])
            run["corpus_sha256"] = {key: doc["sha256"] for key, doc in dataset["documents"].items()}
            run["embedding_model"] = {
                "directory": str(args.model_path.resolve()),
                "identity_verified": False,
                "source": "Caller-supplied local SentenceTransformer directory",
                "required_for_production_parity": "all-MiniLM-L6-v2",
            }
            run["index"] = "FAISS IndexFlatL2; unnormalized; all bundled PDFs"
        args.output.write_text(json.dumps(run, indent=2, ensure_ascii=False), encoding="utf-8")
        print("Evaluation output saved. See file for scores and review denominators.")
    except (OSError, ValueError, KeyError, TypeError) as error:
        # Do not echo external exception text, which might contain local secrets.
        parser.error(f"Evaluation failed ({type(error).__name__}); check inputs and local model")


if __name__ == "__main__":
    main()
