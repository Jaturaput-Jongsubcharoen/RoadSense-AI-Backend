"""Compare two RAG prompts using a saved retrieval run."""

import argparse
import json
from pathlib import Path
import time

from evaluation.metrics import expected_fact_coverage, mean_score, score_reviews
from services.llm_config import OLLAMA_MODEL
from services.ollama_service import estimate_tokens, request_ollama
from services.rag_service import build_rag_prompt

BASELINE_VERSION = "production-rag-v1"
CANDIDATE_VERSION = "candidate-context-focused-v1"
CANDIDATE_TEMPLATE = """You are a careful road-safety assistant.
Answer the question directly using only the supplied context.
Keep relevant dates, numbers, units, and attribution accurate.
Do not add facts that are not stated in the context.
If the context does not contain enough information, reply exactly:
"I don't have enough information in the uploaded document to answer that."

CONTEXT:
{context}

QUESTION:
{question}
"""
GENERATION_OPTIONS = {"temperature": 0.1}


def build_candidate_prompt(question, context):
    return CANDIDATE_TEMPLATE.format(question=question, context=context)


def _token_count(value):
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return None


def _generate(prompt, timeout, request_function):
    started_at = time.monotonic()
    data = request_function(prompt, timeout=timeout, options=GENERATION_OPTIONS)
    latency_ms = round((time.monotonic() - started_at) * 1000, 2)

    if data is None:
        return {
            "status": "failed",
            "error_category": "request_failed",
            "answer": None,
            "latency_ms": latency_ms,
            "prompt_tokens": None,
            "completion_tokens": None,
            "total_tokens": None,
            "token_source": "unavailable",
        }

    answer = data.get("response")
    prompt_tokens = _token_count(data.get("prompt_eval_count"))
    completion_tokens = _token_count(data.get("eval_count"))
    prompt_source = "provider" if prompt_tokens is not None else "estimate"
    completion_source = "provider" if completion_tokens is not None else "unavailable"
    if prompt_tokens is None:
        prompt_tokens = estimate_tokens(prompt)
    if completion_tokens is None and isinstance(answer, str):
        completion_tokens = estimate_tokens(answer)
        completion_source = "estimate"

    total_tokens = None
    if completion_tokens is not None:
        total_tokens = prompt_tokens + completion_tokens

    token_source = (
        prompt_source if prompt_source == completion_source
        else f"{prompt_source}+{completion_source}"
    )
    return {
        "status": "success",
        "error_category": None,
        "answer": answer,
        "latency_ms": latency_ms,
        "prompt_tokens": prompt_tokens,
        "completion_tokens": completion_tokens,
        "total_tokens": total_tokens,
        "token_source": token_source,
    }


def _review_fields(review, answer, chunk_ids):
    review = review or {}
    accuracy = review.get("accuracy", "Unreviewed")
    grounding = review.get("grounding", "Unreviewed")
    reviewed = accuracy != "Unreviewed" or grounding != "Unreviewed"
    if reviewed and (
        review.get("answer") != answer
        or review.get("retrieved_chunk_ids") != chunk_ids
    ):
        raise ValueError("Manual review must match the saved answer and retrieved chunks")
    return {
        "answer": answer,
        "retrieved_chunk_ids": chunk_ids,
        "accuracy": accuracy,
        "grounding": grounding,
        "reviewer": review.get("reviewer", ""),
        "notes": review.get("notes", ""),
    }


def _score_prompt(records, reviews, prompt_name):
    for result in records:
        answer = result[prompt_name]["answer"]
        if answer is None:
            result[f"{prompt_name}_expected_fact_coverage"] = None
        else:
            result[f"{prompt_name}_expected_fact_coverage"] = expected_fact_coverage(
                answer, result["expected_facts"]
            )
    review_scores = score_reviews([
        {
            "accuracy": item["accuracy"],
            "grounding": item["grounding"],
            "reviewer": item["reviewer"],
            "notes": item["notes"],
        }
        for item in reviews
    ])
    outcomes = [result[prompt_name] for result in records]
    successful = [item for item in outcomes if item["status"] == "success"]
    failed = [item for item in outcomes if item["status"] == "failed"]
    return {
        "requests_succeeded": sum(item["status"] == "success" for item in outcomes),
        "requests_failed": sum(item["status"] == "failed" for item in outcomes),
        "expected_fact_coverage": mean_score(
            result[f"{prompt_name}_expected_fact_coverage"] for result in records
        ),
        "average_latency_ms": mean_score(item["latency_ms"] for item in successful),
        "average_failed_request_latency_ms": mean_score(
            item["latency_ms"] for item in failed
        ),
        "average_prompt_tokens": mean_score(item["prompt_tokens"] for item in outcomes),
        "average_completion_tokens": mean_score(
            item["completion_tokens"] for item in outcomes
        ),
        "average_total_tokens": mean_score(item["total_tokens"] for item in outcomes),
        "average_actual_provider_total_tokens": mean_score(
            item["total_tokens"] if item["token_source"] == "provider" else None
            for item in outcomes
        ),
        "accuracy_reviewed": review_scores["accuracy_reviewed"],
        "strict_accuracy_percent": review_scores["strict_accuracy_percent"],
        "partial_credit_percent": review_scores["partial_credit_percent"],
        "grounding_reviewed": review_scores["grounding_reviewed"],
        "supported_percent": review_scores["supported_percent"],
    }


def run_comparison(retrieval_run, reviews=None, request_function=None,
                  model=OLLAMA_MODEL, timeout=180):
    """Run both prompt versions against exactly the same saved context."""
    request_function = request_function or request_ollama
    review_map = reviews or {}
    if not isinstance(review_map, dict):
        raise ValueError("Reviews must be a JSON object keyed by question ID")
    records = []
    baseline_reviews = []
    candidate_reviews = []

    known_question_ids = {result["question_id"] for result in retrieval_run["results"]}
    if set(review_map) - known_question_ids:
        raise ValueError("Reviews contain unknown question IDs")

    for result in retrieval_run["results"]:
        question_id = result["question_id"]
        context = "\n\n".join(chunk["text"] for chunk in result["retrieved"])
        baseline_prompt = build_rag_prompt(result["question"], context)
        candidate_prompt = build_candidate_prompt(result["question"], context)
        supplied = review_map.get(question_id, {})
        chunk_ids = [chunk["chunk_id"] for chunk in result["retrieved"]]

        baseline = _generate(baseline_prompt, timeout, request_function)
        candidate = _generate(candidate_prompt, timeout, request_function)

        common = {
            "question_id": question_id,
            "question": result["question"],
            "reference_answer": result.get("reference_answer"),
            "expected_document_ids": result["expected_document_ids"],
            "expected_facts": result["expected_facts"],
            "retrieved_chunk_ids": chunk_ids,
            "retrieved_context": context,
        }
        baseline_review = _review_fields(
            supplied.get("baseline"), baseline["answer"], chunk_ids
        )
        candidate_review = _review_fields(
            supplied.get("candidate"), candidate["answer"], chunk_ids
        )
        records.append({
            **common,
            "baseline": baseline,
            "baseline_review": baseline_review,
            "candidate": candidate,
            "candidate_review": candidate_review,
        })
        baseline_reviews.append(records[-1]["baseline_review"])
        candidate_reviews.append(records[-1]["candidate_review"])

    return {
        "model": model,
        "generation_options": GENERATION_OPTIONS,
        "timeout_seconds": timeout,
        "prompt_templates": {
            BASELINE_VERSION: "Existing production prompt from build_rag_prompt",
            CANDIDATE_VERSION: CANDIDATE_TEMPLATE,
        },
        "baseline_prompt_version": BASELINE_VERSION,
        "candidate_prompt_version": CANDIDATE_VERSION,
        "retrieval_run_summary": retrieval_run["summary"],
        "embedding_model": retrieval_run.get("embedding_model"),
        "retrieval_index": retrieval_run.get("index"),
        "baseline_summary": _score_prompt(records, baseline_reviews, "baseline"),
        "candidate_summary": _score_prompt(records, candidate_reviews, "candidate"),
        "results": records,
        "manual_review_template": [
            {
                "question_id": result["question_id"],
                "reference_answer": result.get("reference_answer"),
                "retrieved_context": result["retrieved_context"],
                "retrieved_chunk_ids": result["retrieved_chunk_ids"],
                "baseline": {
                    "answer": result["baseline"]["answer"],
                    "retrieved_chunk_ids": result["retrieved_chunk_ids"],
                    "accuracy": "Unreviewed",
                    "grounding": "Unreviewed",
                    "reviewer": "",
                    "notes": "",
                },
                "candidate": {
                    "answer": result["candidate"]["answer"],
                    "retrieved_chunk_ids": result["retrieved_chunk_ids"],
                    "accuracy": "Unreviewed",
                    "grounding": "Unreviewed",
                    "reviewer": "",
                    "notes": "",
                },
            }
            for result in records
        ],
        "limitations": [
            "Manual accuracy and grounding are unscored until a reviewer supplies labels.",
            "Expected-fact coverage is literal phrase matching, not factual accuracy.",
            "Token estimates use whitespace-separated words when provider counts are absent.",
            "Latency is measured sequentially and can vary with model/server load.",
        ],
    }


def render_markdown(report):
    lines = [
        "# Offline RAG Prompt Comparison",
        "",
        f"- Model: `{report['model']}`",
        f"- Baseline: `{report['baseline_prompt_version']}`",
        f"- Candidate: `{report['candidate_prompt_version']}`",
        f"- Generation settings: `{json.dumps(report['generation_options'], sort_keys=True)}`",
        f"- Test questions: {report['retrieval_run_summary'].get('questions', 0)}",
        f"- Retrieval index: `{report.get('retrieval_index') or 'unavailable'}`",
        f"- Embedding model path: "
        f"`{(report.get('embedding_model') or {}).get('directory', 'unavailable')}`",
        "",
        "No prompt is automatically selected or copied into production.",
        "Manual accuracy and grounding remain unscored until answers are reviewed.",
        "",
        "## Retrieval metrics from the shared context run",
        "",
        "| Metric | Measured value |",
        "|---|---:|",
        f"| Hit@1 | {report['retrieval_run_summary'].get('hit_at_1')} |",
        f"| Hit@3 | {report['retrieval_run_summary'].get('hit_at_3')} |",
        f"| Hit@5 | {report['retrieval_run_summary'].get('hit_at_5')} |",
        f"| MRR@5 | {report['retrieval_run_summary'].get('mrr_at_5')} |",
        "",
        "| Metric | Baseline | Candidate |",
        "|---|---:|---:|",
    ]
    metric_labels = (
        ("requests_succeeded", "Successful requests"),
        ("requests_failed", "Failed requests"),
        ("expected_fact_coverage", "Expected-fact coverage (not accuracy)"),
        ("average_actual_provider_total_tokens", "Mean actual provider tokens"),
        ("average_total_tokens", "Mean actual-or-estimated tokens"),
        ("average_latency_ms", "Mean successful response latency (ms)"),
        ("average_failed_request_latency_ms", "Mean failed request duration (ms)"),
        ("strict_accuracy_percent", "Human strict accuracy (%)"),
        ("partial_credit_percent", "Human partial-credit (%)"),
        ("supported_percent", "Human grounding supported (%)"),
    )
    for key, label in metric_labels:
        baseline = report["baseline_summary"].get(key)
        candidate = report["candidate_summary"].get(key)
        lines.append(f"| {label} | {baseline if baseline is not None else 'unavailable'} "
                     f"| {candidate if candidate is not None else 'unavailable'} |")

    lines.extend(["", "## Per-question outcomes", "",
                  "| Question ID | Baseline | Candidate | Fact coverage (B/C) |",
                  "|---|---|---|---:|"])
    for result in report["results"]:
        baseline_status = result["baseline"]["status"]
        candidate_status = result["candidate"]["status"]
        baseline_coverage = result.get("baseline_expected_fact_coverage")
        candidate_coverage = result.get("candidate_expected_fact_coverage")
        coverage = (
            f"{baseline_coverage if baseline_coverage is not None else 'n/a'} / "
            f"{candidate_coverage if candidate_coverage is not None else 'n/a'}"
        )
        lines.append(
            f"| {result['question_id']} | {baseline_status} | "
            f"{candidate_status} | {coverage} |"
        )

    lines.extend(["", "## Prompt versions", "",
                  "### Baseline", "", "Uses the unchanged production RAG prompt.", "",
                  "### Candidate", "", "```text"])
    lines.extend(report["prompt_templates"][CANDIDATE_VERSION].splitlines())
    lines.extend(["```", "", "## Limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    return "\n".join(lines) + "\n"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--retrieval-run", type=Path, required=True)
    parser.add_argument("--reviews", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args(argv)
    retrieval_run = json.loads(args.retrieval_run.read_text(encoding="utf-8"))
    reviews = json.loads(args.reviews.read_text(encoding="utf-8")) if args.reviews else {}
    report = run_comparison(retrieval_run, reviews)
    args.output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    args.markdown.write_text(render_markdown(report), encoding="utf-8")
    print("Prompt comparison written; inspect success counts and review denominators.")


if __name__ == "__main__":
    main()
