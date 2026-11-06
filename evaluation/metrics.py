"""Pure metrics: retrieval, phrase coverage, and separate human review scores."""

import re


def validate_k(k):
    if isinstance(k, bool) or not isinstance(k, int) or k < 0:
        raise ValueError("K must be a non-negative integer")


def reciprocal_rank(retrieved_document_ids, expected_document_ids, k):
    """First relevant chunk rank within K; None means no relevance labels."""
    validate_k(k)
    if not expected_document_ids:
        return None
    expected = set(expected_document_ids)
    for rank, document_id in enumerate((retrieved_document_ids or [])[:k], 1):
        if document_id in expected:
            return 1 / rank
    return 0.0


def hit_at_k(retrieved_document_ids, expected_document_ids, k):
    rank = reciprocal_rank(retrieved_document_ids, expected_document_ids, k)
    return None if rank is None else float(rank > 0)


def mean_score(values):
    """Exclude missing scores; an empty population is unscored, not zero."""
    scored = [value for value in values if value is not None]
    return sum(scored) / len(scored) if scored else None


def _normalize(text):
    # Preserve decimal tokens; remove numeric thousands separators.
    text = re.sub(r"(?<=\d),(?=\d)", "", text.casefold())
    return " ".join(re.findall(r"\w+(?:\.\d+)*", text))


def expected_fact_coverage(answer, facts):
    """Literal phrase coverage only. Does NOT detect truth, negation or support."""
    if not facts:
        return None
    phrases_by_fact = []
    for fact in facts:
        phrases = fact.get("accepted_phrases", [])
        if (not isinstance(phrases, list) or not phrases
                or any(not isinstance(p, str) or not _normalize(p) for p in phrases)):
            raise ValueError("Every fact needs nonempty accepted_phrases")
        phrases_by_fact.append(phrases)
    if answer is None:
        return None
    if not isinstance(answer, str):
        raise ValueError("Answer must be a string or None")
    normalized = f" {_normalize(answer)} "
    covered = sum(any(f" {_normalize(p)} " in normalized for p in phrases)
                  for phrases in phrases_by_fact)
    return covered / len(facts)


def score_reviews(records):
    """Score accuracy and grounding independently, excluding unreviewed."""
    accuracy_weights = {
        "Correct": 1.0,
        "Partially correct": 0.5,
        "Incorrect": 0.0,
        "Not enough evidence": 0.0,
    }
    grounding_labels = {"Supported", "Partially supported", "Unsupported"}
    strict, partial, grounding = [], [], []
    for record in records:
        if record is None:
            continue
        accuracy = record.get("accuracy")
        support = record.get("grounding")
        if accuracy not in (None, "Unreviewed") and accuracy not in accuracy_weights:
            raise ValueError("Unknown accuracy label")
        if support not in (None, "Unreviewed") and support not in grounding_labels:
            raise ValueError("Unknown grounding label")
        reviewed = accuracy in accuracy_weights or support in grounding_labels
        if reviewed and any(not isinstance(record.get(field), str)
                            or not record[field].strip()
                            for field in ("reviewer", "notes")):
            raise ValueError("Reviewed records require reviewer and notes")
        if accuracy in accuracy_weights:
            strict.append(float(accuracy == "Correct"))
            partial.append(accuracy_weights[accuracy])
        if support in grounding_labels:
            grounding.append(float(support == "Supported"))

    def percent(values):
        score = mean_score(values)
        return None if score is None else 100 * score

    return {
        "accuracy_reviewed": len(strict),
        "strict_accuracy_percent": percent(strict),
        "partial_credit_percent": percent(partial),
        "grounding_reviewed": len(grounding),
        "supported_percent": percent(grounding),
    }
