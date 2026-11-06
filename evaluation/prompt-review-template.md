# Manual factual accuracy and grounding review

Review the exact baseline and candidate answers in
[`prompt-comparison.json`](./prompt-comparison.json). For each answer:

- **Accuracy:** compare with the reference answer and expected facts.
- **Grounding:** check every material claim against that question's retrieved
  context in the same JSON file.
- Leave a field `Unreviewed` until a person has performed that review. Expected
  fact coverage is automated phrase matching and is not a manual score.

| Question ID | Baseline accuracy | Baseline grounding | Candidate accuracy | Candidate grounding | Reviewer / notes |
|---|---|---|---|---|---|
| `bc-road-length` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `bc-unpaved-share` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `bc-roadless-definition` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `ontario-fatalities-change` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `ontario-fatality-rate` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `ontario-inattentive-driving` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `canada-strategy-slogan` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `canada-occupant-protection` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `canada-seat-belt-benefits` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |
| `transport-2025-source` | Unreviewed | Unreviewed | Unreviewed | Unreviewed | |

Use the accepted label spellings in the evaluation README. Keep detailed labels,
reviewer identity and evidence notes in the JSON review format when scoring;
this table is a human-readable checklist, not a scored result.
