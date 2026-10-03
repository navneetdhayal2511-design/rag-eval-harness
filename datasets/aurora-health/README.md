# Aurora Health — golden set

A synthetic health-insurance policy corpus written for this harness. The text is
invented, so there is no licensing or privacy question, but it is shaped like
real policy wording: cross-references between documents, exceptions that
override the schedule they sit under, near-identical clauses that differ only by
plan, and figures that are meaningless without the heading above them.

That last property is the point. A corpus where every answer sits in its own
paragraph makes every retriever look good.

- 10 documents, 65 questions (60 answerable, 5 unanswerable).
- Categories include single-clause lookups, conflict resolution between a plan
  schedule and the exclusions document, and multi-document questions whose
  answer needs two sources.

## Question schema

```json
{
  "qid": "q001",
  "question": "What is the annual benefit maximum per person under the Essential plan?",
  "reference_answer": "500,000 units per insured person per policy year.",
  "category": "limits",
  "unanswerable": false,
  "evidence": [{ "doc_id": "plan-essential", "quote": "an annual benefit maximum of 500,000 units ..." }]
}
```

Gold evidence is a **verbatim quote**, not a chunk id. The harness resolves each
quote to a character span at load time and derives chunk relevance from span
overlap. This is why the chunking strategy can change without the labels being
redone, and it is why `rageval validate` rejects a quote that is missing or that
matches a document in more than one place.

Questions with `"unanswerable": true` carry no evidence. The correct behaviour is
to abstain, and they exist so that a system cannot score well by answering
everything confidently.

## Human labels

`human_labels.jsonl` holds correctness labels for the `bm25-fixed` run, used by
`rageval calibrate-judge` to measure how far the automatic judge can be trusted.

The rule applied when labelling was: **a label of 1 means a reader who saw only
this answer would come away with the right answer to the question asked.** An
answer that contains the correct clause but also an unattributed clause
contradicting it — common here, because many rules differ between the Essential
and Premier plans — is labelled 0, since the reader cannot tell which applies.

Labels are tied to a specific run. Changing the answerer invalidates them, and
the judge must be recalibrated.

## Extending the set

Adding a question means adding a row to `questions.jsonl` and running
`rageval validate configs/hybrid-rrf.yaml`. Validation fails loudly on a quote
that does not resolve, so a mistyped label cannot silently become a permanent
false negative in every future run.
