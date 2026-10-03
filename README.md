# rag-eval-harness

An evaluation and regression-testing harness for retrieval-augmented generation.

Most RAG projects can tell you what their system does. Far fewer can tell you
whether last week's prompt change made it better or worse, and fewer still can
say so with a confidence interval. This repository is the measurement layer: a
labelled golden set, retrieval and answer metrics with bootstrap intervals,
paired significance testing between configurations, a calibrated automatic
judge, and a CI gate that fails a pull request when quality regresses.

The RAG pipeline in here exists to be measured. The measurement is the project.

Everything runs offline, deterministically, with no API key and no model
download, so `git clone && pip install -e . && rageval run` reproduces every
number below exactly.

---

## Quick start

```bash
pip install -e ".[dev]"

rageval validate configs/bm25-fixed.yaml     # check every gold quote resolves
rageval run configs/bm25-fixed.yaml          # evaluate one pipeline
rageval compare runs/bm25-fixed.json runs/hybrid-rrf.json   # paired A/B test
rageval calibrate-judge configs/bm25-fixed.yaml             # judge vs humans
rageval gate runs/candidate.json --baseline baselines/bm25-fixed.json
```

## What is measured

**Retrieval.** `recall@k`, `precision@k`, `nDCG@k`, `hit_rate@k` and `MRR`,
plus budget-relative variants described below.

**Attribution.** `citation_precision` and `citation_recall` against the gold
evidence, and `groundedness`, the share of the answer's tokens that appear in
the passages it cites.

**Answers.** Token `answer_f1`, `exact_match`, `evidence_coverage` (does the
answer actually reproduce the operative detail, such as the number of days in a
deadline), `abstention_correct`, and `judge_correct` from the automatic judge.

**Cost.** Per-query latency.

Every aggregate carries a 2000-sample bootstrap confidence interval, and every
A/B claim goes through a paired sign-flip permutation test. On a golden set of
this size, metric differences are routinely smaller than the sampling noise, so
a point estimate on its own is not evidence.

## Baseline results

The shipped baseline is `bm25-fixed` over the Aurora Health golden set: 10
policy documents, 65 questions, 60 answerable and 5 that should be refused.

| metric | value | 95% CI |
| --- | ---: | :---: |
| recall@budget | 0.767 | [0.669, 0.858] |
| ndcg@budget | 0.831 | [0.733, 0.918] |
| mrr | 0.885 | [0.811, 0.948] |
| citation_precision | 0.783 | [0.683, 0.875] |
| judge_correct | 0.833 | [0.733, 0.917] |
| latency_ms | 3.0 | [1.8, 3.6] |

## Findings

### Comparing chunkings at a fixed `k` is not a fair comparison

The first version of this ablation compared every pipeline at `top_k=5`. Fixed
windows produce 30 chunks averaging 982 characters; section chunking produces 66
averaging 359. At `k=5` the coarse chunker was handed roughly 2.7x more text and
unsurprisingly looked better. The conclusion was an artefact of the measurement.

The harness therefore supports a **context budget**: every pipeline shows the
generator the same 2000 characters, and `recall@budget` / `ndcg@budget` score
whatever fits. That is the comparison a production system actually faces, since
the constraint is the context window, not the number of chunks. Changing to a
budget reversed the ranking on retrieval recall.

### Better retrieval produced worse answers

| pipeline | recall@budget | ndcg@budget | citation_precision | judge_correct | latency_ms |
| --- | ---: | ---: | ---: | ---: | ---: |
| bm25-fixed | 0.767 | 0.831 | **0.783** | **0.833** | 3.0 |
| bm25-section | **0.933** | **0.870** | 0.400 | 0.700 | 4.1 |
| dense-section | 0.933 | 0.789 | 0.386 | 0.717 | 3.9 |
| hybrid-weighted | 0.933 | 0.868 | 0.386 | 0.700 | 8.8 |
| hybrid-rrf | 0.933 | 0.860 | 0.392 | 0.700 | 7.8 |
| hybrid-rrf-mmr | 0.925 | 0.779 | 0.394 | 0.733 | 44.3 |

Section chunking beat fixed windows on retrieval by a wide and significant
margin (recall@budget +0.167, 95% CI [+0.092, +0.250], p < 0.001). It also made
the end-to-end system significantly *worse*: citation precision fell 0.383
(p < 0.001) and judged answer correctness fell 0.133 (p = 0.022).

The cause is visible in the answers. The same 2000-character budget holds 1.3
fixed chunks but 4.5 section chunks, so the generator receives fragments drawn
from several documents instead of one coherent passage. In a corpus where the
Essential and Premier plans state the same rule with different numbers, that
produces answers like "Generic medicines are covered in full. Generic medicines
are covered at 90 per cent" — both sentences true, neither attributed, and the
reader cannot tell which plan applies.

This is the finding the project exists to produce. Optimising the retrieval
metric in isolation would have shipped a measurably worse product.

### Hybrid retrieval did not pay for itself

Against `bm25-section`, reciprocal-rank fusion of lexical and dense arms changed
nothing measurable (recall@budget and judge_correct identical, nDCG@budget
-0.009, p = 0.252) while costing an extra 10.1 ms per query (p = 0.002). Adding
MMR reranking on top cost 44 ms per query and significantly *reduced* nDCG.

This is corpus-specific and should not be read as a general claim. These are
short, lexically dense policy clauses where the query usually shares exact
terminology with the answer, which is close to the best case for BM25 and close
to the worst case for a 192-dimensional latent semantic model fitted on 24,000
characters. The point is that the hybrid stack was assumed to help, was
measured, and did not.

### The judge is usable, with a known bias

`rageval calibrate-judge` scores the automatic judge against 60 hand-written
human labels:

```
agreement      0.883
cohen's kappa  0.625
judge says correct 0.833, humans say correct 0.783
false positives 5, false negatives 2
```

κ = 0.625 is substantial agreement, so `judge_correct` is reported as a headline
metric and gated. The errors are not random: all five false positives are
answers that contain the correct clause alongside an unattributed clause from
the other plan contradicting it. The judge rewards vocabulary overlap, and that
failure mode is exactly the one the section-chunking regression produces — so
the true gap between `bm25-fixed` and `bm25-section` is probably wider than the
judge reports, not narrower.

### Two metrics are degenerate and are deliberately not gated

`groundedness` is 1.000 with a zero-width interval, because the extractive
answerer builds its answer out of the sentences it cites; the metric cannot do
anything else. `abstention_correct` is 0.923, which is just the share of
answerable questions, because the answerer never abstains.

Three abstention signals were tried before concluding that: query-term coverage
(unanswerable questions reach 0.667 while answerable ones drop to 0.30),
rarest-missing-term rarity (saturates, because paraphrases introduce
out-of-vocabulary words too), and top-1 BM25 score (unanswerable spans
3.9–10.9, entirely inside the answerable range). No threshold separates the
classes, so a purely lexical answerer cannot calibrate abstention on this
corpus.

Both metrics are implemented and reported because they become informative with
an LLM answerer. Neither is in the gate, because a threshold a system cannot
move is a test that only looks like a test.

## The regression gate

`rageval gate` is what CI runs on every pull request. It checks absolute floors
and movement against the committed baseline in `baselines/`:

```
$ rageval gate runs/dense-section.json --baseline baselines/bm25-fixed.json
gate failed with 6 violation(s)
  x mrr: 0.749 is below the floor of 0.780
  x citation_precision: 0.386 is below the floor of 0.680
  x judge_correct: 0.717 is below the floor of 0.720
  x citation_precision: dropped 0.397 against the baseline (limit 0.050, p=0.000)
  x judge_correct: dropped 0.117 against the baseline (limit 0.060, p=0.038)
  x mrr: dropped 0.136 against the baseline (limit 0.040, p=0.000)
```

A regression must clear both a size limit and a paired significance test before
it breaks the build. Without the significance requirement, a 65-question set
produces enough run-to-run variation to fail at random, and a flaky gate is one
people learn to re-run until it goes green.

## Design decisions

**Gold evidence is labelled as verbatim quotes, not chunk ids.** Quotes are
resolved to character spans at load time and chunk relevance is derived from
span overlap. This is what makes the chunking ablation possible at all: all six
configurations share one set of labels. `rageval validate` rejects a quote that
is missing or that matches a document twice, so a mistyped label fails loudly
instead of becoming a permanent false negative.

**The default stack is offline and deterministic.** The embedder is TF-IDF
followed by truncated SVD fitted on the corpus — real latent semantic indexing,
not a hashing trick — and the answerer is extractive. A regression gate is only
meaningful if identical input produces identical output, and CI should not
depend on a key or a vendor's uptime. OpenAI and sentence-transformers
embedders, an LLM answerer and a cross-encoder reranker are available behind
optional extras and share the same interfaces.

**BM25 precomputes the document side of the formula.** Every factor except the
choice of query terms depends only on the document, so the entire weight matrix
is built once at index time and stored in CSC form. A query becomes a sparse
column gather plus a row sum, and ranking uses `argpartition` rather than a full
sort. A test asserts the vectorised scores match a textbook per-posting
implementation to 1e-4, because an optimisation that changes the numbers is a
bug, not an optimisation.

**Unreachable gold counts as a miss.** If an answerable question's evidence
falls outside every chunk, its retrieval metrics score zero rather than being
skipped. Skipping would let a chunking strategy improve its average by making
questions impossible.

**Metrics are averaged only over the questions where they are defined.**
Retrieval metrics do not apply to unanswerable questions, so they are averaged
over 60 rather than imputed as zero over 65.

## Layout

```
src/rageval/
  corpus/        chunking strategies, quote resolution, dataset validation
  embeddings/    offline TF-IDF+SVD, optional OpenAI and sentence-transformers
  retrieval/     vectorised BM25, dense, RRF and weighted fusion, MMR rerank
  generation/    extractive citing answerer, optional LLM answerer
  metrics/       retrieval, attribution, judge calibration, bootstrap and tests
  harness/       runner, content-addressed cache, reports, regression gate
datasets/aurora-health/   corpus, golden set, human labels, labelling protocol
configs/         one YAML per pipeline, plus the gate thresholds
baselines/       the committed run CI compares against
reports/         generated ablation report
```

## Dataset

Ten synthetic health-insurance policy documents with the properties that make
real policy text hard: cross-references between documents, exceptions that
override the schedule they sit under, and near-identical clauses that differ
only by plan. A corpus where every answer sits in its own tidy paragraph makes
every retriever look good.

See [`datasets/aurora-health/README.md`](datasets/aurora-health/README.md) for
the schema, the human-labelling protocol and how to extend the set.

## Limitations

The golden set is 65 questions on one synthetic corpus, so the confidence
intervals here are wide and the conclusions are about this corpus, not about RAG
in general. `groundedness` is lexical overlap, not claim-level entailment: it
catches an answer drifting off its sources but not a fluent contradiction built
from the source's own vocabulary. The human labels are single-annotator, so
there is no inter-annotator agreement figure to bound the judge's ceiling.

## License

MIT.
