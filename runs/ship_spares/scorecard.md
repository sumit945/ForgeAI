# Scorecard

## Per iteration (pass = correct AND grounded AND refused-when-it-should)
| Iteration | Outcome | Dev | Hold-out | Overall | Note |
|---|---|---|---|---|---|
| 0 (baseline) | - | 93% | 100% | 95% | generated agent, untouched |

Dev = the split the optimizer sees. Hold-out = never used for any decision; a large dev/hold-out gap means the loop overfitted.

## Final metrics (accepted definition)
| Metric | Overall | Dev | Hold-out |
|---|---|---|---|
| Cases | 21 | 15 | 6 |
| Pass rate | 95% | 93% | 100% |
| Correct (answerable) | 100% | 100% | 100% |
| Grounded (answerable) | 93% | 90% | 100% |
| Refusal recall (should refuse -> refused) | 100% | 100% | 100% |
| Wrong-refusal rate (answerable -> refused) | 0% | 0% | 0% |
| Retrieval hit rate | 100% | 100% | 100% |
| Model broke output contract (before code-side repair) | 86% | 87% | 83% |

## By category
| Category | n | Pass rate |
|---|---|---|
| ambiguous | 2 | 50% |
| answerable | 9 | 100% |
| multi_doc | 3 | 100% |
| near_miss | 4 | 100% |
| off_topic | 3 | 100% |
