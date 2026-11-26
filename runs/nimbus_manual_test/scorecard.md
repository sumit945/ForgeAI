# Scorecard

## Per iteration (pass = correct AND grounded AND refused-when-it-should)
| Iteration | Outcome | Dev | Hold-out | Overall | Note |
|---|---|---|---|---|---|
| 0 (baseline) | - | 93% | 83% | 90% | generated agent, untouched |

Dev = the split the optimizer sees. Hold-out = never used for any decision; a large dev/hold-out gap means the loop overfitted.

## Final metrics (accepted definition)
| Metric | Overall | Dev | Hold-out |
|---|---|---|---|
| Cases | 21 | 15 | 6 |
| Pass rate | 90% | 93% | 83% |
| Correct (answerable) | 87% | 90% | 80% |
| Grounded (answerable) | 100% | 100% | 100% |
| Refusal recall (should refuse -> refused) | 100% | 100% | 100% |
| Wrong-refusal rate (answerable -> refused) | 0% | 0% | 0% |
| Retrieval hit rate | 100% | 100% | 100% |
| Model broke output contract (before code-side repair) | 76% | 67% | 100% |

## By category
| Category | n | Pass rate |
|---|---|---|
| answerable | 12 | 92% |
| multi_doc | 3 | 67% |
| near_miss | 2 | 100% |
| off_topic | 4 | 100% |
