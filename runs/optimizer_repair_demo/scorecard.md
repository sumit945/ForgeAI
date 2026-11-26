# Scorecard

## Per iteration (pass = correct AND grounded AND refused-when-it-should)
| Iteration | Outcome | Dev | Hold-out | Overall | Note |
|---|---|---|---|---|---|
| 0 (baseline) | - | 50% | 43% | 48% | generated agent, untouched |
| 1 | accepted | 81% | 57% | 74% | 2 change(s), fixed 7, regressed 1 |
| 2 | reverted | 81% | 43% | 70% | 1 change(s), fixed 0, regressed 1 |

Dev = the split the optimizer sees. Hold-out = never used for any decision; a large dev/hold-out gap means the loop overfitted.

## Final metrics (accepted definition)
| Metric | Overall | Dev | Hold-out |
|---|---|---|---|
| Cases | 23 | 16 | 7 |
| Pass rate | 74% | 81% | 57% |
| Correct (answerable) | 67% | 80% | 40% |
| Grounded (answerable) | 93% | 90% | 100% |
| Refusal recall (should refuse -> refused) | 100% | 100% | 100% |
| Wrong-refusal rate (answerable -> refused) | 0% | 0% | 0% |
| Retrieval hit rate | 87% | 100% | 60% |
| Model broke output contract (before code-side repair) | 70% | 62% | 86% |

## By category
| Category | n | Pass rate |
|---|---|---|
| ambiguous | 3 | 67% |
| answerable | 9 | 67% |
| multi_doc | 3 | 33% |
| near_miss | 4 | 100% |
| off_topic | 4 | 100% |
