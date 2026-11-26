# Scorecard

## Per iteration (pass = correct AND grounded AND refused-when-it-should)
| Iteration | Outcome | Dev | Hold-out | Overall | Note |
|---|---|---|---|---|---|
| 0 (baseline) | - | 81% | 67% | 77% | generated agent, untouched |
| 1 | accepted | 88% | 67% | 82% | 1 change(s), fixed 1, regressed 0 |
| 2 | reverted | 81% | 67% | 77% | 1 change(s), fixed 0, regressed 1 |

Dev = the split the optimizer sees. Hold-out = never used for any decision; a large dev/hold-out gap means the loop overfitted.

## Final metrics (accepted definition)
| Metric | Overall | Dev | Hold-out |
|---|---|---|---|
| Cases | 22 | 16 | 6 |
| Pass rate | 82% | 88% | 67% |
| Correct (answerable) | 75% | 82% | 60% |
| Grounded (answerable) | 94% | 100% | 80% |
| Refusal recall (should refuse -> refused) | 100% | 100% | 100% |
| Wrong-refusal rate (answerable -> refused) | 0% | 0% | 0% |
| Retrieval hit rate | 100% | 100% | 100% |
| Model broke output contract (before code-side repair) | 82% | 75% | 100% |

## By category
| Category | n | Pass rate |
|---|---|---|
| ambiguous | 4 | 100% |
| answerable | 9 | 78% |
| multi_doc | 3 | 33% |
| near_miss | 2 | 100% |
| off_topic | 4 | 100% |
