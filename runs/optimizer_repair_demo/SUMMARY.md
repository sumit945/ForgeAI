# Optimizer repair demo (fault injection)

Start: the ship-spares agent with top_k=1, 40-word chunks, no refusal/scope/exactness/combine rules, temperature 0.7.
Same 23 eval cases and documents as `ship_spares_optimizer_demo`. Optimizer stop reason: **no applicable targeted fix remains**.

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


## Change log

### Iteration 1 - accepted: dev pass-rate improved
Dev 50% -> 81%; hold-out 43% -> 57%. Failures before: {'retrieval_miss': 7, 'prompt_defect': 1}. Now passing: ['amb-01', 'amb-02', 'ans-03', 'ans-04', 'ans-06', 'ans-09', 'mul-01']. Newly failing: ['ans-08'].
- **set_param** `knowledge.retrieval.top_k` (retrieval_miss): 1 -> 3  
  why: evidence ranked [2, 2, 2, 2, 2, 2, 3] but only top 1 are passed to the model  cases: ['ans-03', 'ans-04', 'ans-06', 'ans-09', 'amb-02', 'mul-01', 'mul-02']
- **add_prompt_rule** `system_prompt.rules[fix-prompt_defect-5]` (prompt_defect): None -> Include all required facts and citations from the retrieved passage to ensure the answer is comprehensive.  
  why: 1 case(s): the assistant had the right context but omitted required facts, ignored scope, or skipped citations  cases: ['amb-03']

### Iteration 2 - REVERTED: dev pass-rate 81% -> 81% (no strict improvement)
Dev 81% -> 81%; hold-out 57% -> 43%. Failures before: {'prompt_defect': 3}. Now passing: []. Newly failing: ['ans-05'].
- **add_prompt_rule** `system_prompt.rules[fix-prompt_defect-6]` (prompt_defect): None -> Include every specific number, date, condition and exception from the relevant excerpts, and always cite the source file names.  
  why: 3 case(s): the assistant had the right context but omitted required facts, ignored scope, or skipped citations  cases: ['ans-08', 'amb-03', 'mul-02']

## Still failing after the repair
- **ans-02** [prompt_defect] What is the unit price for the safety category part with the lowest unit price in USD? - passage was retrieved and answer is grounded, but required facts/citation are missing
- **ans-07** [retrieval_miss] What is the part number for the Engine category with the highest criticality class and a unit price less than  - answer lacks the expected facts and the needed passage was not retrieved
- **ans-08** [prompt_defect] What is the freight cost for orders above 10,000 USD? - passage was retrieved and answer is grounded, but required facts/citation are missing
- **amb-03** [prompt_defect] ... Whats the max time to check delivery and report probs like no parts or broke stuff? - passage was retrieved and answer is grounded, but required facts/citation are missing
- **mul-02** [prompt_defect] What is the part number for the Engine category with a criticality class of A and a unit price of $412.50? Als - passage was retrieved and answer is grounded, but required facts/citation are missing
- **mul-03** [retrieval_miss] What is the part number for the Engine category with the highest criticality class and a unit price less than  - answer lacks the expected facts and the needed passage was not retrieved
