# PREREG v2 — Sleep-to-LoRA (revised after Phase 4 diagnosis)

## Motivation
Phase 4 v1 failed: 0.6B model cannot memorize 80 facts in 500 steps (capacity ~20 facts @ 500 steps/fact).
Diagnosis: training-budget issue, not code bug. v1 result (FAIL) remains on record.

## Revised thresholds
- Step 1 (capacity proof): 5 facts, 500 steps/fact (2500 total), 1 seed. Pass: train memOFF exact >= 0.6.
- Step 2 (scale): 20 facts, 500 steps/fact (10000 total), 3 seeds. Pass: train memOFF exact >= 0.6, paraphrase >= 0.4.
- Step 3 (retention): PPL ratio <= 1.15, heldout_new memON drop <= 0.05.

## Constraints
- No threshold/scorer/producer/family changes after running.
- v1 result (FAIL) remains on record; v2 results reported separately.
- max_new_tokens for eval: 8 (scorer uses exact_match startswith).
