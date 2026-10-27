# PREREG v1 (retrospective — thresholds were not frozen before runs; documented post-hoc)

## Phase 3 (Coding Task) gates
- G1 (prediction control): <= 0.10
- G2 (oracle): >= 0.90 per rung
- G3 (memON): >= 0.8 * oracle
- G4 (delta): memON - memOFF >= 0.40
- G5 (memOFF leak): leak = 0

## Phase 4 (Sleep) gates (v1 — FAILED)
- train memOFF exact >= 0.6 (80 facts, 500 steps, 3 seeds)
- paraphrase exact >= 0.4
- PPL ratio <= 1.15
- heldout_new memON exact drop <= 0.05

## Phase 5 (Scale) gates
- N=100: lexical recall@1 >= 0.90, memON >= 0.80, near_miss twin accuracy >= 0.80

NOTE: v1 sleep gate failed. See PREREG_v2.md for revised sleep thresholds.
