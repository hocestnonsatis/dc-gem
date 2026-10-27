# DC-GEM: Final Report — Project Closure

## 1. Executive Summary

DC-GEM (Differentiable Core + Gradient-Free Episodic Memory) is a hybrid CLS architecture
combining a pre-trained transformer core (Qwen3-0.6B) with a gradient-free episodic memory (GEM).
The project demonstrates that **GEM enables fact recall that the core cannot achieve alone**,
without modifying core weights (PPL ratio = 1.00, zero catastrophic forgetting).

**Key results:**
- Fact-recall (N=20, 3 query variants): memON = 1.00, memOFF = 0.00 (gap = 1.00)
- Retrieval scaling: N=10 → 1.00, N=100 → 0.98, N=1000 → 0.915 (hybrid, k=16 rerank)
- Semantic retrieval: bge-small 0.84 vs core mean-pool 0.04 (chance 0.02) at N=50
- Coding task (T1, 3 families): oracle 1.0, memON 1.0, memOFF 0.0, delta 1.0
- Sleep: prevented forgetting (PPL ~1.00) but failed to learn facts (train 0.0-0.1)

**Production configuration: GEM memON (Sleep Disabled).** Sleep module retained as
experimental/future work for larger base models or isolated adapter routing.

## 2. Gate Summary and Evidence Map

| Phase | Gate | Status | Evidence |
|---|---|---|---|
| 0 Repo hygiene | tests PASS, tag, env lock | ✅ PASS | tests/test_memory.py (4 PASS), ENV.md, requirements.lock |
| 1 Architecture | unit tests + regression | ✅ PASS | dc_gem/{config,core,embeddings,memory,retrieval,inject,sleep}.py |
| 2 Valid measurement | G1 (0.0≤0.10), leak=0, memOFF classification | ✅ PASS | runs/evidence/py/g1.json, phase2.json |
| 3 Coding T1 | G1-G5 (lexical, near_miss) | ✅ PASS | runs/evidence/py/phase3_t1.json |
| 4 Sleep v1 | train memOFF ≥0.6 (80 facts) | ❌ FAIL | runs/evidence/py/phase4_sleep.json |
| 4 Sleep v2 Step 1 | 5 facts, 500 steps/fact | ✅ PASS | runs/evidence/py/phase4_v2_step1.json |
| 4 Sleep v2 Step 2 | 20 facts, PPL ≤1.15 | ❌ FAIL | runs/evidence/py/phase4_v2_step2.json |
| 4 Sleep v3 (LoRA) | 20 facts, PPL ≤1.15 + train ≥0.6 | ❌ FAIL | runs/evidence/py/phase4_v3_lora.json |
| 5 Scale | retrieval + semantic ablasyon | ✅ PASS | runs/fact_recall_proof.json, runs/evidence/py/semantic_ablation.json |

### Evidence files
- `runs/evidence/py/g1.json` — prediction control (0/30 = 0.000)
- `runs/evidence/py/phase2.json` — memOFF classification, leak check (0 leaks)
- `runs/evidence/py/phase3_t1.json` — coding task T1 (3 families, 5 facts)
- `runs/evidence/py/phase4_sleep.json` — v1 sleep (80 facts, FAIL)
- `runs/evidence/py/phase4_v2_step1.json` — v2 Step 1 (5 facts, PASS)
- `runs/evidence/py/phase4_v2_step2.json` — v2 Step 2 (20 facts, catastrophic forgetting)
- `runs/evidence/py/phase4_v3_lora.json` — v3 LoRA (20 facts, underfitting)
- `runs/evidence/py/semantic_ablation.json` — embedding model comparison
- `runs/fact_recall_proof.json` — fact-recall proof (N=10/100/1000)
- `runs/evidence/py/GATES.md` — all phase statuses

## 3. Phase 4 (Sleep) Empirical Boundary Analysis

### Comparison Table

| Config | Facts | Steps | Train memOFF | PPL ratio | heldout_drop | Verdict |
|---|---|---|---|---|---|---|
| v1 (full FT) | 80 | 500/fact | 0.0 | 10^9+ | 1.0 | FAIL (forgetting) |
| v2 Step 1 (full FT) | 5 | 500/fact | 1.0 | 0.94 | — | PASS |
| v2 Step 2 (full FT) | 20 | 500/fact | 1.0 | 10^10 | 1.0 | FAIL (forgetting) |
| v3 (LoRA r=8 + KL) | 20 | 300 total | 0.0-0.1 | 1.00 | 0.0 | FAIL (underfitting) |

### Key Finding: Stability-Plasticity Dilemma

This table demonstrates a clear **Stability-Plasticity Dilemma** at 0.6B scale:

- **Full Fine-Tuning**: High plasticity (train=1.0) but zero stability (PPL ~10^10, heldout_drop=1.0).
  The model memorizes facts but destroys general language modeling and GEM functionality.

- **LoRA + KL Divergence**: Full stability (PPL ~1.00, heldout_drop=0.0) but insufficient plasticity
  (train=0.0-0.1). The model preserves general capability but cannot learn new facts.

- **Conclusion**: At 0.6B scale, factual knowledge should be stored in **non-parametric episodic memory (GEM)**
  rather than parametric weights (Sleep). GEM achieves memON=1.0 without any weight modification,
  avoiding the stability-plasticity trade-off entirely.

This is direct empirical evidence for why gradient-free episodic memory is preferable to
Sleep-to-LoRA consolidation at small model scales.

## 4. Architectural Decision and Production Configuration

- **Default mode**: GEM memON (Sleep Disabled). The system retrieves facts from episodic memory
  and injects them as text context. No weight modification.
- **Sleep module**: Retained as experimental/future work. Potential directions:
  - Larger base models (3B+) with more capacity
  - Isolated adapter routing (separate LoRA per fact cluster)
  - Higher-rank adapters with longer training
- **Retrieval**: Hybrid (dense bge-small top-16 + lexical rerank). Achieves 0.915 at N=1000.
- **Injection**: Text channel (facts as <memory> block in prompt). Hidden channel available but not primary.

## 5. Repository Structure

```
dc_gem/
  config.py       # all hyperparameters, seeds
  core.py         # Qwen3-0.6B wrapper + hidden-state access
  memory.py       # EpisodicMemory + Episode/Store
  retrieval.py    # dense + lexical/BM25 + rerank + score_ablation
  embeddings.py   # bge-small encoder
  inject.py       # text + hidden channels
  sleep.py        # consolidation (experimental)
  lora.py         # LoRA adapter
  eval/           # producer, scorer, watchdog, jsonl
examples/
  fact_recall_demo.py, run_fact_recall_proof.py
  phase4_v2_step1.py, phase4_v2_step2.py, phase4_v3_lora.py
  codebase_agent_demo.py, obscure_agent_demo.py
  semantic_ablation.py, g1_prediction_control.py
tests/
  test_memory.py (4 PASS)
runs/evidence/py/
  GATES.md, LEGACY.md, PREREG_v1.md, PREREG_v2.md
  g1.json, phase2.json, phase3_t1.json, phase4_*.json, semantic_ablation.json
```

## 6. Reproducibility

- Python 3.14.4, venv /home/an-l-z/.venvs/rocm
- Torch 2.14.1+rocm7.2, ROCm 7.2, transformers 5.19.0
- GPU: RX 9070 XT, 16 GB VRAM (never exceed 14 GB)
- Full lock: requirements.lock, ENV.md
- Tests: `python -m pytest` → 4 passed

## 7. Honest Limitations

1. Sleep consolidation not achieved at 0.6B scale (capacity/training-budget).
2. Retrieval degrades at N=1000 (0.915) — acceptable but not perfect.
3. Semantic retrieval (fully paraphrased, no identifier) is at chance for core mean-pool; bge-small reaches 0.84.
4. Single seed for Phase 3/5 (3 seeds required for final publication).
5. T2 (retry function) not fully evaluated (oracle spot-check only).
