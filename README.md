# DC-GEM — Differentiable Core + Gradient-Free Episodic Memory

A hybrid Continual Learning System (CLS) architecture: a slow-learning differentiable
transformer core augmented with a gradient-free episodic memory (GEM). Built on
pre-trained models (Qwen3-0.6B), no training from scratch.

## Key Results

| Metric | Value |
|---|---|
| Fact-recall (memON) | 1.00 (3 query variants) |
| Fact-recall (memOFF) | 0.00 (natural, not a trivial gate) |
| Coding T1 (oracle/memON/memOFF) | 1.00 / 1.00 / 0.00 |
| Retrieval hybrid @ N=1000 | 0.915 |
| PPL ratio (GEM, no Sleep) | 1.00 (zero forgetting) |
| Semantic retrieval (bge vs core mean-pool) | 0.84 vs 0.04 (chance 0.02) |

## Main Finding

**GEM enables fact recall that the core cannot do alone.** With pre-trained Qwen3-0.6B:
- memON retrieves the correct fact and generates the correct answer (1.00)
- memOFF has no access to facts and fails (0.00)
- Zero weight modification → zero catastrophic forgetting (PPL ratio = 1.00)

## Sleep (Stability-Plasticity Dilemma)

Parametric consolidation (Sleep-to-LoRA) was tested and failed at 0.6B scale:
- Full Fine-Tuning: high plasticity (train=1.0) but zero stability (PPL ~10^10)
- LoRA + KL: full stability (PPL ~1.00) but insufficient plasticity (train=0.05)

This is direct empirical evidence that factual knowledge should be stored in
non-parametric episodic memory (GEM) rather than parametric weights at small scales.

## Evidence

- [FINAL_REPORT.md](FINAL_REPORT.md) — full closure report
- [runs/evidence/py/GATES.md](runs/evidence/py/GATES.md) — all phase statuses
- [runs/evidence/py/PREREG_v1.md](runs/evidence/py/PREREG_v1.md) — v1 thresholds
- [runs/evidence/py/PREREG_v2.md](runs/evidence/py/PREREG_v2.md) — v2 thresholds (Sleep revised)
- [runs/evidence/py/LEGACY.md](runs/evidence/py/LEGACY.md) — pre-PREREG results labeled

## Reproducibility

```bash
# Environment
python -m venv .venv-rocm
source .venv-rocm/bin/activate
pip install -r requirements.lock

# Tests
python -m pytest  # 4 passed

# Fact-recall proof
HIP_VISIBLE_DEVICES=0 python examples/run_fact_recall_proof.py --n-facts 20

# Sleep consolidation (v3, LoRA)
HIP_VISIBLE_DEVICES=0 python examples/phase4_v3_lora.py
```

## Architecture

```
dc_gem/
  config.py       # all hyperparameters, seeds
  core.py         # Qwen3-0.6B wrapper + hidden-state access
  memory.py       # EpisodicMemory + Episode/Store
  retrieval.py    # dense k=16 + lexical/BM25 + rerank
  embeddings.py   # bge-small encoder
  inject.py       # text + hidden channels
  sleep.py        # consolidation (experimental)
  lora.py         # LoRA adapter
  eval/           # producer, scorer, watchdog, jsonl
```

## License

MIT
