#!/usr/bin/env python3
"""G1: Prediction control — no-fact task, correct rate <= 0.10."""

from __future__ import annotations
import json, random, re
from pathlib import Path
import torch
from dc_gem.core import PretrainedCore
from dc_gem.eval.producer import make_facts
from dc_gem.eval.scorer import exact_match

def main():
    seed = 0
    rng = random.Random(seed)
    gen = PretrainedCore("Qwen/Qwen3-0.6B", "cuda")
    facts = make_facts(30, seed)
    correct = 0
    details = []
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        prompt = f"Question: {q}\nAnswer:"
        out = gen.generate(prompt, max_new_tokens=8, temperature=0.0)
        code = re.search(r"\d{4,6}", out)
        code = code.group(0) if code else ""
        ok = (code == f["code"])
        correct += int(ok)
        details.append({"id": f["id"], "expected": f["code"], "generated": out.strip(), "correct": bool(ok)})
    rate = correct / len(facts)
    print(f"G1 correct rate: {correct}/{len(facts)} = {rate:.3f}")
    print(f"Gate (<=0.10): {'PASS' if rate <= 0.10 else 'FAIL'}")
    Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
    with open("runs/evidence/py/g1.json", "w") as fh:
        json.dump({"seed": seed, "n": len(facts), "correct": correct,
                   "rate": rate, "gate_threshold": 0.10,
                   "gate_pass": rate <= 0.10, "details": details}, fh, indent=2)
    print("saved runs/evidence/py/g1.json")

if __name__ == "__main__":
    main()
