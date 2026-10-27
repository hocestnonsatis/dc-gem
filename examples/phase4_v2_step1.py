#!/usr/bin/env python3
"""Phase 4 v2 Step 1: capacity proof — 5 facts, 500 steps/fact, 1 seed."""

from __future__ import annotations
import json, random, torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from dc_gem.eval.producer import make_facts
from dc_gem.eval.scorer import exact_match as em

MODEL = "Qwen/Qwen3-0.6B"
SEED = 0
N_TRAIN = 5
STEPS_PER_FACT = 500
TOTAL = N_TRAIN * STEPS_PER_FACT

facts = make_facts(N_TRAIN, SEED)
train = facts[:N_TRAIN]
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")

examples = []
for f in train:
    prompt = f"Question: What is the password for castle {f['name']}?\nAnswer:"
    full = prompt + " " + f["code"]
    p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    f_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
    labels = torch.full((f_ids.shape[0],), -100, dtype=torch.long)
    labels[len(p_ids):] = f_ids[len(p_ids):]
    examples.append((f_ids, labels, f))

for p in model.parameters():
    p.requires_grad = True
opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
model.train()
step = 0
while step < TOTAL:
    import random as _r
    _r.shuffle(examples)
    for ids, labels, f in examples:
        if step >= TOTAL:
            break
        ids = ids.to("cuda"); labels = labels.to("cuda")
        opt.zero_grad(set_to_none=True)
        out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
model.eval()

details = []
exact = 0
for f in train:
    q = f"What is the password for castle {f['name']}?"
    prompt = f"Question: {q}\nAnswer:"
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
    with torch.no_grad():
        g = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id)
    text = tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
    ok = em(text.strip(), f["code"])
    exact += int(ok)
    details.append({"name": f["name"], "code": f["code"], "generated": text.strip(), "exact": bool(ok)})

result = {
    "prereg": "v2", "step": 1, "seed": SEED, "n_facts": N_TRAIN,
    "steps_per_fact": STEPS_PER_FACT, "total_steps": TOTAL,
    "exact": exact, "exact_rate": exact / N_TRAIN,
    "gate_pass": (exact / N_TRAIN) >= 0.6,
    "details": details,
}
Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
with open("runs/evidence/py/phase4_v2_step1.json", "w") as fh:
    json.dump(result, fh, indent=2)
print(f"exact: {exact}/{N_TRAIN} = {exact/N_TRAIN:.3f}")
print(f"gate (>=0.6): {'PASS' if result['gate_pass'] else 'FAIL'}")
