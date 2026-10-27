#!/usr/bin/env python3
"""Phase 2: memOFF classification, leak check, held-out split."""

from __future__ import annotations
import json, random, re, hashlib
from pathlib import Path
import torch
from dc_gem.core import PretrainedCore
from dc_gem.eval.producer import make_facts
from dc_gem.eval.scorer import classify_memoff
from dc_gem.memory import Store, Episode
from dc_gem.embeddings import EmbeddingModel
from dc_gem.retrieval import hybrid_retrieve

def leak_check(fact_text: str, training_texts: list[str], prompt: str) -> bool:
    """Return True if leak detected (fact present in training or prompt)."""
    for t in training_texts:
        if fact_text in t:
            return True
    if fact_text in prompt:
        return True
    return False

def main():
    seed = 0
    rng = random.Random(seed)
    gen = PretrainedCore("Qwen/Qwen3-0.6B", "cuda")
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")

    facts = make_facts(30, seed)
    # split: train (sleep), heldout_new, paraphrase (of train)
    train = facts[:15]
    heldout_new = facts[15:]

    # Build store with train facts only
    store = Store(capacity=1000)
    for f in train:
        store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)

    # Sleep training texts (for leak check)
    sleep_texts = [f["text"] for f in train]

    # Evaluate memON/memOFF on train facts (paraphrase template)
    paraphrase_templates = [
        "What is the password for castle {name}?",
        "Tell me the code for castle {name}.",
        "Castle {name} password?",
    ]

    results = []
    for f in facts:
        is_train = f in train
        is_heldout = f in heldout_new
        # pick a paraphrase template for train facts
        if is_train:
            tmpl = rng.choice(paraphrase_templates)
            q = tmpl.format(name=f["name"])
            split = "train_paraphrase" if True else "train"
        else:
            q = f"What is the password for castle {f['name']}?"
            split = "heldout_new"

        # memON
        q_emb = embed.encode(q, is_query=True)
        ret = hybrid_retrieve(store, q_emb, q, k_dense=16, top_n=1)
        ctx = ret[0][2] if ret else ""
        prompt_on = f"Fact: {ctx}\nQuestion: {q}\nAnswer:"
        out_on = gen.generate(prompt_on, max_new_tokens=8, temperature=0.0)
        memon_code = None
        m = re.search(r"\d{4,6}", out_on)
        if m: memon_code = m.group(0)
        memon_correct = (memon_code == f["code"])

        # memOFF
        prompt_off = f"Question: {q}\nAnswer:"
        out_off = gen.generate(prompt_off, max_new_tokens=8, temperature=0.0)
        leak = leak_check(f["text"], sleep_texts, prompt_off) if is_train else leak_check(f["text"], sleep_texts, prompt_off)
        memoff_class = classify_memoff(out_off, f["code"])

        results.append({
            "fact_id": f["id"], "name": f["name"], "code": f["code"],
            "split": split, "is_train": is_train, "is_heldout": is_heldout,
            "query": q, "memON_output": out_on.strip(), "memON_correct": bool(memon_correct),
            "memOFF_output": out_off.strip(), "memOFF_class": memoff_class,
            "leak": bool(leak),
        })

    # Summary
    train_par = [r for r in results if r["is_train"]]
    heldout = [r for r in results if r["is_heldout"]]
    memon_train = sum(1 for r in train_par if r["memON_correct"]) / len(train_par) if train_par else 0
    memon_heldout = sum(1 for r in heldout if r["memON_correct"]) / len(heldout) if heldout else 0
    leak_count = sum(1 for r in results if r["leak"] and r["is_heldout"])
    memoff_classes = {}
    for r in results:
        memoff_classes[r["memOFF_class"]] = memoff_classes.get(r["memOFF_class"], 0) + 1

    summary = {
        "seed": seed, "n_facts": len(facts), "n_train": len(train), "n_heldout": len(heldout_new),
        "memON_train_exact": memon_train, "memON_heldout_exact": memon_heldout,
        "leak_count": leak_count, "memOFF_class_counts": memoff_classes,
    }
    print(json.dumps(summary, indent=2))

    Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
    with open("runs/evidence/py/phase2.json", "w") as fh:
        json.dump({"summary": summary, "results": results}, fh, indent=2)
    print("saved runs/evidence/py/phase2.json")

if __name__ == "__main__":
    main()
