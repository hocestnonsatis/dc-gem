#!/usr/bin/env python3
"""Phase 4 v2 Step 2: 20 facts, 3 seeds, 500 steps/fact."""

from __future__ import annotations
import json, random, time, re
import torch
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer
from dc_gem.eval.producer import make_facts
from dc_gem.eval.scorer import exact_match as em
from dc_gem.memory import Store
from dc_gem.embeddings import EmbeddingModel

MODEL = "Qwen/Qwen3-0.6B"
N_TRAIN = 20
N_HELDOUT = 5
STEPS_PER_FACT = 500
SEEDS = [0, 1, 2]
RETENTION_TEXT = "The quick brown fox jumps over the lazy dog. " * 10

def ppl(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())

def eval_memoff(model, tokenizer, facts, device):
    correct = 0
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        if em(text.strip(), f["code"]):
            correct += 1
    return correct / len(facts) if facts else 0.0

def eval_paraphrase(model, tokenizer, facts, device):
    correct = 0
    for f in facts:
        q = f"Tell me the code for castle {f['name']}."
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        if em(text.strip(), f["code"]):
            correct += 1
    return correct / len(facts) if facts else 0.0

def eval_heldout_memon(model, tokenizer, embed, facts, device):
    # GEM contains heldout facts; retrieve + generate
    store = Store(capacity=10000)
    for f in facts:
        store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)
    from dc_gem.retrieval import hybrid_retrieve
    correct = 0
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        q_emb = embed.encode(q, is_query=True)
        ret = hybrid_retrieve(store, q_emb, q, k_dense=16, top_n=1)
        ctx = ret[0][2] if ret else ""
        prompt = f"Fact: {ctx}\nQuestion: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        if em(text.strip(), f["code"]):
            correct += 1
    return correct / len(facts) if facts else 0.0

all_results = []
for seed in SEEDS:
    print(f"\n=== seed={seed} ===", flush=True)
    facts = make_facts(N_TRAIN + N_HELDOUT, seed)
    train = facts[:N_TRAIN]
    heldout = facts[N_TRAIN:]
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    model.eval()

    # build examples
    examples = []
    for f in train:
        prompt = f"Question: What is the password for castle {f['name']}?\nAnswer:"
        full = prompt + " " + f["code"]
        p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        f_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        labels = torch.full((f_ids.shape[0],), -100, dtype=torch.long)
        labels[len(p_ids):] = f_ids[len(p_ids):]
        examples.append((f_ids, labels, f))

    store = Store(capacity=10000)
    for f in train:
        store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)

    # pre-sleep
    pre_train = eval_memoff(model, tokenizer, train, "cuda")
    pre_para = eval_paraphrase(model, tokenizer, train, "cuda")
    pre_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")
    # heldout memON pre
    heldout_store = Store(capacity=10000)
    for f in heldout:
        heldout_store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)
    # memON pre for heldout
    heldout_memon_pre = 0
    for f in heldout:
        q = f"What is the password for castle {f['name']}?"
        q_emb = embed.encode(q, is_query=True)
        from dc_gem.retrieval import hybrid_retrieve
        ret = hybrid_retrieve(heldout_store, q_emb, q, k_dense=16, top_n=1)
        ctx = ret[0][2] if ret else ""
        prompt = f"Fact: {ctx}\nQuestion: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
        with torch.no_grad():
            g = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False, use_cache=False, pad_token_id=tokenizer.eos_token_id)
        text = tokenizer.decode(g[0, ids.shape[1]:], skip_special_tokens=True)
        if em(text.strip(), f["code"]):
            heldout_memon_pre += 1
    heldout_memon_pre /= len(heldout)

    # sleep
    for p in model.parameters():
        p.requires_grad = True
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.0)
    model.train()
    step = 0
    TOTAL = N_TRAIN * STEPS_PER_FACT
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

    # post-sleep
    post_train = eval_memoff(model, tokenizer, train, "cuda")
    post_para = eval_paraphrase(model, tokenizer, train, "cuda")
    post_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")
    # heldout memON post
    heldout_memon_post = eval_heldout_memon(model, tokenizer, embed, heldout, "cuda")

    result = {
        "seed": seed, "n_train": N_TRAIN, "n_heldout": len(heldout),
        "steps": TOTAL,
        "pre_train_memoff": pre_train, "post_train_memoff": post_train,
        "pre_paraphrase": pre_para, "post_paraphrase": post_para,
        "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "ppl_ratio": post_ppl / pre_ppl if pre_ppl > 0 else None,
        "heldout_memon_pre": heldout_memon_pre,
        "heldout_memon_post": heldout_memon_post,
        "heldout_memon_drop": heldout_memon_pre - heldout_memon_post,
    }
    all_results.append(result)
    print(json.dumps(result, indent=2), flush=True)

    Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
    with open("runs/evidence/py/phase4_v2_step2.json", "w") as fh:
        json.dump({"results": all_results}, fh, indent=2)

    del model
    torch.cuda.empty_cache()

print("\n=== DONE ===")
