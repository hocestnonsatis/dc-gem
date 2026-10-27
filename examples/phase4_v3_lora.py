#!/usr/bin/env python3
"""Phase 4 v3: LoRA-based Sleep with KL-divergence and forgetting prevention."""

from __future__ import annotations
import json, random, re, math
import torch
import torch.nn.functional as F
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, get_cosine_schedule_with_warmup
from dc_gem.eval.producer import make_facts
from dc_gem.eval.scorer import exact_match as em
from dc_gem.memory import Store
from dc_gem.embeddings import EmbeddingModel
from dc_gem.retrieval import hybrid_retrieve

MODEL = "Qwen/Qwen3-0.6B"
N_TRAIN = 20
N_HELDOUT = 5
STEPS = 300  # global steps (200-400 budget)
SEEDS = [0, 1, 2]
RETENTION_TEXT = "The quick brown fox jumps over the lazy dog. " * 10
ANCHOR_TEXTS = [
    "The capital of France is Paris.",
    "Water boils at 100 degrees Celsius.",
    "Python is a programming language.",
    "The Earth orbits the Sun.",
    "Machine learning uses data to learn patterns.",
]

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
    store = Store(capacity=10000)
    for f in facts:
        store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)
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

def kl_divergence(base_logits, lora_logits, T=2.0):
    p_base = F.log_softmax(base_logits / T, dim=-1)
    p_lora = F.log_softmax(lora_logits / T, dim=-1)
    # KL(p_base || p_lora) = sum p_base * (log p_base - log p_lora)
    kl = F.kl_div(p_lora, p_base.exp(), reduction='batchmean', log_target=False)
    return kl * (T * T)

all_results = []
for seed in SEEDS:
    print(f"\n=== seed={seed} ===", flush=True)
    rng = random.Random(seed)
    facts = make_facts(N_TRAIN + N_HELDOUT, seed)
    train = facts[:N_TRAIN]
    heldout = facts[N_TRAIN:]
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")
    tokenizer = AutoTokenizer.from_pretrained(MODEL)

    # frozen base model
    base_model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    base_model.eval()
    for p in base_model.parameters():
        p.requires_grad = False

    # lora model
    model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16, device_map="cuda")
    from dc_gem.lora import attach_lora
    params = attach_lora(model, rank=8, alpha=16, targets=("gate_proj", "up_proj", "down_proj"),
                        max_layers=None, dropout=0.05)
    model.eval()

    # build fact examples
    fact_examples = []
    for f in train:
        prompt = f"Question: What is the password for castle {f['name']}?\nAnswer:"
        full = prompt + " " + f["code"]
        p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        f_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        labels = torch.full((f_ids.shape[0],), -100, dtype=torch.long)
        labels[len(p_ids):] = f_ids[len(p_ids):]
        fact_examples.append((f_ids, labels, f))

    # anchor examples
    anchor_examples = []
    for text in ANCHOR_TEXTS:
        ids = tokenizer(text, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        anchor_examples.append(ids)

    # pre-sleep
    pre_train = eval_memoff(model, tokenizer, train, "cuda")
    pre_para = eval_paraphrase(model, tokenizer, train, "cuda")
    pre_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")
    heldout_store_pre = Store(capacity=10000) if False else None
    heldout_memon_pre = eval_heldout_memon(model, tokenizer, embed, heldout, "cuda")

    # optimizer + scheduler
    opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.01)
    scheduler = get_cosine_schedule_with_warmup(opt, num_warmup_steps=20, num_training_steps=STEPS)

    best_train_acc = 0.0
    best_ppl_ratio = 1.0
    stopped_early = False
    stop_reason = ""
    step = 0

    while step < STEPS:
        # 80% fact, 20% anchor
        if random.random() < 0.8:
            ids, labels, f = random.choice(fact_examples)
            ids = ids.to("cuda"); labels = labels.to("cuda")
            opt.zero_grad(set_to_none=True)
            # base logits (frozen)
            with torch.no_grad():
                base_out = base_model(ids.unsqueeze(0))
                base_logits = base_out.logits
            # lora logits
            lora_out = model(ids.unsqueeze(0))
            lora_logits = lora_out.logits
            # CE on answer tokens
            shift_logits = lora_logits[:, :-1, :].contiguous()
            shift_labels = labels[1:].unsqueeze(0).contiguous()
            ce_loss = F.cross_entropy(shift_logits.view(-1, shift_logits.size(-1)),
                                     shift_labels.view(-1), ignore_index=-100)
            # KL divergence
            kl = kl_divergence(base_logits, lora_logits)
            loss = ce_loss + 0.5 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()
        else:
            # anchor: KL only
            a_ids = random.choice(anchor_examples).to("cuda")
            opt.zero_grad(set_to_none=True)
            with torch.no_grad():
                base_out = base_model(a_ids.unsqueeze(0))
                base_logits = base_out.logits
            lora_out = model(a_ids.unsqueeze(0))
            lora_logits = lora_out.logits
            kl = kl_divergence(base_logits, lora_logits)
            loss = 0.5 * kl
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            scheduler.step()

        step += 1

        # early stopping check every 25 steps
        if step % 25 == 0:
            model.eval()
            train_acc = eval_memoff(model, tokenizer, train, "cuda")
            cur_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")
            ppl_ratio = cur_ppl / pre_ppl if pre_ppl > 0 else 1.0
            model.train()
            print(f"  step {step}: train_acc={train_acc:.3f} ppl_ratio={ppl_ratio:.3f}", flush=True)
            if train_acc >= 0.75:
                stopped_early = True
                stop_reason = f"train_acc {train_acc:.3f} >= 0.75"
                break
            if ppl_ratio > 1.12:
                stopped_early = True
                stop_reason = f"ppl_ratio {ppl_ratio:.3f} > 1.12"
                break
            best_train_acc = max(best_train_acc, train_acc)
            best_ppl_ratio = min(best_ppl_ratio, ppl_ratio)

    model.eval()

    # post-sleep eval
    post_train = eval_memoff(model, tokenizer, train, "cuda")
    post_para = eval_paraphrase(model, tokenizer, train, "cuda")
    post_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")
    heldout_memon_post = eval_heldout_memon(model, tokenizer, embed, heldout, "cuda")

    result = {
        "seed": seed, "n_train": N_TRAIN, "n_heldout": len(heldout),
        "steps": step, "stopped_early": stopped_early, "stop_reason": stop_reason,
        "pre_train_memoff": pre_train, "post_train_memoff": post_train,
        "pre_paraphrase": pre_para, "post_paraphrase": post_para,
        "pre_ppl": pre_ppl, "post_ppl": post_ppl,
        "ppl_ratio": post_ppl / pre_ppl if pre_ppl > 0 else None,
        "heldout_memon_pre": heldout_memon_pre,
        "heldout_memon_post": heldout_memon_post,
        "heldout_memon_drop": heldout_memon_pre - heldout_memon_post,
        "best_train_acc": best_train_acc, "best_ppl_ratio": best_ppl_ratio,
    }
    all_results.append(result)
    print(json.dumps(result, indent=2), flush=True)

    Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
    with open("runs/evidence/py/phase4_v3_lora.json", "w") as fh:
        json.dump({"results": all_results}, fh, indent=2)

    del model, base_model
    torch.cuda.empty_cache()

print("\n=== DONE ===")
