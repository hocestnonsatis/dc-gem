#!/usr/bin/env python3
"""Phase 4: Sleep-to-LoRA valid experiment."""

from __future__ import annotations
import json, random, re, time
from pathlib import Path
import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer
from dc_gem.eval.producer import make_facts
from dc_gem.memory import Store
from dc_gem.embeddings import EmbeddingModel
from dc_gem.sleep import build_replay_examples, select_episodes
from dc_gem.lora import attach_lora

MODEL = "Qwen/Qwen3-0.6B"
REPLAY_RATIOS = [0.0, 0.1, 0.3]
RANKS = [8, 16]
SEEDS = [0, 1, 2]
N_FACTS = 100
N_TRAIN = 80
STEPS = 500

def eval_memoff(model, tokenizer, facts, device):
    correct = 0
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        new = out[0, ids.shape[1]:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        m = re.search(r"\d{4,6}", text)
        if m and m.group(0) == f["code"]:
            correct += 1
    return correct / len(facts) if facts else 0.0

def eval_memoff_paraphrase(model, tokenizer, facts, device):
    correct = 0
    for f in facts:
        q = f"Tell me the code for castle {f['name']}."
        prompt = f"Question: {q}\nAnswer:"
        ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to(device)
        with torch.no_grad():
            out = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False,
                                pad_token_id=tokenizer.eos_token_id)
        new = out[0, ids.shape[1]:]
        text = tokenizer.decode(new, skip_special_tokens=True)
        m = re.search(r"\d{4,6}", text)
        if m and m.group(0) == f["code"]:
            correct += 1
    return correct / len(facts) if facts else 0.0

def eval_memON(store, embed, gen_model, facts, device):
    # GEM retrieval + gen
    correct = 0
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        q_emb = embed.encode(q, is_query=True)
        # dense search
        hits = store.search_dense(q_emb, 4)
        ctx = hits[0][0].key_text + " " + hits[0][0].value_text if hits else ""
        prompt = f"Fact: {ctx}\nQuestion: {q}\nAnswer:"
        ids = gen_model.tokenizer(prompt, return_tensors="pt")["input_ids"].to(device) if hasattr(gen_model, 'tokenizer') else None
    return 0.0  # placeholder; memON tested via gen model

def ppl(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())

RETENTION_TEXT = "The quick brown fox jumps over the lazy dog. " * 10

all_results = []
for seed in SEEDS:
    rng = random.Random(seed)
    facts = make_facts(N_FACTS, seed)
    train = facts[:N_TRAIN]
    heldout = facts[N_TRAIN:]
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")

    for replay in REPLAY_RATIOS:
        for rank in RANKS:
            print(f"\n=== seed={seed} replay={replay} rank={rank} ===", flush=True)
            tokenizer = AutoTokenizer.from_pretrained(MODEL)
            model = AutoModelForCausalLM.from_pretrained(MODEL, torch_dtype=torch.bfloat16,
                                                        device_map="cuda")
            model.eval()
            store = Store(capacity=10000)
            for f in train:
                store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)

            # pre-sleep
            pre_train_memoff = eval_memoff(model, tokenizer, train, "cuda")
            pre_para = eval_memoff_paraphrase(model, tokenizer, train, "cuda")
            pre_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")
            # heldout memON pre (GEM works)
            # add heldout to a separate store for memON test
            store2 = Store(capacity=10000)
            for f in heldout:
                store2.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)

            # sleep
            episodes = select_episodes(store)
            from dc_gem.sleep import build_replay_examples
            examples = build_replay_examples(episodes, tokenizer, replay_ratio=replay)
            params = attach_lora(model, rank=rank, alpha=rank*2, targets=("q_proj","v_proj"), max_layers=None)
            opt = torch.optim.AdamW(params, lr=1e-4, weight_decay=0.0)
            model.train()
            import random as _r
            for step in range(STEPS):
                idx = _r.randrange(len(examples))
                ids, labels = examples[idx]
                ids = ids.to("cuda"); labels = labels.to("cuda")
                opt.zero_grad(set_to_none=True)
                out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
                out.loss.backward()
                torch.nn.utils.clip_grad_norm_(params, 1.0)
                opt.step()
            model.eval()

            # post-sleep
            post_train_memoff = eval_memoff(model, tokenizer, train, "cuda")
            post_para = eval_memoff_paraphrase(model, tokenizer, train, "cuda")
            post_ppl = ppl(model, tokenizer, RETENTION_TEXT, "cuda")

            # heldout memON post: add heldout to GEM, retrieve+gen
            # Use gen via model directly with retrieved context
            heldout_memon_post = 0
            for f in heldout:
                q = f"What is the password for castle {f['name']}?"
                q_emb = embed.encode(q, is_query=True)
                hits = store2.search_dense(q_emb, 4)
                ctx = hits[0][0].key_text + " " + hits[0][0].value_text if hits else ""
                prompt = f"Fact: {ctx}\nQuestion: {q}\nAnswer:"
                ids = tokenizer(prompt, return_tensors="pt")["input_ids"].to("cuda")
                with torch.no_grad():
                    out = model.generate(input_ids=ids, max_new_tokens=8, do_sample=False,
                                        pad_token_id=tokenizer.eos_token_id)
                text = tokenizer.decode(out[0, ids.shape[1]:], skip_special_tokens=True)
                m = re.search(r"\d{4,6}", text)
                if m and m.group(0) == f["code"]:
                    heldout_memon_post += 1
            heldout_memon_post /= len(heldout)

            result = {
                "seed": seed, "replay": replay, "rank": rank,
                "n_train": N_TRAIN, "n_heldout": len(heldout),
                "pre_train_memoff": pre_train_memoff, "post_train_memoff": post_train_memoff,
                "pre_paraphrase": pre_para, "post_paraphrase": post_para,
                "pre_ppl": pre_ppl, "post_ppl": post_ppl,
                "ppl_ratio": post_ppl / pre_ppl if pre_ppl > 0 else None,
                "heldout_memon_post": heldout_memon_post,
            }
            all_results.append(result)
            print(json.dumps(result, indent=2))

            # save incrementally
            Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
            with open("runs/evidence/py/phase4_sleep.json", "w") as fh:
                json.dump({"results": all_results}, fh, indent=2)

            # cleanup
            del model
            torch.cuda.empty_cache()

print("\n=== DONE ===")
