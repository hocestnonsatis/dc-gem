#!/usr/bin/env python3
"""Fact-recall proof: pre-trained model (Qwen2.5-Coder-3B) + GEM.

Compares memON (retrieve from GEM + generate) vs memOFF (generate without memory)
across multiple query variants. Reports exact-match and retrieval accuracy.
Also tests retrieval scaling (N=10/100/1000).
"""

from __future__ import annotations
import argparse
import json
import re
import time
from pathlib import Path

import torch

from dc_gem.core import PretrainedCore
from dc_gem.embeddings import EmbeddingModel
from dc_gem.memory import EpisodicMemory
from dc_gem.retrieval import hybrid_retrieve, extract_code, lexical_score


def make_facts(n: int) -> list[dict]:
    facts = []
    for i in range(n):
        code = f"{1000 + (i * 37) % 9000}"
        facts.append({
            "id": i,
            "name": f"Z{i:02d}",
            "code": code,
            "text": f"The secret password of castle Z{i:02d} is {code}.",
        })
    return facts


def variants(fact: dict) -> dict[str, str]:
    i = fact["id"]
    name = fact["name"]
    return {
        "direct": f"The secret password of castle {name} is",
        "prefix": f"The secret password of castle {name} is",
        "paraphrase": f"What is the password for castle {name}?",
        "question": f"Tell me the code for castle {name}.",
    }


def ingest(mem, embed, facts):
    for f in facts:
        emb = embed.encode(f["text"])
        mem.write_slot(f["id"], emb, emb, text=f["text"])


def retrieve_one(mem, embed, query_text, mode="hybrid"):
    q_emb = embed.encode(query_text, is_query=True)
    if mode == "dense":
        hits = mem.search_top_k(q_emb, 1)
        if not hits:
            return None
        return mem.get_text(hits[0][0])
    # hybrid
    res = hybrid_retrieve(mem, q_emb, query_text, k_dense=16, top_n=1)
    if not res:
        return None
    return res[0][2]


def eval_retrieval(mem, embed, facts, mode="hybrid"):
    """Top-1 retrieval accuracy."""
    correct = 0
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        ret = retrieve_one(mem, embed, q, mode=mode)
        if ret and f["name"] in (ret or ""):
            # check castle name match
            m = re.search(r"castle (Z\d+)", ret)
            if m and m.group(1) == f["name"]:
                correct += 1
    return correct / len(facts) if facts else 0.0


def eval_generation(gen, mem, embed, facts, variant_name, mode="hybrid",
                    disable_memory=False):
    correct = 0
    total = 0
    details = []
    for f in facts:
        qs = variants(f)
        q = qs[variant_name]
        total += 1
        if disable_memory:
            prompt = f"Question: {q}\nAnswer:"
            out = gen.generate(prompt, max_new_tokens=12, temperature=0.0)
        else:
            ret = retrieve_one(mem, embed, q, mode=mode)
            ctx = ret if ret else ""
            prompt = f"Fact: {ctx}\nQuestion: {q}\nAnswer:"
            out = gen.generate(prompt, max_new_tokens=12, temperature=0.0)
        code = extract_code(out)
        ok = (code == f["code"])
        if ok:
            correct += 1
        details.append({
            "id": f["id"], "name": f["name"], "code": f["code"],
            "query": q, "generated": out.strip(), "extracted": code,
            "correct": bool(ok),
        })
    return correct / total if total else 0.0, details


def scaling_test(embed, ns):
    results = {}
    for n in ns:
        facts = make_facts(n)
        mem = EpisodicMemory(size=max(n + 100, 256), d_k=embed.dim,
                            d_v=embed.dim, K=4, tau=0.1)
        ingest(mem, embed, facts)
        # dense top-1
        dense_correct = 0
        for f in facts:
            q = f"What is the password for castle {f['name']}?"
            q_emb = embed.encode(q, is_query=True)
            hits = mem.search_top_k(q_emb, 1)
            ret = mem.get_text(hits[0][0]) if hits else ""
            m = re.search(r"castle (Z\d+)", ret)
            if m and m.group(1) == f["name"]:
                dense_correct += 1
        # dense top-16 recall
        top16_correct = 0
        for f in facts:
            q = f"What is the password for castle {f['name']}?"
            q_emb = embed.encode(q, is_query=True)
            hits = mem.search_top_k(q_emb, 16)
            ids = [re.search(r"castle (Z\d+)", mem.get_text(s)).group(1)
                   for s, _ in hits if re.search(r"castle (Z\d+)", mem.get_text(s))]
            if f["name"] in ids:
                top16_correct += 1
        # hybrid top-1 (k=16)
        hybrid_correct = 0
        for f in facts:
            q = f"What is the password for castle {f['name']}?"
            ret = retrieve_one(mem, embed, q, mode="hybrid")
            m = re.search(r"castle (Z\d+)", ret or "")
            if m and m.group(1) == f["name"]:
                hybrid_correct += 1
        results[str(n)] = {
            "dense_top1": dense_correct / n,
            "dense_top16_recall": top16_correct / n,
            "hybrid_top1": hybrid_correct / n,
        }
        print(f"  N={n}: dense_top1={dense_correct/n:.3f} "
              f"dense_top16={top16_correct/n:.3f} hybrid_top1={hybrid_correct/n:.3f}")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-facts", type=int, default=20)
    ap.add_argument("--gen-model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--embed-model", default="BAAI/bge-small-en-v1.5")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--output", default="runs/fact_recall_proof.json")
    ap.add_argument("--skip-scaling", action="store_true")
    args = ap.parse_args()

    print(f"Loading gen model: {args.gen_model}")
    gen = PretrainedCore(args.gen_model, args.device)
    print(f"Loading embed model: {args.embed_model}")
    embed = EmbeddingModel(args.embed_model, args.device)

    facts = make_facts(args.n_facts)
    print(f"Facts: {len(facts)}")

    mem = EpisodicMemory(size=max(args.n_facts + 100, 256),
                        d_k=embed.dim, d_v=embed.dim, K=4, tau=0.1)
    ingest(mem, embed, facts)
    print(f"Ingested {mem.n_valid} facts")

    # retrieval accuracy
    dense_ret = eval_retrieval(mem, embed, facts, mode="dense")
    # dense eval_retrieval uses paraphrase; recompute for hybrid
    hybrid_ret_acc = 0
    for f in facts:
        q = f"What is the password for castle {f['name']}?"
        ret = retrieve_one(mem, embed, q, mode="hybrid")
        m = re.search(r"castle (Z\d+)", ret or "")
        if m and m.group(1) == f["name"]:
            hybrid_ret_acc += 1
    hybrid_ret_acc /= len(facts)
    print(f"Retrieval accuracy (paraphrase): dense={dense_ret:.3f} hybrid={hybrid_ret_acc:.3f}")

    result = {
        "gen_model": args.gen_model,
        "embed_model": args.embed_model,
        "n_facts": args.n_facts,
        "retrieval_accuracy": {"dense": dense_ret, "hybrid": hybrid_ret_acc},
        "variants": {},
    }

    variant_names = ["direct", "paraphrase", "question"]
    for vn in variant_names:
        mem_on_acc, details_on = eval_generation(gen, mem, embed, facts, vn,
                                                mode="hybrid", disable_memory=False)
        mem_off_acc, details_off = eval_generation(gen, mem, embed, facts, vn,
                                                 mode="hybrid", disable_memory=True)
        result["variants"][vn] = {
            "memON_exact_match": mem_on_acc,
            "memOFF_exact_match": mem_off_acc,
            "gap": mem_on_acc - mem_off_acc,
            "memON_details": details_on,
            "memOFF_details": details_off,
        }
        print(f"  {vn}: memON={mem_on_acc:.3f} memOFF={mem_off_acc:.3f} "
              f"gap={mem_on_acc - mem_off_acc:.3f}")

    # scaling
    if not args.skip_scaling:
        print("Scaling test:")
        result["scaling"] = scaling_test(embed, [10, 100, 1000])

    # save
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved: {out_path}")

    # summary
    print("\n=== SUMMARY ===")
    for vn in variant_names:
        v = result["variants"][vn]
        print(f"{vn}: memON={v['memON_exact_match']:.3f} "
              f"memOFF={v['memOFF_exact_match']:.3f} gap={v['gap']:.3f}")


if __name__ == "__main__":
    main()
