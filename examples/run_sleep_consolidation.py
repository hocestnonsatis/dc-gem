#!/usr/bin/env python3
"""Sleep consolidation: fold GEM facts into Qwen via LoRA (Sleep-to-LoRA).

Trains a LoRA adapter on the fact Q&A pairs so the core (with LoRA) can recall
facts without GEM retrieval. Reports pre/post memOFF exact match and retention.
"""

from __future__ import annotations
import argparse
import json
import re
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

from dc_gem.core import PretrainedCore
from dc_gem.lora import attach_lora, lora_state_dict
from dc_gem.retrieval import extract_code


def make_facts(n):
    out = []
    for i in range(n):
        out.append({"id": i, "name": f"Z{i:02d}",
                    "code": f"{1000 + (i*37)%9000}",
                    "text": f"The secret password of castle Z{i:02d} is {1000+(i*37)%9000}."})
    return out


def build_training_examples(facts, tokenizer):
    """Return list of (input_ids tensor, labels tensor)."""
    templates = [
        ("What is the password for castle {name}?", "{code}"),
        ("Tell me the code for castle {name}.", "{code}"),
        ("The secret password of castle {name} is", "{code}."),
    ]
    examples = []
    for f in facts:
        for qtmpl, atmpl in templates:
            q = qtmpl.format(name=f["name"])
            a = atmpl.format(code=f["code"])
            prompt = f"Question: {q}\nAnswer:"
            full = prompt + " " + a
            p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            f_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
            labels = torch.full_like(f_ids, -100)
            labels[len(p_ids):] = f_ids[len(p_ids):]
            examples.append((f_ids, labels))
    return examples


def ppl_on_text(model, tokenizer, text, device):
    ids = tokenizer(text, return_tensors="pt")["input_ids"].to(device)
    with torch.no_grad():
        out = model(ids, labels=ids)
    return float(torch.exp(out.loss).item())


def gen_with_lora(model, tokenizer, prompt, device, max_new=8):
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, do_sample=False,
                            pad_token_id=tokenizer.eos_token_id)
    new = out[0, inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new, skip_special_tokens=True)


def eval_memoff(model, tokenizer, facts, device):
    correct = 0
    for f in facts:
        for q in [f"What is the password for castle {f['name']}?",
                  f"Tell me the code for castle {f['name']}?"]:
            prompt = f"Question: {q}\nAnswer:"
            out = gen_with_lora(model, tokenizer, prompt, device, max_new=8)
            code = extract_code(out)
            if code == f["code"]:
                correct += 1
                break
        else:
            pass
    # count per fact: correct if any query variant yields code
    correct = 0
    for f in facts:
        ok = False
        for q in [f"What is the password for castle {f['name']}?",
                  f"Tell me the code for castle {f['name']}?"]:
            prompt = f"Question: {q}\nAnswer:"
            out = gen_with_lora(model, tokenizer, prompt, device, max_new=8)
            if extract_code(out) == f["code"]:
                ok = True
                break
        correct += int(ok)
    return correct / len(facts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-0.6B")
    ap.add_argument("--n-facts", type=int, default=20)
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--rank", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--max-layers", type=int, default=4)
    ap.add_argument("--output", default="runs/sleep_consolidation.json")
    args = ap.parse_args()

    print("Loading model...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16,
        device_map=args.device)
    model.eval()
    model.gradient_checkpointing_enable()

    facts = make_facts(args.n_facts)
    print(f"Facts: {len(facts)}")

    # pre-sleep memOFF
    pre_acc = eval_memoff(model, tokenizer, facts, args.device)
    print(f"pre-sleep memOFF exact: {pre_acc:.3f}")

    # retention text
    retention_text = "The quick brown fox jumps over the lazy dog. " * 5
    pre_ppl = ppl_on_text(model, tokenizer, retention_text, args.device)
    print(f"pre-sleep retention PPL: {pre_ppl:.3f}")

    # attach lora
    params = attach_lora(model, rank=args.rank, alpha=args.rank * 2,
                        targets=("q_proj", "v_proj"), max_layers=args.max_layers)
    print(f"LoRA params: {sum(p.numel() for p in params)}")

    # training data
    examples = build_training_examples(facts, tokenizer)
    print(f"Training examples: {len(examples)}")

    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0)

    model.train()
    for step in range(args.steps):
        # sample a batch of 4
        idxs = torch.randperm(len(examples))[:1].tolist()
        opt.zero_grad(set_to_none=True)
        torch.cuda.empty_cache()
        total_loss = 0.0
        for i in idxs:
            ids, labels = examples[i]
            ids = ids.to(args.device)
            labels = labels.to(args.device)
            out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
            loss = out.loss
            loss.backward()
            total_loss += float(loss.item())
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if (step + 1) % 50 == 0:
            print(f"  step {step+1}/{args.steps} loss={total_loss:.4f}")

    model.eval()

    # post-sleep memOFF
    post_acc = eval_memoff(model, tokenizer, facts, args.device)
    print(f"post-sleep memOFF exact: {post_acc:.3f}")

    post_ppl = ppl_on_text(model, tokenizer, retention_text, args.device)
    print(f"post-sleep retention PPL: {post_ppl:.3f}")

    result = {
        "n_facts": args.n_facts,
        "steps": args.steps,
        "rank": args.rank,
        "pre_sleep_memoff_exact": pre_acc,
        "post_sleep_memoff_exact": post_acc,
        "improvement": post_acc - pre_acc,
        "pre_retention_ppl": pre_ppl,
        "post_retention_ppl": post_ppl,
        "ppl_ratio": post_ppl / pre_ppl if pre_ppl > 0 else None,
    }

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as fh:
        json.dump(result, fh, indent=2)
    print(f"Saved: {out_path}")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
