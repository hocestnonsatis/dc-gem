#!/usr/bin/env python3
"""Phase 3: Memory-required coding task (T1, T2) with hidden tests."""

from __future__ import annotations
import json, random, re, textwrap, subprocess, sys, os
from pathlib import Path
import torch
from dc_gem.core import PretrainedCore
from dc_gem.embeddings import EmbeddingModel
from dc_gem.memory import Store
from dc_gem.retrieval import hybrid_retrieve
from dc_gem.eval.producer import make_facts
from dc_gem.eval.scorer import classify_memoff

GEN_MODEL = "Qwen/Qwen3-0.6B"

def extract_function_code(text: str, fn_name: str) -> str | None:
    # find ```python ... ``` block
    m = re.search(r"```python\s*\n(.*?)```", text, re.DOTALL)
    if m:
        code = m.group(1)
    else:
        # find def
        m = re.search(r"(def\s+" + re.escape(fn_name) + r"\s*\(.*?)(?=\n\S|\Z)", text, re.DOTALL)
        if m:
            code = m.group(1)
        else:
            return None
    # ensure the function is present
    if fn_name not in code:
        return None
    return code.strip()

def run_hidden_test(code: str, test_code: str, timeout: int = 10) -> bool:
    src = code + "\n\n" + test_code
    try:
        exec(compile(src, "<gen>", "exec"), {})
        return True
    except Exception as e:
        return False

def t1_hidden_test(cap: int) -> str:
    return textwrap.dedent(f"""
def _t1():
    assert get_retry_cap() == {cap}, f"expected {cap}, got {{get_retry_cap()}}"
_t1()
""")

def main():
    seed = 0
    rng = random.Random(seed)
    gen = PretrainedCore(GEN_MODEL, "cuda")
    embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")

    # Facts: module name -> cap
    modules = [f"MOD{i:02d}" for i in range(5)]
    caps = [rng.randint(3, 9) for _ in modules]
    facts = [{"id": i, "name": m, "code": str(c),
              "text": f"The retry cap for module {m} is {c}."}
             for i, (m, c) in enumerate(zip(modules, caps))]

    store = Store(capacity=1000)
    for f in facts:
        store.write("Fact", f["text"], f["code"], embed.encode(f["text"]), pinned=False)

    # near-miss distractors: twin pairs with different values
    distractors = []
    for f in facts:
        twin_name = f["name"] + "_TWIN"
        twin_cap = int(f["code"]) + rng.choice([-1, 1])
        distractors.append({"id": f"twin_{f['id']}", "name": twin_name,
                           "code": str(twin_cap),
                           "text": f"The retry cap for module {twin_name} is {twin_cap}."})
    for d in distractors:
        store.write("Fact", d["text"], d["code"], embed.encode(d["text"]), pinned=False)

    # near_miss family: query for original but distractor present
    # lexical family: task shares "retry cap" keyword
    # semantic family: task uses different wording

    families = ["lexical", "near_miss", "semantic"]
    results = []

    def make_task(fact, family):
        name = fact["name"]
        if family == "lexical":
            return f"Write a Python function `get_retry_cap()` that returns the retry cap for module {name}."
        elif family == "near_miss":
            return f"Write a Python function `get_retry_cap()` that returns the retry cap for module {name}."
        else:  # semantic
            return f"Write a Python function `get_retry_cap()` that returns the maximum number of retries allowed for module {name}."

    def make_prompt(fact, family, mode, ctx=""):
        task = make_task(fact, family)
        if mode == "oracle":
            return f"Fact: {fact['text']}\n\n{task}\nAnswer with code only:"
        elif mode == "memON":
            return f"Fact: {ctx}\n\n{task}\nAnswer with code only:"
        else:
            return f"{task}\nAnswer with code only:"

    for family in families:
        for mode in ["oracle", "memON", "memOFF"]:
            passed = 0
            total = 0
            memoff_classes = {}
            for f in facts:
                total += 1
                cap = int(f["code"])
                task = make_task(f, family)
                if mode == "oracle":
                    ctx = ""
                    prompt = make_prompt(f, family, "oracle")
                elif mode == "memON":
                    q = f"What is the retry cap for module {f['name']}?"
                    q_emb = embed.encode(q, is_query=True)
                    ret = hybrid_retrieve(store, q_emb, q, k_dense=16, top_n=1)
                    ctx = ret[0][2] if ret else ""
                    # remove distractor if near_miss? keep as is (retrieval challenge)
                    prompt = make_prompt(f, family, "memON", ctx=ctx)
                else:
                    prompt = make_prompt(f, family, "memOFF")

                out = gen.generate(prompt, max_new_tokens=120, temperature=0.0)
                code = extract_function_code(out, "get_retry_cap")
                test_code = t1_hidden_test(cap)
                ok = False
                if code:
                    ok = run_hidden_test(code, test_code)
                passed += int(ok)

                if mode == "memOFF":
                    cls = classify_memoff(out, str(cap))
                    memoff_classes[cls] = memoff_classes.get(cls, 0) + 1

                leak = (f["text"] in prompt) if mode == "memOFF" else False
                results.append({
                    "family": family, "mode": mode, "fact_id": f["id"],
                    "name": f["name"], "cap": cap, "passed": bool(ok),
                    "generated": out.strip()[:500],
                    "extracted_code": code,
                    "leak": bool(leak),
                })

            rate = passed / total if total else 0
            print(f"  {family}/{mode}: {passed}/{total} = {rate:.3f}")
            if mode == "memOFF":
                print(f"    memOFF classes: {memoff_classes}")

    # Save
    Path("runs/evidence/py").mkdir(parents=True, exist_ok=True)
    with open("runs/evidence/py/phase3_t1.json", "w") as fh:
        json.dump({"seed": seed, "rung": "T1", "n_facts": len(facts),
                   "n_distractors": len(distractors), "results": results}, fh, indent=2)
    print("saved runs/evidence/py/phase3_t1.json")

if __name__ == "__main__":
    main()
