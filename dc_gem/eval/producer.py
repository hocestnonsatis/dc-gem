"""Producer: generate test instances (facts, tasks) deterministically from seed."""

from __future__ import annotations
import random


def make_facts(n: int, seed: int) -> list[dict]:
    rng = random.Random(seed)
    facts = []
    used_codes = set()
    for i in range(n):
        # low-prior 4-6 digit codes, avoid round numbers
        while True:
            code = rng.randint(10000, 99999)
            if code % 1000 != 0 and code not in used_codes:
                used_codes.add(code)
                break
        name = f"Z{i:03d}"
        facts.append({"id": i, "name": name, "code": str(code),
                      "text": f"The secret password of castle {name} is {code}."})
    return facts
