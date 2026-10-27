"""Sleep consolidation: fold selected episodes into LoRA via replay."""

from __future__ import annotations
import random
import torch
import torch.nn.functional as F
from dc_gem.lora import attach_lora


def select_episodes(store, max_n: int = 200):
    """Select episodes for consolidation: pinned first, then most recent."""
    eps = store.all() if hasattr(store, "all") else []
    pinned = [e for e in eps if getattr(e, "pinned", False)]
    rest = [e for e in eps if not getattr(e, "pinned", False)]
    rest.sort(key=lambda e: getattr(e, "last_used", 0), reverse=True)
    return (pinned + rest)[:max_n]


def build_replay_examples(episodes, tokenizer, replay_ratio: float = 0.3):
    """Build (input_ids, labels) training examples from episodes."""
    examples = []
    for ep in episodes:
        q = f"What is {ep.key_text}?"
        a = ep.value_text
        prompt = f"Question: {q}\nAnswer:"
        full = prompt + " " + a
        p_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        f_ids = tokenizer(full, return_tensors="pt", add_special_tokens=True)["input_ids"][0]
        labels = torch.full((f_ids.shape[0],), -100, dtype=torch.long)
        labels[len(p_ids):] = f_ids[len(p_ids):]
        examples.append((f_ids, labels))
    if replay_ratio < 1.0 and len(examples) > 1:
        k = max(1, int(len(examples) * replay_ratio))
        examples = random.sample(examples, k)
    return examples


def consolidate(model, tokenizer, store, *, replay_ratio=0.3, rank=16,
               steps=500, lr=1e-4, max_layers=None, device="cuda"):
    """Run sleep consolidation. Returns dict of metrics."""
    episodes = select_episodes(store)
    params = attach_lora(model, rank=rank, alpha=rank * 2,
                        targets=("q_proj", "v_proj"), max_layers=max_layers)
    examples = build_replay_examples(episodes, tokenizer, replay_ratio=replay_ratio)
    if not examples:
        return {"steps": 0, "n_episodes": 0, "n_examples": 0}
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=0.0)
    model.train()
    for step in range(steps):
        import random as _r
        idx = _r.randrange(len(examples))
        ids, labels = examples[idx]
        ids = ids.to(device); labels = labels.to(device)
        opt.zero_grad(set_to_none=True)
        out = model(ids.unsqueeze(0), labels=labels.unsqueeze(0))
        out.loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
    model.eval()
    return {"steps": steps, "n_episodes": len(episodes),
            "n_examples": len(examples), "replay_ratio": replay_ratio,
            "rank": rank}
