"""Gradient-Free Episodic Memory (GEM) — pre-trained embedding compatible."""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class EpisodicMemory:
    def __init__(self, size: int, d_k: int, d_v: int, K: int, tau: float = 0.1):
        self.size = size
        self.d_k = d_k
        self.d_v = d_v
        self.K = K
        self.tau = tau
        self.keys = torch.zeros(size, d_k)
        self.values = torch.zeros(size, d_v)
        self.valid = torch.zeros(size, dtype=torch.bool)
        self.n_valid = 0
        self._texts = {}  # slot -> text

    def write_slot(self, i: int, k: torch.Tensor, v: torch.Tensor, text: str = ""):
        k = k.detach().cpu().reshape(self.d_k)
        v = v.detach().cpu().reshape(self.d_v)
        k = k / k.norm().clamp_min(1e-8)
        self.keys[i] = k
        self.values[i] = v
        if not bool(self.valid[i]):
            self.valid[i] = True
            self.n_valid += 1
        self._texts[i] = text

    def search_top_k(self, q: torch.Tensor, k: int) -> list[tuple[int, float]]:
        if self.n_valid == 0:
            return []
        q = q.detach().cpu().reshape(self.d_k)
        q = q / q.norm().clamp_min(1e-8)
        scores = q @ self.keys.T
        scores = scores.masked_fill(~self.valid, float("-inf"))
        K = min(k, self.n_valid, self.size)
        topk = scores.topk(K, dim=-1)
        return [(int(idx), float(sc)) for idx, sc in zip(topk.indices.tolist(), topk.values.tolist())]

    def get_text(self, idx: int) -> str:
        return self._texts.get(idx, "")

    def reset(self):
        self.keys.zero_()
        self.values.zero_()
        self.valid.zero_()
        self.n_valid = 0
        self._texts.clear()

import time as _time
from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class Episode:
    id: str
    kind: str  # Fact | Convention | SymbolNote
    key_text: str
    value_text: str
    embedding: torch.Tensor
    created: float = field(default_factory=lambda: _time.time())
    last_used: float = field(default_factory=lambda: _time.time())
    pinned: bool = False


class Store:
    """In-place overwrite store with pinning and consolidation."""

    def __init__(self, capacity: int = 4096):
        self.capacity = capacity
        self._episodes: dict[str, Episode] = {}
        self._counter = 0

    def write(self, kind: str, key_text: str, value_text: str,
              embedding: torch.Tensor, pinned: bool = False) -> Episode:
        self._counter += 1
        ep = Episode(
            id=f"ep_{self._counter}", kind=kind, key_text=key_text,
            value_text=value_text, embedding=embedding.detach().cpu().clone(),
            pinned=pinned,
        )
        if key_text in self._episodes:
            old = self._episodes[key_text]
            # pinned not overwritten by non-pinned
            if old.pinned and not pinned:
                old.last_used = _time.time()
                return old
            self._episodes[key_text] = ep
        else:
            if len(self._episodes) >= self.capacity:
                self._evict()
            self._episodes[key_text] = ep
        return ep

    def _evict(self):
        cands = [(k, v) for k, v in self._episodes.items() if not v.pinned]
        if not cands:
            return
        oldest_key = min(cands, key=lambda kv: kv[1].last_used)[0]
        del self._episodes[oldest_key]

    def get(self, key_text: str) -> Optional[Episode]:
        return self._episodes.get(key_text)

    def all(self) -> List[Episode]:
        return list(self._episodes.values())

    def search_dense(self, q_emb: torch.Tensor, k: int) -> List[tuple[Episode, float]]:
        if not self._episodes:
            return []
        q = q_emb.detach().cpu().reshape(-1).float()
        q = q / q.norm().clamp_min(1e-8)
        scored = []
        for ep in self._episodes.values():
            key = ep.embedding.reshape(-1).float()
            key = key / key.norm().clamp_min(1e-8)
            scored.append((ep, float(q @ key)))
        scored.sort(key=lambda x: -x[1])
        return scored[:k]

    def consolidate(self):
        """Merge duplicates (same key) and prune old unpinned."""
        # duplicates already handled by in-place overwrite; prune low-utility old
        # Simple: remove unpinned episodes older than median age if over capacity
        if len(self._episodes) <= self.capacity:
            return
        # evict oldest unpinned until under capacity
        while len(self._episodes) > self.capacity:
            cands = [(k, v) for k, v in self._episodes.items() if not v.pinned]
            if not cands:
                break
            oldest_key = min(cands, key=lambda kv: kv[1].last_used)[0]
            del self._episodes[oldest_key]

    def search_top_k(self, q_emb, k):
        hits = self.search_dense(q_emb, k)
        return [(ep.key_text, float(sc)) for ep, sc in hits]

    def get_text(self, key_text):
        ep = self._episodes.get(key_text)
        if ep is None:
            return ""
        return ep.key_text + " " + ep.value_text

    def reset(self):
        self._episodes.clear()
        self._counter = 0
