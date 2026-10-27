"""Retrieval helpers: dense + lexical rerank."""

from __future__ import annotations
import re
import torch


def lexical_score(query: str, text: str) -> float:
    """Token overlap score (case-insensitive)."""
    qtoks = set(re.findall(r"\w+", query.lower()))
    ttoks = set(re.findall(r"\w+", text.lower()))
    if not qtoks or not ttoks:
        return 0.0
    return len(qtoks & ttoks) / len(qtoks)


def hybrid_retrieve(mem, q_emb, query_text, k_dense: int = 16, top_n: int = 1):
    """Dense top-k then lexical rerank. Returns list of (slot, score, text)."""
    hits = mem.search_top_k(q_emb, k_dense)
    # rerank by lexical overlap
    scored = []
    for slot, dscore in hits:
        t = mem.get_text(slot)
        ls = lexical_score(query_text, t)
        # combined: dense + lexical boost
        combined = dscore + 0.5 * ls
        scored.append((slot, combined, t, dscore, ls))
    scored.sort(key=lambda x: -x[1])
    return scored[:top_n]


def extract_code(text: str) -> str | None:
    m = re.search(r"\b(\d{3,6})\b", text)
    return m.group(1) if m else None


import math


def char_ngrams(text: str, n: int = 3) -> list[str]:
    t = text.lower()
    return [t[i:i+n] for i in range(len(t) - n + 1)] if len(t) >= n else [t]


def bm25_score(query: str, doc: str, k1: float = 1.5, b: float = 0.75,
               avgdl: float = 50.0) -> float:
    q_terms = query.lower().split()
    d_terms = doc.lower().split()
    dl = len(d_terms)
    tf_map = {}
    for t in d_terms:
        tf_map[t] = tf_map.get(t, 0) + 1
    score = 0.0
    for q in q_terms:
        tf = tf_map.get(q, 0)
        if tf == 0:
            continue
        idf = math.log(1 + 1.0 / (0.5))  # simplified idf
        denom = tf + k1 * (1 - b + b * dl / avgdl)
        score += idf * tf * (k1 + 1) / denom
    return score


def lexical_char_ngram_score(query: str, doc: str, n: int = 3) -> float:
    qn = set(char_ngrams(query, n))
    dn = set(char_ngrams(doc, n))
    if not qn or not dn:
        return 0.0
    return len(qn & dn) / len(qn)


def score_ablation(q_emb, query_text, candidates, embed_fn=None):
    """Return dict of scores per method for ablation.

    candidates: list of (episode_or_text, embedding)
    Returns: {method: [(item, score), ...]} sorted desc.
    """
    # cos-only
    q = q_emb.detach().cpu().reshape(-1).float()
    qn = q / q.norm().clamp_min(1e-8)
    cos_scored = []
    lex_scored = []
    hybrid_scored = []
    for item, emb in candidates:
        text = item if isinstance(item, str) else getattr(item, "key_text", "") + " " + getattr(item, "value_text", "")
        # cos
        e = emb.detach().cpu().reshape(-1).float()
        e = e / e.norm().clamp_min(1e-8)
        cos = float(qn @ e)
        # lex
        lex = lexical_char_ngram_score(query_text, text)
        # bm25
        bm = bm25_score(query_text, text)
        hybrid = 0.5 * cos + 0.5 * (lex + bm) / 2.0
        cos_scored.append((item, cos))
        lex_scored.append((item, lex))
        hybrid_scored.append((item, hybrid))
    cos_scored.sort(key=lambda x: -x[1])
    lex_scored.sort(key=lambda x: -x[1])
    hybrid_scored.sort(key=lambda x: -x[1])
    return {"cos_only": cos_scored, "lex_only": lex_scored, "hybrid": hybrid_scored}
