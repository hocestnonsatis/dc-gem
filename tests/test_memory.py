"""Memory + retrieval tests (CPU, no model download)."""

import torch
import re
from dc_gem.memory import EpisodicMemory
from dc_gem.retrieval import lexical_score, extract_code


def test_memory_write_read():
    mem = EpisodicMemory(size=8, d_k=16, d_v=16, K=4, tau=0.1)
    key = torch.randn(16)
    val = torch.randn(16)
    mem.write_slot(0, key, val, text="fact A")
    assert mem.n_valid == 1
    q = torch.randn(1, 16)
    # query_batch not present in this memory (search_top_k instead); test search
    hits = mem.search_top_k(q.squeeze(0), 2)
    assert len(hits) == 1
    assert mem.get_text(hits[0][0]) == "fact A"


def test_memory_multiple_and_reset():
    mem = EpisodicMemory(size=8, d_k=16, d_v=16, K=4, tau=0.1)
    for i in range(5):
        mem.write_slot(i, torch.randn(16), torch.randn(16), text=f"fact {i}")
    assert mem.n_valid == 5
    hits = mem.search_top_k(torch.randn(16), 3)
    assert len(hits) == 3
    mem.reset()
    assert mem.n_valid == 0


def test_lexical_score():
    assert lexical_score("password castle Z05", "The secret password of castle Z05 is 1185.") > 0.5
    assert lexical_score("apple", "banana orange") == 0.0


def test_extract_code():
    assert extract_code(" 1037\nThis answer") == "1037"
    assert extract_code("no digits here") is None
