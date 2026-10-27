#!/usr/bin/env python3
"""DC-GEM MCP Server — exposes episodic memory operations as MCP tools."""

from __future__ import annotations
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from mcp.server.mcpserver import MCPServer as FastMCP

# Lazy-loaded models
_core = None
_embed = None
_store = None

def _get_models():
    global _core, _embed, _store
    if _core is None:
        from dc_gem.core import PretrainedCore
        from dc_gem.embeddings import EmbeddingModel
        from dc_gem.memory import Store
        _core = PretrainedCore("Qwen/Qwen3-0.6B", "cuda")
        _embed = EmbeddingModel("BAAI/bge-small-en-v1.5", "cuda")
        _store = Store(capacity=4096)
    return _core, _embed, _store

mcp = FastMCP("dc-gem")

@mcp.tool()
def dc_gem_write_fact(key: str, value: str, pinned: bool = False) -> str:
    """Write a fact (key-value pair) into episodic memory. Overwrites if key exists."""
    core, embed, store = _get_models()
    text = f"{key}: {value}"
    emb = embed.encode(text)
    ep = store.write("Fact", text, value, emb, pinned=pinned)
    return json.dumps({"id": ep.id, "key": key, "value": value, "pinned": pinned, "n_facts": len(store.all())})

@mcp.tool()
def dc_gem_search_facts(query: str, top_k: int = 4) -> str:
    """Search episodic memory for facts relevant to a query. Returns top-k matches."""
    core, embed, store = _get_models()
    q_emb = embed.encode(query, is_query=True)
    hits = store.search_dense(q_emb, top_k)
    results = []
    for ep, score in hits:
        results.append({"key": ep.key_text, "value": ep.value_text, "score": round(score, 4), "pinned": ep.pinned})
    return json.dumps({"query": query, "results": results, "n_total": len(store.all())})

@mcp.tool()
def dc_gem_list_facts() -> str:
    """List all facts currently stored in episodic memory."""
    core, embed, store = _get_models()
    facts = [{"key": ep.key_text, "value": ep.value_text, "pinned": ep.pinned} for ep in store.all()]
    return json.dumps({"n_facts": len(facts), "facts": facts})

@mcp.tool()
def dc_gem_clear_memory() -> str:
    """Clear all non-pinned facts from episodic memory."""
    core, embed, store = _get_models()
    n_before = len(store.all())
    store.reset()
    return json.dumps({"cleared": n_before, "remaining": len(store.all())})

@mcp.tool()
def dc_gem_generate(question: str, use_memory: bool = True) -> str:
    """Generate an answer to a question, optionally using episodic memory for context."""
    core, embed, store = _get_models()
    if use_memory and len(store.all()) > 0:
        q_emb = embed.encode(question, is_query=True)
        hits = store.search_dense(q_emb, 4)
        ctx = "\n".join(f"{ep.key_text}: {ep.value_text}" for ep, _ in hits)
        prompt = f"Fact: {ctx}\nQuestion: {question}\nAnswer:"
    else:
        prompt = f"Question: {question}\nAnswer:"
    answer = core.generate(prompt, max_new_tokens=64, temperature=0.0)
    return json.dumps({"question": question, "use_memory": use_memory, "answer": answer.strip()})

if __name__ == "__main__":
    mcp.run()
