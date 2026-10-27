"""Injection channels: text (prompt) and hidden (m_t adapter)."""

from __future__ import annotations
import hashlib
import torch


def inject_text(retrieved_texts: list[str]) -> tuple[str, str]:
    """Build a <memory> block from retrieved texts. Returns (block_text, hash)."""
    lines = ["<memory>"]
    for t in retrieved_texts:
        lines.append(t.strip())
    lines.append("</memory>")
    block = "\n".join(lines)
    h = hashlib.sha256(block.encode()).hexdigest()[:16]
    return block, h


def inject_hidden(m_t: torch.Tensor, adapter: torch.nn.Module | None = None) -> torch.Tensor:
    """Adapt retrieved memory vector m_t via an adapter (or identity)."""
    if adapter is None:
        return m_t
    return adapter(m_t)


class HiddenAdapter(torch.nn.Module):
    """Small MLP adapter for hidden-channel injection."""

    def __init__(self, d_in: int, d_out: int):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(d_in, d_out),
            torch.nn.GELU(),
            torch.nn.Linear(d_out, d_out),
        )

    def forward(self, x):
        return self.net(x)
