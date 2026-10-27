"""LoRA adapters for pre-trained models."""

from __future__ import annotations
import torch
import torch.nn as nn


class LoRAWrapper(nn.Module):
    def __init__(self, base: nn.Linear, rank: int = 8, alpha: float = 16.0, dropout: float = 0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad = False
        in_dim = base.in_features
        out_dim = base.out_features
        self.rank = rank
        self.scale = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        dev = base.weight.device
        dtype = base.weight.dtype
        self.lora_A = nn.Parameter(torch.zeros(rank, in_dim, device=dev, dtype=dtype))
        self.lora_B = nn.Parameter(torch.zeros(out_dim, rank, device=dev, dtype=dtype))
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        base_out = self.base(x)
        x_d = self.dropout(x)
        lora_out = (x_d @ self.lora_A.t() @ self.lora_B.t()) * self.scale
        return base_out + lora_out


def attach_lora(model, rank=8, alpha=16.0, targets=("q_proj", "v_proj"), max_layers=None, dropout=0.0):
    """Attach LoRA to target modules in all transformer layers. Returns list of params."""
    params = []
    # determine which layer indices to adapt
    n_layers = len(model.model.layers) if hasattr(model, 'model') and hasattr(model.model, 'layers') else None
    allowed = None
    if max_layers is not None and n_layers is not None:
        allowed = set(range(n_layers - max_layers, n_layers))
    for name, module in list(model.named_modules()):
        if not any(name.endswith(t) for t in targets):
            continue
        # filter by layer index if requested
        m = None
        import re as _re
        mm = _re.search(r'model\.layers\.(\d+)\.', name)
        if allowed is not None and mm:
            if int(mm.group(1)) not in allowed:
                continue
        parent_name = name.rsplit(".", 1)[0]
        attr = name.rsplit(".", 1)[1]
        parent = model.get_submodule(parent_name)
        wrapped = LoRAWrapper(module, rank=rank, alpha=alpha, dropout=dropout)
        setattr(parent, attr, wrapped)
        params += [wrapped.lora_A, wrapped.lora_B]
    return params


def lora_state_dict(model):
    sd = {}
    for name, mod in model.named_modules():
        if isinstance(mod, LoRAWrapper):
            sd[name + ".lora_A"] = mod.lora_A.detach().cpu()
            sd[name + ".lora_B"] = mod.lora_B.detach().cpu()
    return sd


def load_lora_state_dict(model, sd):
    for name, mod in model.named_modules():
        if isinstance(mod, LoRAWrapper):
            if name + ".lora_A" in sd:
                mod.lora_A.data.copy_(sd[name + ".lora_A"].to(mod.lora_A.device, mod.lora_A.dtype))
            if name + ".lora_B" in sd:
                mod.lora_B.data.copy_(sd[name + ".lora_B"].to(mod.lora_B.device, mod.lora_B.dtype))
