"""Lightweight, backbone-agnostic PEFT helpers for MEG-Mamba fine-tuning.

Two primitives:
  - freeze_backbone(model, train=(...))  : freeze everything, re-enable params whose
                                           qualified name contains a `train` substring.
  - add_lora(model, targets=(...))       : wrap matched nn.Linear submodules with a
                                           low-rank trainable update (base stays frozen).

For Mamba the right LoRA targets are the projection Linears (`in_proj`, `out_proj`),
not the scan internals (A/B/C/dt/conv). NB an input-side adapter or LoRA on `in_proj`
still needs gradients to flow back through the selective scan; if scan-backward is
unsupported in your env (see the analysis project's check_backward.py), restrict LoRA
to `out_proj` (+ ffn/head), which sit after the scan.

No external dependency (not `peft`); ~1 file so you keep full control over which
modules are adapted in this custom mamba_ssm fork.
"""

import torch
import torch.nn as nn


def freeze_backbone(model, train=()):
    """Freeze all params, then re-enable any whose qualified name contains a `train` token.

    Returns the list of trainable parameters (for the optimizer).
    Example: freeze_backbone(model, train=("stim_emb", "head")).
    """
    for p in model.parameters():
        p.requires_grad_(False)
    for name, p in model.named_parameters():
        if any(t in name for t in train):
            p.requires_grad_(True)
    return [p for p in model.parameters() if p.requires_grad]


class LoRALinear(nn.Module):
    """Frozen nn.Linear + low-rank trainable update: y = base(x) + (dropout(x) Aᵀ) Bᵀ · scale.

    A is kaiming-init, B is zero → ΔW = 0 at init, so the model output is unchanged
    until training moves B. The base Linear (weight + bias) is frozen.
    """

    def __init__(self, base: nn.Linear, rank=8, alpha=16, dropout=0.0):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.rank = int(rank)
        self.scale = alpha / rank
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Named lora_A/lora_B (not A/B) so a 'lora_' name-token selects them without
        # colliding with mamba params like 'A_log'.
        self.lora_A = nn.Parameter(torch.zeros(self.rank, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, self.rank))
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)

    @property
    def weight(self):
        # Some backbones read `.weight.device/.dtype` directly (e.g. Mamba3 inference-cache
        # allocation). Expose the base weight so those lookups keep working after wrapping.
        return self.base.weight

    def forward(self, x):
        return (
            self.base(x)
            + (self.drop(x) @ self.lora_A.t() @ self.lora_B.t()) * self.scale
        )


def add_lora(model, targets=("in_proj", "out_proj"), rank=8, alpha=16, dropout=0.0):
    """Wrap every nn.Linear whose qualified name contains a `targets` token with LoRALinear.

    Returns the number of Linears adapted. Afterwards select the trainable params
    with freeze_backbone(model, train=("lora_", ...)) — the 'lora_' token matches the
    lora_A/lora_B params (and not mamba's 'A_log'). Matching is by qualified module
    path, so 'out_proj' matches e.g. 'mamba_blocks.0.out_proj' but not 'head'.
    """
    n = 0
    for mod_name, module in list(model.named_modules()):
        for child_name, child in list(module.named_children()):
            full = f"{mod_name}.{child_name}" if mod_name else child_name
            if isinstance(child, nn.Linear) and any(t in full for t in targets):
                setattr(
                    module,
                    child_name,
                    LoRALinear(child, rank=rank, alpha=alpha, dropout=dropout),
                )
                n += 1
    return n
