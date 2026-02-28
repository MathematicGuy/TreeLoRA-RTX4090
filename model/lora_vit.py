"""
LoRA injection for Vision Transformers (ViT).

Injects LoRA adapters into ViT attention Q and V projections using the
`loranew_A` / `loranew_B` naming convention so they are directly compatible
with the existing KD_LoRA_Tree gradient-extraction logic:

    for name, param in model.named_parameters():
        if "loranew_A" in name:   <- picks up every LoRA A matrix
            ...

For ViT-B/16 (12 transformer blocks) with LoRA on Q and V:
    12 blocks × 2 projections = 24 loranew_A matrices  ->  lora_depth = 24

Usage
-----
    model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=0)
    model = inject_lora_vit(model, r=8, lora_alpha=32, lora_dropout=0.1)
    freeze_non_lora(model)
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ---------------------------------------------------------------------------
# Core LoRA linear layer
# ---------------------------------------------------------------------------

class LoRALinear(nn.Module):
    """
    Wraps a frozen nn.Linear and adds two LoRA adapter pairs:

    * ``lora_A`` / ``lora_B``      – frozen weights from the *previous* task
    * ``loranew_A`` / ``loranew_B``– trainable weights for the *current* task

    Forward:  y = x W^T + x A^T B^T * scaling   (prev, frozen)
                        + x loranew_A^T loranew_B^T * scaling  (current, trainable)
    """

    def __init__(
        self,
        linear: nn.Linear,
        r: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        in_features  = linear.in_features
        out_features = linear.out_features

        # Keep original weights (frozen)
        self.weight = linear.weight          # (out, in)
        self.bias   = linear.bias            # (out,) or None
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        self.r       = r
        self.scaling = lora_alpha / r
        self.dropout = nn.Dropout(p=lora_dropout)

        # Previous-task LoRA (frozen, starts at zero – becomes active after first task)
        self.lora_A = nn.Parameter(torch.zeros(r, in_features),  requires_grad=False)
        self.lora_B = nn.Parameter(torch.zeros(out_features, r), requires_grad=False)

        # Current-task LoRA (trainable)
        self.loranew_A = nn.Parameter(torch.zeros(r, in_features))
        self.loranew_B = nn.Parameter(torch.zeros(out_features, r))

        # Kaiming init for loranew_A, zeros for loranew_B (standard LoRA init)
        nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
        nn.init.zeros_(self.loranew_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Base frozen linear
        result = F.linear(x, self.weight, self.bias)

        xd = self.dropout(x)

        # Cast frozen weights to match input dtype (e.g. BF16 / FP16 under autocast)
        dtype = xd.dtype

        # Previous task contribution (frozen, nonzero after task 0 is committed)
        result = result + (xd @ self.lora_A.to(dtype).T @ self.lora_B.to(dtype).T) * self.scaling

        # Current task contribution (trainable)
        result = result + (xd @ self.loranew_A.to(dtype).T @ self.loranew_B.to(dtype).T) * self.scaling

        return result

    def commit_task(self):
        """
        Called after a task finishes.  Merges loranew into lora so that the
        learned adapters are preserved but frozen, and resets loranew for
        the next task.

        The merge formula accumulates: A_prev <- A_prev + A_new (additive LoRA).
        A full implementation can store per-task adapters; here we keep it
        simple and just copy the new weights.
        """
        with torch.no_grad():
            self.lora_A.data.copy_(self.loranew_A.data)
            self.lora_B.data.copy_(self.loranew_B.data)
            # Re-init loranew for next task
            nn.init.kaiming_uniform_(self.loranew_A, a=math.sqrt(5))
            nn.init.zeros_(self.loranew_B)

    def extra_repr(self) -> str:
        return (
            f"in={self.weight.shape[1]}, out={self.weight.shape[0]}, "
            f"r={self.r}, scaling={self.scaling:.3f}"
        )


# ---------------------------------------------------------------------------
# Inject LoRA into a timm ViT model
# ---------------------------------------------------------------------------

def inject_lora_vit(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
    target_modules: tuple = ("qkv",),          # timm ViT uses a single fused qkv
) -> nn.Module:
    """
    Replace target linear layers in a timm ViT with LoRALinear.

    timm's ``vit_base_patch16_224`` uses a *fused* ``qkv`` projection of shape
    ``(3*hidden, hidden)``.  We replace the whole fused layer; at tree-search
    time each replacement contributes one ``loranew_A`` to the gradient tensor,
    so for ViT-B/16 we get 12 ``loranew_A`` matrices.

    If ``"q_proj"`` and ``"v_proj"`` are in ``target_modules`` (HuggingFace ViT),
    each block contributes 2 matrices → 24 total.

    Parameters
    ----------
    model : nn.Module
        Pre-trained ViT (timm or HuggingFace).
    r : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha scaling.
    lora_dropout : float
        Dropout rate applied to LoRA inputs.
    target_modules : tuple[str]
        Names of sub-modules to replace (must be ``nn.Linear``).

    Returns
    -------
    nn.Module
        The same model with LoRA layers injected in-place.
    """
    replaced = 0
    for name, module in list(model.named_modules()):
        # Check if this module's last component matches a target
        short_name = name.split(".")[-1]
        if short_name not in target_modules:
            continue
        if not isinstance(module, nn.Linear):
            continue

        # Navigate to parent and replace
        parent = _get_parent(model, name)
        attr   = name.split(".")[-1]
        lora_layer = LoRALinear(module, r=r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
        setattr(parent, attr, lora_layer)
        replaced += 1

    print(f"[inject_lora_vit] Replaced {replaced} linear layers with LoRALinear "
          f"(target_modules={target_modules})")
    return model


def inject_lora_vit_hf(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """
    LoRA injection specifically for HuggingFace ``ViTForImageClassification``.

    Targets ``query`` and ``value`` projections inside each
    ``ViTSelfAttention`` block → 12 blocks × 2 = **24** ``loranew_A``
    matrices → ``lora_depth = 24``.
    """
    return inject_lora_vit(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=("query", "value"),
    )


def inject_lora_vit_timm(
    model: nn.Module,
    r: int = 8,
    lora_alpha: int = 32,
    lora_dropout: float = 0.1,
) -> nn.Module:
    """
    LoRA injection specifically for timm ViTs (fused ``qkv`` projection).

    12 blocks × 1 fused projection = **12** ``loranew_A`` matrices
    → ``lora_depth = 12``.
    """
    return inject_lora_vit(
        model,
        r=r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=("qkv",),
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_parent(model: nn.Module, dotpath: str) -> nn.Module:
    """Return the parent module of a dotpath-addressed child."""
    parts = dotpath.split(".")
    parent = model
    for part in parts[:-1]:
        parent = getattr(parent, part)
    return parent


def freeze_non_lora(model: nn.Module) -> None:
    """
    Freeze all parameters whose name does NOT contain ``loranew``.
    Classification head(s) are kept trainable.
    """
    for name, param in model.named_parameters():
        if "loranew" in name or "head" in name or "classifier" in name:
            param.requires_grad_(True)
        else:
            param.requires_grad_(False)


def commit_all_lora(model: nn.Module) -> None:
    """
    After finishing a task, copy loranew_A/B → lora_A/B (freeze) and
    re-init loranew_A/B for the next task.
    """
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.commit_task()


def count_trainable_params(model: nn.Module) -> dict:
    """Return counts of trainable and total parameters."""
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total     = sum(p.numel() for p in model.parameters())
    return {"trainable": trainable, "total": total, "ratio": trainable / total}
