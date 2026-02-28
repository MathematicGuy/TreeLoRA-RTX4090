"""
Tree_LoRA_ViT – TreeLoRA for Vision Transformers (task-incremental CL).

This module adapts the LLM TreeLoRA algorithm to ViT image classification
without any DeepSpeed dependency.  The core gradient-tree logic is reused
unchanged from ``utils.kd_lora_tree.KD_LoRA_Tree``.

Architecture
------------
                ┌─────────────────────────────┐
                │  Frozen ViT backbone         │
                │  + LoRALinear adapters        │
                │    loranew_A / loranew_B      │  ← trainable each task
                └─────────────┬───────────────┘
                              │  features
                ┌─────────────▼───────────────┐
                │  Task heads  (one per task)  │  ← always trainable
                │  nn.Linear(hidden, n_cls_t)  │
                └─────────────────────────────┘

Training flow per task t
------------------------
1.  Forward:  loss = CE(head_t(features), labels)
2.  Backward: loss.backward()   (standard PyTorch, no DeepSpeed)
3.  Extract loranew_A param values as gradient proxy
4.  Insert into KD_LoRA_Tree accumulator
5.  For t > 0: tree_search → compute reg_loss → backward reg_loss
6.  optimizer.step() / scheduler.step()
7.  After last epoch: kd_lora_tree.end_task(t); commit_all_lora(model)

Evaluation
----------
accuracy_t = number_correct / total  on test_task_list[task_name]
Two metrics are reported:
* OP  – average accuracy over all seen tasks after training task T
* BWT – average forgetting  (acc at training time − acc at test time)
"""

import os
import copy
import json
import math
import time
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.kd_lora_tree import KD_LoRA_Tree
from model.lora_vit import commit_all_lora, count_trainable_params, LoRALinear


# ---------------------------------------------------------------------------
# Task-head manager
# ---------------------------------------------------------------------------

class TaskHeadManager(nn.Module):
    """Maintains one linear classification head per task."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.heads = nn.ModuleList()

    def add_task(self, n_classes: int, device: torch.device):
        head = nn.Linear(self.hidden_dim, n_classes).to(device)
        nn.init.trunc_normal_(head.weight, std=0.02)
        nn.init.zeros_(head.bias)
        self.heads.append(head)

    def forward(self, features: torch.Tensor, task_id: int) -> torch.Tensor:
        return self.heads[task_id](features)

    def parameters_for_task(self, task_id: int):
        return self.heads[task_id].parameters()


# ---------------------------------------------------------------------------
# Main class
# ---------------------------------------------------------------------------

class Tree_LoRA_ViT:
    """
    TreeLoRA continual-learning trainer for ViT image classifiers.

    Parameters
    ----------
    model : nn.Module
        ViT backbone with LoRA layers already injected
        (``model.lora_vit.inject_lora_vit_hf``).
    optimizer_cls : type
        Optimizer class (e.g. ``torch.optim.AdamW``).
    optimizer_kwargs : dict
        Keyword arguments forwarded to ``optimizer_cls``.
    train_task_list : Dict[str, DataLoader]
        ``{"task_0": ..., "task_1": ..., ...}``
    val_task_list : Dict[str, DataLoader]
    test_task_list : Dict[str, DataLoader]
    class_masks : List[List[int]]
        Original class IDs per task (length = n_tasks).
    args : argparse.Namespace
        Must contain at minimum:
        - ``reg``          float – regularisation strength (0 = disabled)
        - ``lora_depth``   int   – tree depth (= number of loranew_A layers)
        - ``num_tasks``    int
        - ``seed``         int
        - ``output_dir``   str
        - ``device``       str   e.g. "cuda"
    lamda_1 : float
        Weight of the similarity / OPL regularisation loss.
    """

    def __init__(
        self,
        model: nn.Module,
        optimizer_cls,
        optimizer_kwargs: dict,
        train_task_list: Dict[str, DataLoader],
        val_task_list:   Dict[str, DataLoader],
        test_task_list:  Dict[str, DataLoader],
        class_masks:     List[List[int]],
        args,
        lamda_1: float = 0.5,
    ):
        self.model           = model
        self.optimizer_cls   = optimizer_cls
        self.optimizer_kwargs = optimizer_kwargs
        self.train_task_list  = train_task_list
        self.val_task_list    = val_task_list
        self.test_task_list   = test_task_list
        self.class_masks      = class_masks
        self.args             = args
        self.lamda_1          = getattr(args, 'lamda_1', lamda_1)

        self.device = torch.device(getattr(args, 'device', 'cuda')
                                   if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)

        # ── Task heads ──────────────────────────────────────────────────
        # Infer hidden_dim from the model
        self.hidden_dim = self._get_hidden_dim()
        self.head_manager = TaskHeadManager(self.hidden_dim).to(self.device)

        # ── KD-LoRA tree ─────────────────────────────────────────────────
        args.num_tasks  = len(train_task_list)
        args.global_rank = 0                         # single-GPU, no dist
        args.opl_weight  = getattr(args, 'opl_weight', 0.1)
        args.use_opl     = getattr(args, 'use_opl',    True)
        self.kd_lora_tree = KD_LoRA_Tree(args)

        # ── AMP (mixed precision) ────────────────────────────────────────
        self.use_amp = (getattr(args, 'use_amp', False)
                        and torch.cuda.is_available())
        self.scaler  = torch.amp.GradScaler('cuda', enabled=self.use_amp)

        # ── Bookkeeping ──────────────────────────────────────────────────
        self.acc_matrix: Dict[int, Dict[int, float]] = {}   # acc_matrix[after_task][on_task]
        self.best_acc_per_task: Dict[int, float] = {}       # recorded right after training

        print(f"\n[Tree_LoRA_ViT] Initialised")
        print(f"  device    : {self.device}")
        print(f"  hidden_dim: {self.hidden_dim}")
        print(f"  num_tasks : {args.num_tasks}")
        print(f"  lora_depth: {getattr(args, 'lora_depth', -1)}")
        print(f"  reg       : {args.reg}")
        print(f"  lamda_1   : {self.lamda_1}")
        print(f"  use_amp   : {self.use_amp}")
        param_info = count_trainable_params(self.model)
        print(f"  backbone trainable params: {param_info['trainable']:,} "
              f"/ {param_info['total']:,} ({100*param_info['ratio']:.2f}%)")

    # ------------------------------------------------------------------ #
    #  Internal helpers                                                    #
    # ------------------------------------------------------------------ #

    def _get_hidden_dim(self) -> int:
        """Infer the feature dimension from the last linear / norm layer."""
        # HuggingFace ViTForImageClassification
        if hasattr(self.model, 'vit') and hasattr(self.model.vit, 'layernorm'):
            return self.model.vit.layernorm.normalized_shape[0]
        # HuggingFace ViTModel (bare, no classification head)
        if hasattr(self.model, 'layernorm') and hasattr(self.model, 'encoder'):
            return self.model.layernorm.normalized_shape[0]
        # timm ViT
        if hasattr(self.model, 'norm'):
            return self.model.norm.normalized_shape[0]
        # fallback: last named parameter
        for _, p in reversed(list(self.model.named_parameters())):
            return p.shape[-1]

    def _extract_lora_gradients(self) -> List[torch.Tensor]:
        """Collect loranew_A parameter tensors (used as gradient proxy)."""
        grads = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                grads.append(param)
        return grads

    def _compute_gradient_tensor(
        self, grad_list: List[torch.Tensor]
    ) -> Optional[torch.Tensor]:
        """Stack loranew_A tensors → (lora_depth, flattened_dim)."""
        if not grad_list:
            return None
        return torch.stack([g.reshape(-1).detach() for g in grad_list], dim=0)

    def _get_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Forward-pass without the classification head; returns [CLS] features.

        Handles both HuggingFace and timm ViTs.
        """
        # HuggingFace ViTForImageClassification (has a .vit sub-model)
        if hasattr(self.model, 'vit'):
            out = self.model.vit(images)
            # last_hidden_state[:, 0] is the [CLS] token
            return out.last_hidden_state[:, 0, :]

        # HuggingFace ViTModel (bare, no classification head)
        if hasattr(self.model, 'encoder') and hasattr(self.model, 'layernorm'):
            out = self.model(pixel_values=images)
            return out.last_hidden_state[:, 0, :]

        # timm: forward_features returns (B, hidden) after global pool
        if hasattr(self.model, 'forward_features'):
            return self.model.forward_features(images)

        raise RuntimeError("Cannot extract features – unknown model architecture.")

    def _build_optimizer(self, task_id: int):
        """
        Build a fresh AdamW optimizer for the current task covering:
        * all  loranew_A / loranew_B  parameters (backbone LoRA)
        * the  task head  for task_id
        """
        lora_params = [
            p for n, p in self.model.named_parameters()
            if "loranew" in n and p.requires_grad
        ]
        head_params = list(self.head_manager.parameters_for_task(task_id))

        all_params = lora_params + head_params
        print(f"  [optimizer] {len(lora_params)} LoRA params + "
              f"{len(head_params)} head params = {len(all_params)} total groups")

        return self.optimizer_cls(all_params, **self.optimizer_kwargs)

    # ------------------------------------------------------------------ #
    #  Per-task training                                                   #
    # ------------------------------------------------------------------ #

    def train_one_task(self, task: str, task_id: int, epochs: int):
        """
        Train on a single task with TreeLoRA regularisation.

        Steps mirror the LLM version (Tree_LoRA.train_one_task) but use
        standard PyTorch backward instead of DeepSpeed.
        """
        n_classes = len(self.class_masks[task_id])
        self.head_manager.add_task(n_classes, self.device)

        train_dl  = self.train_task_list[task]
        val_dl    = self.val_task_list[task]

        optimizer = self._build_optimizer(task_id)

        # Paper specifies a constant learning rate (no decay schedule)
        scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, factor=1.0, total_iters=0)

        print(f"\n{'='*60}")
        print(f"[Tree_LoRA_ViT] Task {task_id}: {task}")
        print(f"  epochs={epochs}, steps/epoch={len(train_dl)}, "
              f"n_classes={n_classes}")
        print(f"{'='*60}")

        for epoch in range(epochs):
            self.model.train()
            self.head_manager.train()

            # Re-init tree accumulator each epoch
            self.kd_lora_tree.new_epoch_init(len(train_dl))

            pbar = tqdm(train_dl, desc=f"Task {task_id} Ep {epoch+1}/{epochs}",
                        leave=False)

            epoch_task_loss = 0.0
            epoch_reg_loss  = 0.0

            for step, (images, labels) in enumerate(pbar):
                images  = images.to(self.device, non_blocking=True)
                labels  = labels.to(self.device, non_blocking=True)

                if self.args.reg > 0:
                    self.kd_lora_tree.step()

                # ── Forward (inside autocast for FP16 AMP) ───────────────
                with torch.amp.autocast('cuda', enabled=self.use_amp): # type: ignore
                    features = self._get_features(images)
                    logits   = self.head_manager(features, task_id)
                    loss     = F.cross_entropy(logits, labels)

                epoch_task_loss += loss.item()

                # ── TreeLoRA regularisation ───────────────────────────────
                if self.args.reg > 0:
                    _grad_current = self._extract_lora_gradients()

                    if _grad_current:
                        _grad_tensor = self._compute_gradient_tensor(_grad_current)
                        self.kd_lora_tree.insert_grad(_grad_tensor)

                        if task_id > 0:
                            prev_id_matrix = self.kd_lora_tree.tree_search(
                                task_id, device=self.device)

                            reg_loss = self.kd_lora_tree.get_loss(
                                _grad_tensor, loss, task_id, prev_id_matrix)

                            loss = loss - self.lamda_1 * reg_loss
                            epoch_reg_loss += reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss

                # ── Backward with AMP scaler ─────────────────────────────
                optimizer.zero_grad()
                self.scaler.scale(loss).backward()

                # Gradient clipping (matches original setup: gradient_clipping=1.0)
                self.scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    [p for p in self.model.parameters() if p.requires_grad], 1.0)

                self.scaler.step(optimizer)
                self.scaler.update()
                scheduler.step()

                pbar.set_postfix(loss=f"{loss.item():.4f}")

            # Epoch summary
            n = len(train_dl)
            print(f"  Epoch {epoch+1}/{epochs} | task_loss={epoch_task_loss/n:.4f} "
                  f"| reg_loss={epoch_reg_loss/n:.4f}")

            # Validation accuracy
            val_acc = self.evaluate_task(task, task_id, val_dl)
            print(f"  Val acc (task {task_id}): {val_acc*100:.2f}%")

        # ── Post-task bookkeeping ───────────────────────────────────────
        # Record best accuracy
        test_acc = self.evaluate_task(task, task_id, self.test_task_list[task])
        self.best_acc_per_task[task_id] = test_acc

        # Update tree
        if self.args.reg > 0:
            self.kd_lora_tree.end_task(task_id=task_id)

        # Freeze current LoRA, reset for next task
        commit_all_lora(self.model)

        # Save checkpoint
        self._save_checkpoint(task_id)

        print(f"[Tree_LoRA_ViT] Task {task_id} done | test_acc={test_acc*100:.2f}%")

    # ------------------------------------------------------------------ #
    #  Evaluation                                                          #
    # ------------------------------------------------------------------ #

    @torch.no_grad()
    def evaluate_task(
        self,
        task: str,
        task_id: int,
        dataloader: DataLoader,
    ) -> float:
        """Return top-1 accuracy on ``dataloader`` using head ``task_id``."""
        self.model.eval()
        self.head_manager.eval()

        correct = total = 0
        for images, labels in dataloader:
            images  = images.to(self.device, non_blocking=True)
            labels  = labels.to(self.device, non_blocking=True)

            features = self._get_features(images)
            logits   = self.head_manager(features, task_id)
            preds    = logits.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)

        self.model.train()
        self.head_manager.train()
        return correct / max(total, 1)

    def evaluate_all(self, after_task_id: int) -> Dict[str, float]:
        """
        Evaluate all tasks seen so far; returns per-task accuracy dict plus
        OP (overall performance) and BWT (backward transfer).
        """
        results: Dict[str, float] = {}
        accs: List[float] = []

        task_names = list(self.test_task_list.keys())

        for tid in range(after_task_id + 1):
            tname = task_names[tid]
            acc   = self.evaluate_task(tname, tid, self.test_task_list[tname])
            results[tname] = acc
            accs.append(acc)

        self.acc_matrix[after_task_id] = {
            tid: accs[tid] for tid in range(after_task_id + 1)
        }

        op = float(np.mean(accs)) if accs else 0.0
        results["OP"] = op

        # BWT = mean( acc_after_all_tasks[i] - acc_right_after_task_i )
        if after_task_id > 0:
            bwt_terms = []
            for tid in range(after_task_id):
                final_acc = accs[tid]
                best_acc  = self.best_acc_per_task.get(tid, final_acc)
                bwt_terms.append(final_acc - best_acc)
            results["BWT"] = float(np.mean(bwt_terms))
        else:
            results["BWT"] = 0.0

        return results

    # ------------------------------------------------------------------ #
    #  Full CL sequence                                                    #
    # ------------------------------------------------------------------ #

    def train_continual(self, epochs_per_task: List[int]):
        """
        Run the full continual-learning sequence.

        Parameters
        ----------
        epochs_per_task : List[int]
            Number of training epochs for each task.  Length must equal
            ``len(train_task_list)``.
        """
        task_names = list(self.train_task_list.keys())
        assert len(epochs_per_task) == len(task_names), (
            f"epochs_per_task has {len(epochs_per_task)} entries but there are "
            f"{len(task_names)} tasks")

        all_results = {}

        for task_id, (task_name, epochs) in enumerate(
                zip(task_names, epochs_per_task)):

            self.train_one_task(task_name, task_id, epochs)

            # Evaluate on all tasks seen so far
            eval_results = self.evaluate_all(task_id)
            all_results[f"after_task_{task_id}"] = eval_results

            print(f"\n[Tree_LoRA_ViT] Results after task {task_id} ({task_name}):")
            for k, v in eval_results.items():
                print(f"  {k}: {v*100:.2f}%" if isinstance(v, float) else f"  {k}: {v}")

        # Save final results
        if self.args.output_dir:
            rpath = os.path.join(self.args.output_dir, "results.json")
            os.makedirs(self.args.output_dir, exist_ok=True)
            with open(rpath, "w") as fh:
                json.dump(_to_serializable(all_results), fh, indent=2)
            print(f"\n[Tree_LoRA_ViT] Results saved to {rpath}")

        return all_results

    # ------------------------------------------------------------------ #
    #  Checkpoint                                                          #
    # ------------------------------------------------------------------ #

    def _save_checkpoint(self, task_id: int):
        if not self.args.output_dir:
            return
        ckpt_dir = os.path.join(self.args.output_dir, f"task_{task_id}")
        os.makedirs(ckpt_dir, exist_ok=True)

        torch.save(self.model.state_dict(),
                   os.path.join(ckpt_dir, "backbone.pth"))
        torch.save(self.head_manager.state_dict(),
                   os.path.join(ckpt_dir, "heads.pth"))
        print(f"  [ckpt] Saved to {ckpt_dir}")


# ---------------------------------------------------------------------------
# iBOT / timm checkpoint loader
# ---------------------------------------------------------------------------

import re as _re

def load_vit_checkpoint(model: nn.Module, checkpoint_path: str) -> nn.Module:
    """
    Load a ViT-B/16 checkpoint saved in timm format (e.g. checkpoint_student.pth)
    into a HuggingFace ``ViTModel``.

    The checkpoint must be a standard ``.pth`` file containing either:
    * ``{"state_dict": {...}}``  (standard iBOT / timm save)
    * a bare state dict

    Key conversion (timm → HuggingFace ViTModel):
      cls_token                        → embeddings.cls_token
      pos_embed                        → embeddings.position_embeddings
      patch_embed.proj.*               → embeddings.patch_embeddings.projection.*
      blocks.{i}.norm1.*               → encoder.layer.{i}.layernorm_before.*
      blocks.{i}.attn.qkv.weight       → …query/key/value.weight  (split)
      blocks.{i}.attn.qkv.bias         → …query/key/value.bias    (split)
      blocks.{i}.attn.proj.*           → encoder.layer.{i}.attention.output.dense.*
      blocks.{i}.norm2.*               → encoder.layer.{i}.layernorm_after.*
      blocks.{i}.mlp.fc1.*             → encoder.layer.{i}.intermediate.dense.*
      blocks.{i}.mlp.fc2.*             → encoder.layer.{i}.output.dense.*
      norm.*                           → layernorm.*
      masked_embed                     → SKIPPED
    """
    print(f"[load_vit_checkpoint] Loading from {checkpoint_path}")
    raw = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    sd  = raw["state_dict"] if isinstance(raw, dict) and "state_dict" in raw else raw
    print(f"[load_vit_checkpoint] {len(sd)} tensors in checkpoint")

    hf: dict = {}
    skipped: list = []

    for k, v in sd.items():
        # ── skip iBOT-specific tokens ────────────────────────────────────
        if k in ("masked_embed",):
            skipped.append(k)
            continue

        # ── embedding layer ──────────────────────────────────────────────
        if k == "cls_token":
            hf["embeddings.cls_token"] = v; continue
        if k == "pos_embed":
            hf["embeddings.position_embeddings"] = v; continue
        if k == "patch_embed.proj.weight":
            hf["embeddings.patch_embeddings.projection.weight"] = v; continue
        if k == "patch_embed.proj.bias":
            hf["embeddings.patch_embeddings.projection.bias"] = v; continue

        # ── final layer norm ─────────────────────────────────────────────
        if k == "norm.weight":
            hf["layernorm.weight"] = v; continue
        if k == "norm.bias":
            hf["layernorm.bias"] = v; continue

        # ── transformer blocks ───────────────────────────────────────────
        m = _re.match(r"blocks\.(\d+)\.(.*)", k)
        if not m:
            skipped.append(k); continue

        i, sfx = m.group(1), m.group(2)
        pre = f"encoder.layer.{i}"

        _BLOCK_MAP = {
            "norm1.weight":     f"{pre}.layernorm_before.weight",
            "norm1.bias":       f"{pre}.layernorm_before.bias",
            "attn.proj.weight": f"{pre}.attention.output.dense.weight",
            "attn.proj.bias":   f"{pre}.attention.output.dense.bias",
            "norm2.weight":     f"{pre}.layernorm_after.weight",
            "norm2.bias":       f"{pre}.layernorm_after.bias",
            "mlp.fc1.weight":   f"{pre}.intermediate.dense.weight",
            "mlp.fc1.bias":     f"{pre}.intermediate.dense.bias",
            "mlp.fc2.weight":   f"{pre}.output.dense.weight",
            "mlp.fc2.bias":     f"{pre}.output.dense.bias",
        }

        if sfx in _BLOCK_MAP:
            hf[_BLOCK_MAP[sfx]] = v
        elif sfx == "attn.qkv.weight":        # fused (3H, H) → Q, K, V
            q, k_, val = v.chunk(3, dim=0)
            hf[f"{pre}.attention.attention.query.weight"] = q.contiguous()
            hf[f"{pre}.attention.attention.key.weight"]   = k_.contiguous()
            hf[f"{pre}.attention.attention.value.weight"] = val.contiguous()
        elif sfx == "attn.qkv.bias":           # fused (3H,) → Q, K, V
            q, k_, val = v.chunk(3, dim=0)
            hf[f"{pre}.attention.attention.query.bias"] = q.contiguous()
            hf[f"{pre}.attention.attention.key.bias"]   = k_.contiguous()
            hf[f"{pre}.attention.attention.value.bias"] = val.contiguous()
        else:
            skipped.append(k)

    if skipped:
        print(f"[load_vit_checkpoint] Skipped {len(skipped)} keys: {skipped}")

    missing, unexpected = model.load_state_dict(hf, strict=False)
    if missing:
        print(f"[load_vit_checkpoint] Missing  ({len(missing)}): {missing}")
    if unexpected:
        print(f"[load_vit_checkpoint] Unexpected ({len(unexpected)}): {unexpected}")
    print(f"[load_vit_checkpoint] Loaded {len(hf)} keys into ViTModel.")
    return model


# ---------------------------------------------------------------------------
# Serialisation helper
# ---------------------------------------------------------------------------

def _to_serializable(obj):
    if isinstance(obj, dict):
        return {k: _to_serializable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_serializable(v) for v in obj]
    if isinstance(obj, (torch.Tensor, np.ndarray)):
        return float(obj)
    return obj


# ---------------------------------------------------------------------------
# Deferred import (numpy used in evaluate_all)
# ---------------------------------------------------------------------------
import numpy as np
