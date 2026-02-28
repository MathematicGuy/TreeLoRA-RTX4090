"""
main_vit.py – Entry point for TreeLoRA ViT continual learning.

Supports:
  --benchmark    split_cifar100 | split_cub200
  --n_tasks      number of tasks (default 10)
  --epochs       comma-separated epochs per task, e.g. 5,5,5,5,5,5,5,5,5,5
                 or a single integer to use uniformly

Example
-------
    python training/main_vit.py \
        --benchmark split_cifar100 \
        --data_path ./data/cifar100 \
        --n_tasks 10 \
        --epochs 5 \
        --reg 0.5 \
        --lora_depth 24 \
        --lora_r 8 \
        --lora_alpha 32 \
        --lr 1e-4 \
        --batch_size 64 \
        --seed 1234 \
        --output_dir ./outputs_LLM-CL/cl/ViT/Tree_LoRA_cifar100
"""

import argparse
import os
import sys

import torch

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)

from utils.utils import set_random_seed


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    def list_of_ints(s):
        return [int(x) for x in s.split(",")]

    parser = argparse.ArgumentParser(
        description="TreeLoRA ViT – continual learning on image benchmarks"
    )

    # ── Dataset ──────────────────────────────────────────────────────────
    parser.add_argument("--benchmark",   type=str, default="split_cifar100",
                        choices=["split_cifar100", "split_cub200"],
                        help="Which continual-learning benchmark to use.")
    parser.add_argument("--data_path",   type=str, required=True,
                        help="Root directory of the dataset.")
    parser.add_argument("--n_tasks",     type=int, default=10,
                        help="Number of tasks (must divide total class count).")
    parser.add_argument("--img_size",    type=int, default=224,
                        help="Image resize target for ViT.")

    # ── Model ─────────────────────────────────────────────────────────────
    parser.add_argument("--vit_model",   type=str,
                        default="google/vit-base-patch16-224-in21k",
                        help="HuggingFace model ID or local path for the ViT backbone.")
    parser.add_argument("--lora_r",      type=int, default=8,
                        help="LoRA rank.")
    parser.add_argument("--lora_alpha",  type=int, default=32,
                        help="LoRA alpha (scaling = alpha / r).")
    parser.add_argument("--lora_dropout", type=float, default=0.1,
                        help="LoRA dropout rate.")

    # ── TreeLoRA ─────────────────────────────────────────────────────────
    parser.add_argument("--lora_depth",  type=int, default=24,
                        help="KD-tree depth (= number of LoRA Q+V layers = 2 × n_blocks).")
    parser.add_argument("--reg",         type=float, default=0.5,
                        help="TreeLoRA regularisation coefficient λ.")
    parser.add_argument("--lamda_1",     type=float, default=0.5,
                        help="Weight applied to the reg_loss in the training objective.")
    parser.add_argument("--opl_weight",  type=float, default=0.1,
                        help="Weight for the Orthogonal Projection Loss component.")
    parser.add_argument("--use_opl",     action="store_true", default=True,
                        help="Enable OPL component (default: True).")

    # ── Training ─────────────────────────────────────────────────────────
    parser.add_argument("--epochs",      type=str, default="5",
                        help="Epochs per task: single int or comma-separated list.")
    parser.add_argument("--batch_size",  type=int, default=64,
                        help="Per-task training batch size.")
    parser.add_argument("--lr",          type=float, default=1e-4,
                        help="Learning rate for LoRA parameters and task heads.")
    parser.add_argument("--weight_decay", type=float, default=0.0,
                        help="AdamW weight decay.")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="DataLoader worker processes.")
    parser.add_argument("--val_split",   type=float, default=0.1,
                        help="Fraction of training data held out for validation.")

    # ── iBOT checkpoint ───────────────────────────────────────────────────
    parser.add_argument("--ibot_checkpoint", type=str, default=None,
                        help="Path to the iBOT flat-file checkpoint directory "
                             "(e.g. ./model/checkpoint_student). If provided, "
                             "iBOT weights are loaded instead of the HuggingFace "
                             "pre-trained weights.")

    # ── Memory / speed optimisations ─────────────────────────────────────
    parser.add_argument("--use_amp", action="store_true", default=True,
                        help="Enable automatic mixed precision (FP16 AMP) to "
                             "reduce VRAM usage by ~2x.")
    parser.add_argument("--grad_checkpoint", action="store_true", default=True,
                        help="Enable gradient checkpointing on the ViT encoder "
                             "to trade compute for ~40%% additional VRAM savings.")

    # ── Misc ─────────────────────────────────────────────────────────────
    parser.add_argument("--seed",        type=int, default=1234,
                        help="Global random seed.")
    parser.add_argument("--output_dir",  type=str, default=None,
                        help="Directory for checkpoints and results.")
    parser.add_argument("--device",      type=str, default="cuda",
                        help="'cuda' or 'cpu'.")
    parser.add_argument("--no_augment",  action="store_true",
                        help="Disable training-time augmentation.")

    args = parser.parse_args()

    # ── Parse epochs ─────────────────────────────────────────────────────
    if "," in args.epochs:
        args.epochs_list = [int(e) for e in args.epochs.split(",")]
        assert len(args.epochs_list) == args.n_tasks, (
            f"You provided {len(args.epochs_list)} epoch values but --n_tasks={args.n_tasks}"
        )
    else:
        args.epochs_list = [int(args.epochs)] * args.n_tasks

    return args


# ---------------------------------------------------------------------------
# Build ViT model with LoRA
# ---------------------------------------------------------------------------

def build_model(args):
    """
    Load a pre-trained HuggingFace ViT and inject LoRA adapters.

    If ``args.ibot_checkpoint`` is set, iBOT weights are loaded on top of the
    HuggingFace architecture *before* LoRA injection.

    The classification head is NOT loaded (we manage per-task heads separately
    inside ``Tree_LoRA_ViT``).

    Returns the bare ViT *feature extractor* with LoRA.
    """
    from transformers import ViTModel
    from model.lora_vit import inject_lora_vit_hf, freeze_non_lora

    print(f"\n[main_vit] Loading backbone architecture: {args.vit_model}")
    # Use ViTModel (no classification head) so we control the head ourselves
    model = ViTModel.from_pretrained(args.vit_model)

    # ── Optionally override with iBOT / timm weights ───────────────────
    if getattr(args, 'ibot_checkpoint', None):
        from model.Regular.Tree_LoRA_ViT import load_vit_checkpoint
        load_vit_checkpoint(model, args.ibot_checkpoint)
    else:
        print("[main_vit] Using HuggingFace pre-trained weights (no checkpoint provided)")

    # ── Optional: gradient checkpointing (saves ~40% VRAM) ──────────────
    if getattr(args, 'grad_checkpoint', False):
        model.gradient_checkpointing_enable()
        print("[main_vit] Gradient checkpointing enabled on ViT encoder")

    # Inject LoRA: targets query + value in each ViT encoder block
    model = inject_lora_vit_hf(
        model,
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
    )

    # Freeze everything except loranew_* parameters
    freeze_non_lora(model)

    # Count LoRA layers to validate lora_depth
    n_lora_A = sum(1 for n, _ in model.named_parameters() if "loranew_A" in n)
    print(f"[main_vit] Backbone has {n_lora_A} loranew_A matrices "
          f"(lora_depth should be <= {n_lora_A})")

    if args.lora_depth > n_lora_A:
        print(f"[WARNING] --lora_depth {args.lora_depth} > {n_lora_A} "
              f"loranew_A layers. Clamping to {n_lora_A}.")
        args.lora_depth = n_lora_A

    return model


# ---------------------------------------------------------------------------
# Build datasets
# ---------------------------------------------------------------------------

def build_datasets(args):
    from utils.data.vit_data_utils import build_split_cifar100, build_split_cub200

    common_kwargs = dict(
        data_root=args.data_path,
        n_tasks=args.n_tasks,
        batch_size=args.batch_size,
        img_size=args.img_size,
        num_workers=args.num_workers,
        augment=not args.no_augment,
        seed=args.seed,
        val_split=args.val_split,
    )

    if args.benchmark == "split_cifar100":
        return build_split_cifar100(**common_kwargs)
    elif args.benchmark == "split_cub200":
        return build_split_cub200(**common_kwargs)
    else:
        raise ValueError(f"Unknown benchmark: {args.benchmark}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()

    # ── Reproducibility ──────────────────────────────────────────────────
    set_random_seed(args.seed)
    if torch.cuda.is_available() and args.device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32       = True

    # ── Datasets ─────────────────────────────────────────────────────────
    print(f"\n[main_vit] Building dataset: {args.benchmark}")
    train_tasks, val_tasks, test_tasks, class_masks = build_datasets(args)

    print(f"[main_vit] Tasks: {list(train_tasks.keys())}")
    print(f"[main_vit] Classes per task: {[len(m) for m in class_masks]}")

    # ── Model ─────────────────────────────────────────────────────────────
    model = build_model(args)

    # ── Trainer ──────────────────────────────────────────────────────────
    from model.Regular.Tree_LoRA_ViT import Tree_LoRA_ViT

    trainer = Tree_LoRA_ViT(
        model=model,
        optimizer_cls=torch.optim.AdamW,
        optimizer_kwargs=dict(lr=args.lr, weight_decay=args.weight_decay),
        train_task_list=train_tasks,
        val_task_list=val_tasks,
        test_task_list=test_tasks,
        class_masks=class_masks,
        args=args,
        lamda_1=args.lamda_1,
    )

    # ── Train ─────────────────────────────────────────────────────────────
    print(f"\n[main_vit] Starting continual training over {args.n_tasks} tasks")
    print(f"  epochs per task: {args.epochs_list}")

    all_results = trainer.train_continual(args.epochs_list)

    # ── Final summary ─────────────────────────────────────────────────────
    last_key = f"after_task_{args.n_tasks - 1}"
    if last_key in all_results:
        final = all_results[last_key]
        print(f"\n{'='*60}")
        print(f"[main_vit] FINAL RESULTS ({args.benchmark})")
        print(f"{'='*60}")
        for task_name, acc in final.items():
            if task_name not in ("OP", "BWT"):
                print(f"  {task_name}: {acc*100:.2f}%")
        print(f"  OP  (Overall Performance): {final.get('OP', 0)*100:.2f}%")
        print(f"  BWT (Backward Transfer)  : {final.get('BWT', 0)*100:.2f}%")
        print(f"{'='*60}")


if __name__ == "__main__":
    main()
