"""
eval_vit.py – Evaluate a saved TreeLoRA ViT checkpoint.

Loads backbone weights and task heads, then computes per-task top-1 accuracy,
Overall Performance (OP), and Backward Transfer (BWT) from a saved results
directory produced by training/main_vit.py.

Alternatively, recomputes metrics live if ``--live_eval`` is set.

Usage
-----
    # Report from saved results.json (no GPU needed):
    python evaluations/eval_vit.py \
        --results_dir ./outputs_LLM-CL/cl/ViT/Tree_LoRA_cifar100

    # Live re-evaluation from checkpoints:
    python evaluations/eval_vit.py \
        --results_dir ./outputs_LLM-CL/cl/ViT/Tree_LoRA_cifar100 \
        --live_eval \
        --benchmark split_cifar100 \
        --data_path ./data/cifar100 \
        --vit_model google/vit-base-patch16-224-in21k \
        --n_tasks 10

Metrics
-------
OP  = (1/T) Σ  acc_{T,i}         – accuracy on task i after training all T tasks
BWT = (1/(T-1)) Σ acc_{T,i} − acc_{i,i}  – forgetting (negative = catastrophic)
"""

import argparse
import json
import os
import sys

sys.path.append(
    os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
)


# ---------------------------------------------------------------------------
# Result parsing (from saved JSON)
# ---------------------------------------------------------------------------

def parse_results_json(results_dir: str):
    """Load and pretty-print results.json written by main_vit.py."""
    rpath = os.path.join(results_dir, "results.json")
    if not os.path.exists(rpath):
        print(f"[eval_vit] No results.json found at {results_dir}")
        return None

    with open(rpath) as fh:
        data = json.load(fh)

    # Find the last key (final state)
    task_keys = sorted(k for k in data if k.startswith("after_task_"))
    if not task_keys:
        print("[eval_vit] results.json has no 'after_task_*' entries.")
        return None

    last_key  = task_keys[-1]
    n_tasks   = len(task_keys)
    final     = data[last_key]

    print(f"\n{'='*65}")
    print(f" TreeLoRA-ViT  Evaluation  ({os.path.basename(results_dir)})")
    print(f"{'='*65}")
    print(f"{'Task':<20}  {'Acc (%)':>10}")
    print(f"{'-'*65}")

    task_accs = []
    for k, v in final.items():
        if k in ("OP", "BWT"):
            continue
        acc_pct = v * 100 if v <= 1.0 else v   # handle both ratios and %
        task_accs.append(acc_pct)
        print(f"  {k:<20}  {acc_pct:>10.2f}")

    print(f"{'-'*65}")
    op  = final.get("OP",  sum(task_accs)/len(task_accs) if task_accs else 0)
    bwt = final.get("BWT", 0.0)

    op_pct  = op  * 100 if op  <= 1.0 else op
    bwt_pct = bwt * 100 if abs(bwt) <= 1.0 else bwt

    print(f"\n  Overall Performance (OP) : {op_pct:.2f}%")
    print(f"  Backward Transfer  (BWT): {bwt_pct:.2f}%")
    print(f"{'='*65}\n")

    # Forgetting per task
    print("  Per-task forgetting (final acc − best acc after own training):")
    for i, tk in enumerate(task_keys[:-1]):
        ti_acc  = data[task_keys[i]].get(f"task_{i}", 0.0) * 100
        fin_acc = task_accs[i] if i < len(task_accs) else 0.0
        fg      = fin_acc - ti_acc
        marker  = " ✓" if fg >= -1.0 else " ✗"
        print(f"    task_{i}: training={ti_acc:.1f}%  final={fin_acc:.1f}%  "
              f"Δ={fg:+.1f}%{marker}")

    print()
    return data


# ---------------------------------------------------------------------------
# Live evaluation
# ---------------------------------------------------------------------------

def live_eval(args):
    """Re-evaluate from saved backbone / heads checkpoints."""
    import torch
    from transformers import ViTModel
    from model.lora_vit import inject_lora_vit_hf, freeze_non_lora
    from model.Regular.Tree_LoRA_ViT import Tree_LoRA_ViT, TaskHeadManager

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ── Load backbone ──────────────────────────────────────────────────
    last_task_id = args.n_tasks - 1
    ckpt_dir     = os.path.join(args.results_dir, f"task_{last_task_id}")

    backbone_path = os.path.join(ckpt_dir, "backbone.pth")
    heads_path    = os.path.join(ckpt_dir, "heads.pth")

    for p in (backbone_path, heads_path):
        if not os.path.exists(p):
            print(f"[eval_vit] Checkpoint not found: {p}")
            return

    print(f"\n[eval_vit] Loading backbone from {backbone_path}")
    model = ViTModel.from_pretrained(args.vit_model)
    model = inject_lora_vit_hf(model, r=args.lora_r, lora_alpha=args.lora_alpha)
    model.load_state_dict(torch.load(backbone_path, map_location=device))
    model.to(device).eval()

    # Infer hidden_dim
    hidden_dim = model.config.hidden_size

    # ── Load heads ──────────────────────────────────────────────────────
    head_manager = TaskHeadManager(hidden_dim).to(device)
    head_manager.load_state_dict(torch.load(heads_path, map_location=device))
    head_manager.eval()

    # ── Build test datasets ─────────────────────────────────────────────
    from utils.data.vit_data_utils import build_split_cifar100, build_split_cub200
    common = dict(data_root=args.data_path, n_tasks=args.n_tasks,
                  batch_size=64, seed=args.seed, augment=False)
    if args.benchmark == "split_cifar100":
        _, _, test_tasks, _ = build_split_cifar100(**common)
    else:
        _, _, test_tasks, _ = build_split_cub200(**common)

    # ── Evaluate ────────────────────────────────────────────────────────
    print(f"\n[eval_vit]  Live evaluation on {args.benchmark}")
    accs = []
    for tid, (tname, dl) in enumerate(test_tasks.items()):
        correct = total = 0
        with torch.no_grad():
            for images, labels in dl:
                images = images.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                feat   = model(images).last_hidden_state[:, 0, :]
                logits = head_manager(feat, tid)
                preds  = logits.argmax(dim=1)
                correct += (preds == labels).sum().item()
                total   += labels.size(0)
        acc = 100.0 * correct / max(total, 1)
        accs.append(acc)
        print(f"  {tname}: {acc:.2f}%")

    op = sum(accs) / len(accs)
    print(f"\n  OP  (Overall Performance): {op:.2f}%")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--live_eval",   action="store_true")

    # Required only if --live_eval
    parser.add_argument("--benchmark",  type=str, default="split_cifar100")
    parser.add_argument("--data_path",  type=str, default="./data/cifar100")
    parser.add_argument("--vit_model",  type=str,
                        default="google/vit-base-patch16-224-in21k")
    parser.add_argument("--n_tasks",    type=int, default=10)
    parser.add_argument("--lora_r",     type=int, default=8)
    parser.add_argument("--lora_alpha", type=int, default=32)
    parser.add_argument("--seed",       type=int, default=1234)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    parse_results_json(args.results_dir)

    if args.live_eval:
        live_eval(args)
