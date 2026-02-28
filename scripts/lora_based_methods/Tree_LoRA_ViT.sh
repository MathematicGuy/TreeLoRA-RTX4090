#!/usr/bin/env bash
# Tree_LoRA_ViT.sh
# Run TreeLoRA on Split CIFAR-100 and Split CUB-200 with a ViT-B/16 backbone.
#
# Hardware: single RTX 4090 (24 GB VRAM)
# Backbone: iBOT ViT-B/16 pre-trained on ImageNet-21K (self-supervised)
#           Paper: https://github.com/bytedance/ibot
#           Fallback: google/vit-base-patch16-224-in21k (supervised)
#           + LoRA r=8 on Q and V of all 12 blocks → 24 loranew_A layers
#
# Usage:
#   bash scripts/lora_based_methods/Tree_LoRA_ViT.sh
# or override benchmark:
#   BENCHMARK=split_cub200 bash scripts/lora_based_methods/Tree_LoRA_ViT.sh

set -euo pipefail

# ── Time stamp ──────────────────────────────────────────────────────────────
now=$(date +"%m%d_%H%M%S")

# ── GPU ─────────────────────────────────────────────────────────────────────
export CUDA_VISIBLE_DEVICES=0
# Reduce memory fragmentation (fixes "reserved but unallocated" OOM)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ── Benchmark selection ──────────────────────────────────────────────────────
BENCHMARK=${BENCHMARK:-split_cifar100}   # override with env var if needed

if [ "$BENCHMARK" = "split_cifar100" ]; then
    DATA_PATH="./data"
    N_TASKS=10
    # Paper: 20 epochs per task on Split CIFAR-100
    EPOCHS="20,20,20,20,20,20,20,20,20,20"
    # Paper batch size = 192; use 64 as VRAM-safe default with AMP + grad_checkpoint
    BATCH_SIZE=64
elif [ "$BENCHMARK" = "split_cub200" ]; then
    DATA_PATH="./data/CUB_200_2011"
    N_TASKS=10
    # Paper: 50 epochs per task on other benchmarks (CUB-200)
    EPOCHS="50,50,50,50,50,50,50,50,50,50"
    BATCH_SIZE=64
else
    echo "Unknown BENCHMARK=$BENCHMARK. Use split_cifar100 or split_cub200."
    exit 1
fi

# ── Model ────────────────────────────────────────────────────────────────────
# Paper uses iBOT ViT-B/16 pretrained on ImageNet-21K (self-supervised).
# Download the checkpoint from https://github.com/bytedance/ibot and convert,
# or point to a locally cached copy:
# VIT_MODEL="./PTM/ibot-vit-base-patch16-224-in21k"
# Falling back to the supervised HuggingFace checkpoint if iBOT unavailable:
VIT_MODEL="google/vit-base-patch16-224-in21k"

# Path to the iBOT flat-file checkpoint directory.
# Set to "" to skip and use the HuggingFace pre-trained weights.
IBOT_CHECKPOINT="./model/checkpoint_student"

# ── LoRA hyper-params ────────────────────────────────────────────────────────
LORA_R=8
LORA_ALPHA=32
LORA_DROPOUT=0.1
# Paper tree depth = 5 for ViTs
LORA_DEPTH=5

# ── TreeLoRA hyper-params ────────────────────────────────────────────────────
# Paper: λ = 0.5, same as LLM experiments
REG=0.5
LAMDA_1=0.5
OPL_WEIGHT=0.1

# ── Optimiser ────────────────────────────────────────────────────────────────
# Paper: constant LR = 0.005, Adam β1=0.9 β2=0.999
LR=5e-3
WEIGHT_DECAY=0.0

# ── Reproducibility ──────────────────────────────────────────────────────────
SEED=1234

# ── Output ───────────────────────────────────────────────────────────────────
OUTPUT_DIR="./outputs_LLM-CL/cl/ViT/Tree_LoRA_${BENCHMARK}_${now}"

# ── Run ──────────────────────────────────────────────────────────────────────
echo "=================================================="
echo " TreeLoRA ViT  |  benchmark : $BENCHMARK"
echo " backbone      : $VIT_MODEL  (iBOT: $IBOT_CHECKPOINT)"
echo " lora_depth    : $LORA_DEPTH"
echo " reg / lamda_1 : $REG / $LAMDA_1"
echo " lr / bs       : $LR / $BATCH_SIZE"
echo " epochs        : $EPOCHS"
echo " output        : $OUTPUT_DIR"
echo "=================================================="

# Build optional iBOT flag
IBOT_FLAG=""
if [ -n "$IBOT_CHECKPOINT" ]; then
    IBOT_FLAG="--ibot_checkpoint $IBOT_CHECKPOINT"
fi

python training/main_vit.py \
    --benchmark    "$BENCHMARK"   \
    --data_path    "$DATA_PATH"   \
    --n_tasks      "$N_TASKS"     \
    --vit_model    "$VIT_MODEL"   \
    $IBOT_FLAG                    \
    --lora_r       "$LORA_R"      \
    --lora_alpha   "$LORA_ALPHA"  \
    --lora_dropout "$LORA_DROPOUT"\
    --lora_depth   "$LORA_DEPTH"  \
    --reg          "$REG"         \
    --lamda_1      "$LAMDA_1"     \
    --opl_weight   "$OPL_WEIGHT"  \
    --use_opl                     \
    --use_amp                     \
    --grad_checkpoint             \
    --epochs       "$EPOCHS"      \
    --batch_size   "$BATCH_SIZE"  \
    --lr           "$LR"          \
    --weight_decay "$WEIGHT_DECAY"\
    --seed         "$SEED"        \
    --output_dir   "$OUTPUT_DIR"  \
    --num_workers  4

echo ""
echo "=================================================="
echo " Evaluation"
echo "=================================================="

python evaluations/eval_vit.py \
    --results_dir "$OUTPUT_DIR"
