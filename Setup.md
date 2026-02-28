## Computer Vision Setup

LoRA adapters approximate per-task gradients for sparse parameter updates. Tree depth is set to **5** for ViTs.

### Datasets

| Dataset | Tasks | Classes/Task | Train | Test |
|---|---|---|---|---|
| Split CIFAR-100 | 10 | 10 | 5,000/task | 1,000/task |
| Split ImageNet-R | 10 | 20 | ~2,400/task | ~600/task |
| Split CUB-200 | 10 | 20 | ~599/task | ~579/task |

### Hyperparameters

- **Backbone**: ViT-B/16 iBOT-21K (pretrained on ImageNet-21K)
- **Optimizer**: Adam (β1=0.9, β2=0.999)
- **Batch size**: 192 | **LR**: 0.005
- **Epochs**: 20 (CIFAR-100), 50 (others)
- **Input**: 224×224, normalized to [0, 1]

---

## NLP Setup

**Model**: LLaMA-3.2-1B-Instruct (1B params, 2048 hidden dim, 16 heads)

### Hyperparameters

- **Hardware**: 1× Nvidia RTX 4090 (24 GB), DeepSpeed ZeroStage 2, BF16
- **Batch size**: 1 | **Gradient accumulation steps**: 32 | **Effective global batch**: 32
- **Max prompt length**: 1024 tokens | **LR**: 1e-4
- **Tree depth**: 64 | **Regularization (λ)**: 0.5
- **Epochs per task** (8-task sequence): 2, 1, 3, 2, 1, 2, 2, 3

> **Note:** The original paper used 4× A800 GPUs with batch size 4 and 2 accumulation steps (effective batch = 32). The single-GPU setup replicates the same effective global batch size via 32 accumulation steps.

### Results (TRACE Benchmark, 8 tasks)

| Method | Overall Performance (OP) | Backward Transfer (BWT) |
|---|---|---|
| **TreeLoRA** | **36.14%** | **7.36%** |
| HiDeLoRA | 33.73% | 12.36% |
| O-LoRA | 32.94% | 12.89% |
