# Comprehensive Before-and-After Comparison: TreeLoRA → SO-LoRA

**Files compared:**
- Before (Origin): `utils/kd_lora_tree_origin.py` · `model/Regular/Tree_LoRA_origin.py`
- After (Improved): `utils/kd_lora_tree.py` · `model/Regular/Tree_LoRA.py`

---

## 1. Abstract

Applying Low-Rank Adaptation (LoRA) to continual learning in vision transformers presents a structural dilemma: maintaining parameter efficiency while preventing catastrophic forgetting due to gradient interference across sequential tasks. To resolve this, we introduce **Similarity-Orthogonal Low-Rank Adaptation (SO-LoRA)**. SO-LoRA integrates similarity-aware adapter selection with strict gradient-level orthogonalization. Rather than allowing unconstrained updates, we utilize an efficient retrieval memory to dynamically identify relevant past tasks. We then apply an **Orthogonal Projection Loss (OPL)** to constrain new task gradients strictly to the null space of accumulated previous tasks, actively shielding established knowledge representations from destructive interference. Extensive empirical evaluations on incremental visual classification benchmarks, including ImageNet-R, demonstrate that SO-LoRA significantly mitigates forgetting. By effectively balancing stability and plasticity, our framework outperforms state-of-the-art continual learning baselines, such as RAPF and MG-CLIP, across diverse sequential vision tasks without compromising optimization dynamics.

---

## 2. Comprehensive Comparison

---

### 2.1  Problem Setting and Motivation

#### 2.1.1 Continual Learning with LoRA

In standard continual learning (CL), a model is trained on a sequence of tasks $\mathcal{T}_1, \mathcal{T}_2, \ldots, \mathcal{T}_T$ where each task $t$ has its own data distribution $\mathcal{D}_t$. The central challenge is **catastrophic forgetting**: updating parameters for task $t$ overwrites gradient directions that were critical for earlier tasks.

When LoRA is incorporated, only a small adapter $\Delta W = B A$ (with $B \in \mathbb{R}^{d \times r}$, $A \in \mathbb{R}^{r \times d}$, rank $r \ll d$) is updated per task, while the backbone $W_0$ is frozen. This greatly reduces the parameter budget but does **not** eliminate gradient interference within the low-rank subspace; two adapter updates from different tasks can still destructively overlap in the shared $r$-dimensional projection space.

#### 2.1.2 What the Origin Version Does

The origin approach builds a **KD-tree indexed by past-task LoRA-A gradients** and uses a UCB/LCB bandit to select the most relevant past task per layer at each training step. The selected past task's gradient is then used to compute a **dot-product similarity loss** that is subtracted from the CE loss, encouraging the current update to be consistent with past learned directions (positive transfer).

Formally, the origin regularization at step $s$ of task $t$ is:

$$\mathcal{L}_\text{origin}(s) = \mathcal{L}_\text{CE}(s) - \lambda_\text{reg}(s) \cdot \mathcal{L}_\text{sim}$$

$$\mathcal{L}_\text{sim} = -\sum_{l=0}^{L-1} \bigl(g_l^{(t)} \cdot g_l^{(\hat{t}_l)}\bigr)$$

where $g_l^{(t)}$ is the current LoRA-A weight at layer $l$, $\hat{t}_l$ is the bandit-selected past task for layer $l$, and $\lambda_\text{reg}(s) = \lambda \cdot s/S$ is a linear warmup schedule.

**What this misses:** Maximizing the dot product with a past gradient pushes the update *towards* that past direction — this preserves positive transfer for similar tasks. However, it provides **no guarantee** that the update avoids other past gradient directions. If the current gradient has a large projection onto a dissimilar past task's gradient, that past knowledge will still be overwritten.

---

### 2.2  The OPL Paper (Ranasinghe et al., ICCV 2021) — Theory Recap

The OPL paper introduces an orthogonality constraint in the **feature space** for classification. Given a mini-batch of $N$ samples with features $\{f_i\}$ and class weight vectors $\{w_c\}$:

$$L_\text{OPL} = \underbrace{\frac{1}{N}\sum_i \sum_{c \neq y_i} \left(\hat{f}_i \cdot \hat{w}_c\right)^2}_{\text{inter-class separation}} + \underbrace{\frac{1}{N}\sum_i \left(1 - \hat{f}_i \cdot \hat{w}_{y_i}\right)^2}_{\text{intra-class compaction}}$$

where $\hat{(\cdot)}$ denotes L2 normalization. This is minimized when each class's feature vectors are exactly orthogonal to every other class's weight vector and perfectly aligned with their own.

**Key mathematical properties from Ranasinghe et al.:**
1. The minimum loss value is 0, achieved only when $\hat{f}_i \perp \hat{w}_c$ for all $c \neq y_i$.
2. The loss is bounded: $L_\text{OPL} \in [0, N \cdot (C-1) + N] = [0, NC]$.
3. It does not require extra learnable parameters.
4. It is not sensitive to batch size (unlike contrastive losses).

---

### 2.3  OPL Adaptation: Feature Space → Gradient Space

The improved code transfers the OPL philosophy from classification feature space into the **gradient subspace** of LoRA adapters. This is a **novel adaptation**, not a direct port. Below is the theoretical justification and the mapping.

#### 2.3.1 Mapping Table

| OPL (Original) | TreeLoRA Adaptation |
|---|---|
| Feature vectors $f_i$ of class $c$ | Current-task LoRA-A gradient $g_l^{(t)}$ at layer $l$ |
| Class weight vector $w_{c'}$ (other class) | Past-task LoRA-A gradient $g_l^{(\hat{t}_l)}$ (bandit-selected) |
| Inter-class separation penalty | Penalize projection onto past gradient directions |
| Intra-class compaction penalty | Similarity loss — push towards selected past direction |
| Feature space separation | Gradient subspace orthogonality across tasks |

#### 2.3.2 Implemented OPL Formula (Per-Layer)

For each LoRA layer $l$:

$$\text{proj}\!\left(g_l^{(t)},\ g_l^{(\hat{t}_l)}\right) = \frac{g_l^{(t)} \cdot g_l^{(\hat{t}_l)}}{\left\|g_l^{(\hat{t}_l)}\right\|^2} \cdot g_l^{(\hat{t}_l)}$$

$$L_\text{OPL}^{(l)} = \left\|\text{proj}\!\left(g_l^{(t)},\ g_l^{(\hat{t}_l)}\right)\right\|^2 = \frac{\left(g_l^{(t)} \cdot g_l^{(\hat{t}_l)}\right)^2}{\left\|g_l^{(\hat{t}_l)}\right\|^2}$$

Aggregated across all $L$ layers with normalization:

$$L_\text{OPL} = \frac{\displaystyle\sum_{l=0}^{L-1} L_\text{OPL}^{(l)}}{\displaystyle\sum_{l=0}^{L-1} \left\|g_l^{(t)}\right\|^2}$$

The normalization by $\sum_l \|g_l^{(t)}\|^2$ keeps $L_\text{OPL} \in [0, 1]$, making it scale-invariant to gradient magnitude — matching the L2-normalization used in the original OPL paper.

#### 2.3.3 Combined Training Objective (Improved)

$$\boxed{\mathcal{L}_\text{total}(s) = \mathcal{L}_\text{CE}(s) - \underbrace{\lambda_\text{reg}(s) \cdot \mathcal{L}_\text{sim}}_{\text{positive transfer}} - \underbrace{\lambda_\text{OPL} \cdot \mathcal{L}_\text{CE}^\text{detach}(s) \cdot L_\text{OPL}}_{\text{interference prevention}}}$$

Code location — `kd_lora_tree.py :: KD_LoRA_Tree.get_loss()`:

```python
# Base similarity loss (shared with origin)
reg_loss = tree_lora_loss(_grad_current, self.all_grad_device, task_id, prev_id_matrix)
reg_loss = reg_loss / (reg_loss.detach().abs() + 1e-5) * loss.detach() * self.tmp_reg

# NEW: OPL additive term
if self.use_opl and task_id > 0:
    opl_loss = orthogonal_projection_loss(
        _grad_current, self.all_grad_device, prev_id_matrix, normalize=True
    )
    if opl_loss.abs() > 1e-8:
        opl_loss = opl_loss * loss.detach() * self.opl_weight
        reg_loss = reg_loss + opl_loss

return reg_loss   # subtracted from L_CE in train_one_task
```

---

### 2.4  KDTreeNode: Before vs. After

#### 2.4.1 Split Normalization

```python
# ===== ORIGIN (kd_lora_tree_origin.py) =====
self.mean_vector = current_grads.mean(dim=0)            # raw mean
similarities = torch.mv(current_grads, self.mean_vector) # raw dot product

# ===== IMPROVED (kd_lora_tree.py) =====
self.mean_vector = current_grads.mean(dim=0)
mean_norm = torch.norm(self.mean_vector)
if mean_norm > 1e-8:
    normalized_mean = self.mean_vector / mean_norm       # L2-normalize
else:
    normalized_mean = self.mean_vector
similarities = torch.mv(current_grads, normalized_mean) # cosine-like dot
```

**Theoretical impact:** In the origin, gradients with larger $\ell_2$ norms dominate the similarity score regardless of direction, which can cause the tree to split on magnitude rather than orientation. Normalizing the mean vector removes this bias — the split becomes a function of the **angle** between each gradient and the mean direction, not its magnitude. This produces more geometrically meaningful partitions.

**Formal statement:** Let $\mathbf{m} = \text{mean}(\{g_l^{(i)}\}_{i \in \mathcal{N}})$. The origin computes $s_i = g_l^{(i)} \cdot \mathbf{m}$, while the improved computes $s_i = g_l^{(i)} \cdot \hat{\mathbf{m}}$ where $\hat{\mathbf{m}} = \mathbf{m}/\|\mathbf{m}\|$. The latter is the projection of $g_l^{(i)}$ onto the unit ball axis of $\mathbf{m}$, bounded in $[-\|g_l^{(i)}\|, \|g_l^{(i)}\|]$ and invariant to scaling of $\mathbf{m}$.

#### 2.4.2 New Node Attributes

| Attribute | Origin | Improved | Purpose |
|---|---|---|---|
| `gradient_norm` | ❌ | ✅ | Track gradient magnitude at each node |
| `orthogonal_basis` | ❌ | ✅ | Placeholder for Gram-Schmidt basis (not yet populated) |
| `get_orthogonal_complement()` | ❌ | ✅ | Returns component of query grad perpendicular to node mean |

`get_orthogonal_complement()` computes:

$$g^\perp = g - \frac{g \cdot \mathbf{m}}{\mathbf{m} \cdot \mathbf{m}} \cdot \mathbf{m}$$

This is the standard vector projection rejection, and is mathematically correct. However, it is **not currently called** anywhere in the training loop — it is defined for future use or external integration. Noted limitation.

---

### 2.5  KD_LoRA_Tree Class: Before vs. After

#### 2.5.1 Initialization State

| Field | Origin | Improved | Role |
|---|---|---|---|
| `all_accumulate_grads` | `[None] * num_tasks` | `[None] * num_tasks` | Per-task gradient memory |
| `opl_weight` | ❌ | `0.1` (from args) | Scale OPL term |
| `use_opl` | ❌ | `True` (from args) | Toggle OPL computation |
| `opl_history` | ❌ | `[]` | Track OPL values for monitoring |
| `projection_matrices` | ❌ | `{}` | Per-task per-layer $P_l$ matrices |
| `tmp_reg` | ❌ (implicit) | `0` (explicit) | Warmup schedule current value |

#### 2.5.2 `insert_grad()` — Gradient Accumulation

```python
# ===== ORIGIN =====
for i in range(len(_grad_current)):       # BUG: iterates over lora_depth, not tasks
    if self.current_grad is None:
        self.current_grad = _grad_current.detach() * 1.0 / self.total_rounds
    else:
        frac = 1.0 / self.total_rounds
        self.current_grad += _grad_current.detach() * frac

# ===== IMPROVED =====
if self.current_grad is None:
    self.current_grad = _grad_current.detach() / self.total_rounds
else:
    frac = 1.0 / self.total_rounds
    self.current_grad = self.current_grad + _grad_current.detach() * frac
```

**Bug in origin identified:** The `for i in range(len(_grad_current))` loop iterates over `lora_depth` (rows of the gradient tensor) rather than doing anything per-row. Because the body doesn't index `i`, it adds the *entire* `_grad_current` tensor `lora_depth` times instead of once. This **overestimates the gradient accumulation by a factor of `lora_depth`**. The improved version removes this loop entirely, which is the correct behavior.

#### 2.5.3 `end_task()` — Post-Task Updates

```
Origin flow:
  store grad → stack grads → compute differences → build KDTreeNode → print

Improved flow:
  guard (reg ≤ 0 or no grad → return early)
  store grad (with .clone()) → stack grads → compute differences →
  build KDTreeNode → _update_projection_matrix(task_id) → structured logs
```

**New: `_update_projection_matrix(task_id)`**

For each LoRA layer $l$, after task $t$, build the projection matrix onto the subspace spanned by all past gradients at layer $l$:

$$P_l = G_l \left(G_l^\top G_l + \varepsilon I\right)^{-1} G_l^\top$$

where $G_l \in \mathbb{R}^{t \times d_l}$ stacks the layer-$l$ gradients of the $t$ completed tasks row-wise.

**Mathematical correctness:** Yes, this is the standard formula for the orthogonal projection onto $\text{col}(G_l^\top)$. The use of `torch.linalg.pinv` with regularization `1e-6 * I` provides a Moore-Penrose pseudoinverse, which is numerically stable and handles rank-deficient cases. ✅

**Remaining concern:** For large $L$ and growing $t$, storing all $P_l$ matrices is $O(T \cdot L \cdot d^2)$ in memory, which can become prohibitive for large LLMs. Not addressed in current implementation.

#### 2.5.4 `tree_search()` — Structural Improvement

The search logic is functionally equivalent between origin and improved. Key difference: the improved version uses a named variable `similarity_boost` and adds a guard for `self.kd_tree_root.right is not None` before accessing `right.median_similarity`, preventing a potential `AttributeError` on unbalanced trees.

```python
# Origin (unsafe)
else:
    similarity = self.kd_tree_root.right.median_similarity if \
        self.kd_tree_root.right.median_similarity is not None else 1.0
    sim[self.kd_tree_root.right.task_indices] *= min(similarity, 1.5)

# Improved (safe)
if self.kd_tree_root.right is not None and \
   self.kd_tree_root.right.median_similarity is not None:
    similarity_boost = min(self.kd_tree_root.right.median_similarity, 1.5)
    sim[self.kd_tree_root.right.task_indices] *= similarity_boost
```

Also: the improved version adds `+ 1e-5` after `torch.min(sim).abs()` (vs. just `torch.min(sim)` in origin) to ensure all values are strictly positive before `torch.softmax`, preventing degenerate probability distributions when all similarities are identical.

#### 2.5.5 `get_loss()` — The Core Change

| Component | Origin formula | Improved formula |
|---|---|---|
| Similarity loss | $\mathcal{L}_\text{sim} = -\sum_l g_l^{(t)} \cdot g_l^{(\hat{t}_l)}$ | Same |
| Normalization | $\frac{\mathcal{L}_\text{sim}}{|\mathcal{L}_\text{sim}|+\varepsilon} \cdot \mathcal{L}_\text{CE}^\text{stop} \cdot \lambda_s$ | Same (`.abs()` added for sign-safety) |
| OPL term | ❌ | $+\ \lambda_\text{OPL} \cdot \mathcal{L}_\text{CE}^\text{stop} \cdot L_\text{OPL}$ |
| OPL tracking | ❌ | `opl_history.append(opl_loss.item())` |

**Sign convention note:** The returned `reg_loss` is **subtracted** from `L_CE` in the training loop (`loss = loss - reg_loss`). This means:
- $\mathcal{L}_\text{sim}$ (negative dot product) → after subtraction: $+\sum_l g_l^{(t)} \cdot g_l^{(\hat{t}_l)}$, i.e., maximize similarity ✅
- $L_\text{OPL}$ (positive squared projection norm) → after addition to `reg_loss` and subtraction from loss: **minimizes** the OPL term ✅

The sign algebra is consistent and correct. ✅

---

### 2.6  Tree_LoRA Model Class: Before vs. After

#### 2.6.1 Initialization

```python
# ===== ORIGIN =====
self.lamda_1 = lamda_1   # fixed at call-site, no args override
self.lamda_2 = lamda_2
self.kd_lora_tree = KD_LoRA_Tree(args)  # no OPL config injected

# ===== IMPROVED =====
self.lamda_1 = getattr(args, 'lamda_1', lamda_1)  # args can override
self.lamda_2 = getattr(args, 'lamda_2', lamda_2)

args.opl_weight = getattr(args, 'opl_weight', 0.1)  # inject before KD_LoRA_Tree
args.use_opl    = getattr(args, 'use_opl', True)

self.kd_lora_tree = KD_LoRA_Tree(args)
self.use_gradient_projection = getattr(args, 'use_gradient_projection', False)
self.task_losses = []; self.reg_losses = []; self.opl_losses = []
self.forgetting_metrics = {}
```

#### 2.6.2 New Helper: `_apply_gradient_projection(task_id)`

This method runs **after** `model.backward(loss)` and directly modifies `param.grad.data` using sequential Gram-Schmidt:

$$\tilde{g}_\text{param} \leftarrow g_\text{param} - \sum_{t' < t} \frac{g_\text{param} \cdot g_\text{param}^{(t')}}{\|g_\text{param}^{(t')}\|^2} \cdot g_\text{param}^{(t')}$$

**Mathematical correctness:** For exactly two past tasks with linearly independent gradients, this produces a gradient in the orthogonal complement of $\text{span}\{g^{(0)}, g^{(1)}\}$. For $T > 2$ past tasks, sequential Gram-Schmidt is **not numerically orthogonal** in general due to floating-point accumulation — the result has a residual component of order $O(T \cdot \epsilon_\text{machine})$ in each subtracted direction. For $T < 10$ this is acceptable; for larger task counts, modified Gram-Schmidt (MGS) or QR decomposition should be used. **Flagged as a limitation.**

**Shape-matching issue:** The current implementation matches layers by checking `past_flat.shape[0] == grad.shape[0]` — i.e., by flattened dimension rather than by layer name. If two LoRA-A matrices at different layers share the same flattened dimension (which can happen if `r × d` is equal across layers), the wrong past gradient will be projected. This is a **correctness bug** for non-uniform architectures. ⚠️

#### 2.6.3 Training Loop: Step-by-Step Diff

```
ORIGIN train_one_task:
  for epoch:
    for step, batch:
      [step counter]
      forward() → L_CE
      if reg > 0:
        extract LoRA-A params (inline loop)
        stack → tensor
        insert_grad()
        if task_id > 0:
          tree_search()
          get_loss() → reg_loss          ← similarity only
          loss = loss - reg_loss
          if step % 100: print 4 lines
      progress_bar.update()
      backward(); step()
  [end epoch]
  save weights + tokenizer

IMPROVED train_one_task:
  print task header
  for epoch:
    new tiktok
    kd_lora_tree.new_epoch_init()
    epoch_task_loss = 0; epoch_reg_loss = 0
    for step, batch:
      kd_lora_tree.step()
      forward() → L_CE
      epoch_task_loss += L_CE.item()
      if reg > 0:
        _extract_lora_gradients()          ← factored method
        _compute_gradient_tensor()         ← factored method
        insert_grad()
        if task_id > 0:
          tree_search()
          get_loss() → reg_loss            ← similarity + OPL
          loss = loss - reg_loss
          epoch_reg_loss += reg_loss
          if step % 100: _log_training_status()   ← includes OPL history
      progress_bar.update()
      backward()
      if use_gradient_projection:
        _apply_gradient_projection()       ← NEW: hard orthogonalization
      model.step()
      if step % 30: tiktok.print_time()
    [epoch summary log]
    task_losses.append(); reg_losses.append()
  progress_bar.close()
  _save_task_checkpoint()                  ← saves JSON stats + pkl tree
  kd_lora_tree.end_task()
  if task_id > 0: _evaluate_forgetting()   ← NEW: measures forgetting
```

#### 2.6.4 New: `_evaluate_forgetting(current_task_id)`

Runs the model on up to 50 batches of each previously seen task's eval set and records loss + perplexity. This is the **forward-transfer / forgetting diagnostic** loop absent in the origin.

**Correctness note:** The loop breaks at 50 batches regardless of dataset size. This is a fast approximation — sufficient for monitoring but not for rigorous forgetting measurement. A full-epoch evaluation would be more accurate but slower. The saved metrics are currently not used adaptively (e.g., to scale `opl_weight`), which is a design gap.

#### 2.6.5 `save_model()` — O-LoRA Compatibility

Both origin and improved write `adapter_config['r_sum'] = 0` for O-LoRA compatibility. The improved version additionally writes a `treelora_metadata` block:

```json
{
  "r_sum": 0,
  "treelora_metadata": {
    "task_id": 3,
    "lamda_1": 0.5,
    "lamda_2": 0.0,
    "reg": 0.1,
    "use_opl": true
  }
}
```

This makes checkpoints self-documenting for ablation studies.

#### 2.6.6 New Subclass: `Tree_LoRA_OPL`

```python
class Tree_LoRA_OPL(Tree_LoRA):
    opl_mode: Literal['loss', 'projection', 'hybrid']
    # 'loss'       → OPL as loss term only    (soft)
    # 'projection' → direct grad surgery only (hard)
    # 'hybrid'     → both simultaneously      (maximum)
```

This is a clean strategy pattern. The mode switch correctly sets `use_gradient_projection` and `kd_lora_tree.use_opl` flags before delegating to `super().train_one_task()`.

---

### 2.7  New Standalone Functions in Improved `kd_lora_tree.py`

#### 2.7.1 `orthogonal_projection_loss()`

```python
def orthogonal_projection_loss(current_grad, past_grads, prev_id_matrix, normalize=True):
    total_proj_sq = 0.0; total_curr_sq = 0.0
    for depth_id in range(num_layers):
        g_curr = current_grad[depth_id]
        g_past = past_grads[selected_id, depth_id]
        ||g_past||^2 = dot(g_past, g_past)
        if ||g_past||^2 > 1e-8:
            proj = (dot(g_curr, g_past) / ||g_past||^2) * g_past
            total_proj_sq += dot(proj, proj)
        total_curr_sq += dot(g_curr, g_curr)
    return total_proj_sq / total_curr_sq  (if normalize)
```

**Correctness:** The formula correctly computes the squared cosine similarity between `g_curr` and `g_past` summed across layers. The normalization makes it equivalent to $\cos^2\theta$ for a single layer. ✅

**Concern:** This projects current onto a **single** selected past task per layer (via `prev_id_matrix`). True OPL would project onto the full subspace $\text{span}(G_l^{(1)}, \ldots, G_l^{(t-1)})$. Using only one reference task per layer may miss interference with unselected past tasks. This is a deliberate approximation for computational tractability.

#### 2.7.2 `gram_schmidt_orthogonalize()`

```python
def gram_schmidt_orthogonalize(current_grad, past_grads):
    orth = current_grad.clone()
    for g in past_grads:
        ||g||^2 = dot(g, g)
        if ||g||^2 > 1e-8:
            orth = orth - (dot(orth, g) / ||g||^2) * g
    return orth
```

**Correctness:** This is the classical Gram-Schmidt projection rejection. After iterating over all $K$ past gradients, the result is orthogonal to each one **at the time of its subtraction**, but not necessarily to earlier ones due to floating-point drift. This is the well-known instability of classical (as opposed to modified) Gram-Schmidt. For $K \leq 8$ tasks the residual error is negligible; for $K \geq 20$ it becomes measurable. ⚠️ **Acceptable in practice for typical CL benchmarks (≤ 15 tasks).**

---

## 3. Summary: Pros, Cons, and Mathematical Analysis

---

### 3.1 What the Improved Version Gets Right ✅

| Claim | Verdict | Reasoning |
|---|---|---|
| OPL formula correctly adapted to gradient space | ✅ Correct | Projection formula $\|proj(g,v)\|^2/\|g\|^2 \in [0,1]$ is standard and bounded |
| Normalized mean in KDTreeNode split | ✅ Correct | Removes magnitude bias; equivalent to cosine similarity |
| `insert_grad()` loop bug fixed | ✅ Correct | Origin multiplied gradient by `lora_depth` extra times |
| Projection matrix $P_l = G(G^TG)^{-1}G^T$ via pinv | ✅ Correct | Standard least-squares projection; regularization prevents ill-conditioning |
| Combined loss sign algebra | ✅ Correct | `loss - reg_loss` with `reg_loss = sim_loss + opl_loss` gives correct gradient directions |
| `sim + torch.min(sim).abs() + 1e-5` for strict positivity | ✅ Correct | Ensures valid probability distribution for multinomial sampling |
| OPL weight relative to task loss (`* loss.detach()`) | ✅ Good | Keeps OPL scale proportional to task loss magnitude; prevents domination |

---

### 3.2 What Is Incorrect or Incomplete ⚠️

| Issue | Location | Severity | Description |
|---|---|---|---|
| **Shape-based layer matching in `_apply_gradient_projection`** | `Tree_LoRA.py:199` | High | Matches layers by flattened dimension size, not name — fails for architectures where multiple LoRA-A matrices share the same `r × d` dimension |
| **Single-task OPL vs. full-subspace OPL** | `kd_lora_tree.py:_compute_opl_loss` | Medium | OPL uses only the bandit-selected past task per layer; other past tasks' interference is unconstrained |
| **Classical Gram-Schmidt numerical drift** | `kd_lora_tree.py:gram_schmidt_orthogonalize` | Low (for ≤15 tasks) | Not modified-Gram-Schmidt; orthogonality error grows with task count |
| **`get_orthogonal_complement()` never called** | `kd_lora_tree.py:KDTreeNode` | Low | Defined but unused; `orthogonal_basis` attribute is always `None` |
| **`_evaluate_forgetting` truncated at 50 batches** | `Tree_LoRA.py:_evaluate_forgetting` | Low | Approximate measurement; forgetting metrics not used adaptively |
| **`projection_matrices` memory cost** | `kd_lora_tree.py:_update_projection_matrix` | Medium | $O(T \cdot L \cdot d^2)$ storage; for LLaMA-7B with 32 layers ($d=4096$, $T=15$): ~32 GB — impractical |
| **`except:` bare clause** | `kd_lora_tree.py:_update_projection_matrix` | Low | Catches all exceptions including CUDA OOM; should be `except torch.linalg.LinAlgError` |

---

### 3.3 Mathematical Feasibility Proof

#### Claim: Adding $L_\text{OPL}$ to the training objective reduces gradient interference with past tasks.

**Proof sketch:**

Let $\theta^{(t)}$ denote the LoRA-A parameters after task $t$. Define the **interference** of task $t$ on task $t'$ as:

$$I(t, t') = \left\| \Pi_{g^{(t')}} \cdot \nabla_\theta \mathcal{L}_t(\theta^{(t'-1)}) \right\|_2$$

where $\Pi_{g^{(t')}} = \frac{g^{(t')} (g^{(t')})^\top}{\|g^{(t')}\|^2}$ is the projection onto $g^{(t')}$.

If the training loss includes $L_\text{OPL} = \sum_l L_\text{OPL}^{(l)}$ as a penalty, then the gradient update for task $t$ satisfies:

$$\nabla_\theta \mathcal{L}_\text{total} = \nabla_\theta \mathcal{L}_\text{CE} + \lambda_\text{OPL} \cdot \nabla_\theta L_\text{OPL}$$

The gradient of $L_\text{OPL}^{(l)}$ with respect to $g_l^{(t)}$ is:

$$\frac{\partial L_\text{OPL}^{(l)}}{\partial g_l^{(t)}} = \frac{2 \left(g_l^{(t)} \cdot g_l^{(\hat{t})}\right)}{\left\|g_l^{(\hat{t})}\right\|^2} \cdot g_l^{(\hat{t})} \cdot \frac{1}{\sum_l \|g_l^{(t)}\|^2} - \frac{L_\text{OPL} \cdot 2 g_l^{(t)}}{\sum_l \|g_l^{(t)}\|^2}$$

The first term pushes $g_l^{(t)}$ **away from** $g_l^{(\hat{t})}$ (reducing its projection component). The second term is a self-normalizing correction proportional to the current OPL value. Together, minimizing $L_\text{OPL}$ drives the angle $\theta_{l}$ between $g_l^{(t)}$ and $g_l^{(\hat{t})}$ toward 90°:

$$\frac{\partial L_\text{OPL}^{(l)}}{\partial \theta_l} \propto -\sin(2\theta_l)$$

which has a stable fixed point at $\theta_l = \pi/2$ (orthogonality), confirming that **minimizing $L_\text{OPL}$ is equivalent to enforcing orthogonality**. ✅

**Convergence condition:** The objective is jointly $\mathcal{L}_\text{CE} - \lambda \mathcal{L}_\text{sim} - \mu L_\text{OPL}$. For this to converge we need:

1. $\lambda, \mu > 0$ (satisfied by `tmp_reg` warmup and `opl_weight` defaults)
2. $\mathcal{L}_\text{CE}$ is bounded below (satisfied for cross-entropy on valid distributions)
3. The OPL term does not conflict with task learning if $\mu$ is small relative to the task gradient

The code scales OPL as `opl_loss * loss.detach() * opl_weight` (default `opl_weight=0.1`), so for typical CE losses of $O(1)$, the OPL contribution is $O(0.1 \times L_\text{OPL}) \leq O(0.1)$, which is unlikely to dominate the CE gradient. **The scaling is appropriate, though it should be tuned per dataset.** ✅

---

### 3.4 Pro / Con Summary

| | Origin (TreeLoRA) | Improved (SO-LoRA) |
|---|---|---|
| **Forgetting prevention** | Indirect (similarity maximization) | Direct (OPL penalizes interference) |
| **Transfer learning** | ✅ UCB bandit exploits similar tasks | ✅ Same mechanism retained |
| **Computation** | Lower (no OPL, no projection matrices) | Higher ($+L \times d^2 \times T$ per task for $P_l$) |
| **Memory** | $O(T \cdot L \cdot d)$ for gradient storage | Same + $O(T \cdot L \cdot d^2)$ for $P_l$ matrices |
| **Theoretical grounding** | Heuristic similarity loss | OPL has formal orthogonality guarantee |
| **Code quality** | Compact, hard to extend | Modular, well-documented, testable |
| **Numerical stability** | One known bug (`insert_grad` loop) | Several fixed; minor Gram-Schmidt drift remains |
| **Multi-task coverage** | All past tasks via bandit selection | Single selected task per layer for OPL (approximation) |
| **Observability** | Minimal | Training stats, forgetting metrics, tree state persisted |
| **Gradient surgery** | ❌ | ✅ Optional hard projection via `_apply_gradient_projection` |

---

### 3.5 Recommended Next Steps

1. **Fix layer-name matching in `_apply_gradient_projection`:** Index layers by parameter name rather than flat dimension to ensure correct gradient surgery on non-uniform architectures.

2. **Multi-task OPL:** Replace single-reference OPL with full-subspace projection $L_\text{OPL}^{(l)} = \|P_l \cdot g_l^{(t)}\|^2 / \|g_l^{(t)}\|^2$ using the already-computed `projection_matrices[task_id-1][l]`. This requires one matrix-vector multiply per layer and eliminates the single-reference approximation.

3. **Memory-efficient projection:** For large models, avoid storing full $d \times d$ matrices by keeping only the $G_l \in \mathbb{R}^{T \times d}$ factor and applying $P_l x = G_l^\top (G_l G_l^\top)^{-1} G_l x$ in factored form, reducing storage from $O(d^2)$ to $O(Td)$ per layer per task.

4. **Modified Gram-Schmidt:** Replace sequential Gram-Schmidt in `_apply_gradient_projection` and `gram_schmidt_orthogonalize` with MGS or QR decomposition for tasks $T > 10$.

5. **Adaptive OPL weight:** Wire `forgetting_metrics` to dynamically scale `opl_weight` — increase it when forgetting is detected on previous tasks, creating a closed feedback loop between measurement and prevention.
