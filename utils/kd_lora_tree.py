"""
KD_LoRA_Tree with Orthogonal Projection Loss (OPL)

Tree-structured LoRA Selection with Orthogonal Projection Loss for
Continual Learning in Large Language Models.

Key Components:
1. KDTreeNode: Tree node for hierarchical gradient organization
2. KD_LoRA_Tree: Main class managing tree structure and OPL computation
3. Orthogonal Projection Loss: Prevents catastrophic forgetting by ensuring
   new gradients are orthogonal to past task gradients

Algorithm:
- Builds a KD-tree structure based on LoRA gradient similarities across tasks
- Uses UCB (Upper Confidence Bound) / LCB for exploration-exploitation in task selection
- Applies OPL to penalize gradient interference with previous tasks
- Maintains per-layer gradient memory for fine-grained forgetting prevention
"""

import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple

from utils.utils import print_rank_0


class KDTreeNode:
    """
    KD-Tree Node for hierarchical organization of task gradients.

    The tree is built based on gradient similarities at each LoRA layer depth.
    This enables efficient retrieval of relevant past task gradients for OPL.

    Attributes:
        task_indices: List of task IDs contained in this subtree
        depth: Current depth (corresponds to LoRA layer index)
        left/right: Child nodes split by gradient similarity
        mean_vector: Mean gradient vector for this node's tasks
        median_similarity: Threshold for left/right split
    """

    def __init__(self, task_indices: List[int], depth: int,
                 grads_tensor: torch.Tensor, lora_depth: int):
        """
        Initialize and build the KD-tree node.

        Args:
            task_indices: List of global task indices in this node
            depth: Current depth of the node (0 to lora_depth-1)
            grads_tensor: Tensor of shape (num_tasks, lora_depth, feature_dim)
            lora_depth: Maximum depth of the tree (number of LoRA layers)
        """
        self.task_indices = task_indices
        self.depth = depth
        self.left = None
        self.right = None
        self.is_leaf = False
        self.lora_depth = lora_depth
        self.mean_vector = None
        self.median_similarity = None

        # Statistics for OPL
        self.gradient_norm = None
        self.orthogonal_basis = None  # For Gram-Schmidt orthogonalization

        self.build_node(grads_tensor)

    def build_node(self, grads_tensor: torch.Tensor):
        """
        Build the tree node by splitting tasks based on gradient similarity.

        Uses the gradient at the current depth to compute similarities,
        then splits tasks into left (high similarity) and right (low similarity).
        """
        # Leaf condition: max depth reached or single task
        if self.depth >= self.lora_depth or len(self.task_indices) <= 1:
            self.is_leaf = True
            if len(self.task_indices) > 0 and self.depth < grads_tensor.shape[1]:
                current_grads = grads_tensor[self.task_indices, self.depth, :]
                self.mean_vector = current_grads.mean(dim=0)
                self.gradient_norm = torch.norm(self.mean_vector).item()
            return

        # Get gradients at current depth for all tasks in this node
        current_grads = grads_tensor[self.task_indices, self.depth, :]  # Shape: (N, D)

        # Compute mean vector as the splitting hyperplane normal
        self.mean_vector = current_grads.mean(dim=0)  # Shape: (D,)
        self.gradient_norm = torch.norm(self.mean_vector).item()

        # Normalize mean vector for stable similarity computation
        mean_norm = torch.norm(self.mean_vector)
        if mean_norm > 1e-8:
            normalized_mean = self.mean_vector / mean_norm
        else:
            normalized_mean = self.mean_vector

        # Compute similarities (dot product with normalized mean)
        similarities = torch.mv(current_grads, normalized_mean)  # Shape: (N,)

        # Split by median similarity
        self.median_similarity = torch.median(similarities).item()

        # Partition tasks
        left_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() >= self.median_similarity
        ]
        right_indices = [
            self.task_indices[i]
            for i in range(len(self.task_indices))
            if similarities[i].item() < self.median_similarity
        ]

        # Handle edge case: all tasks on one side
        if len(left_indices) == 0 or len(right_indices) == 0:
            median_idx = len(self.task_indices) // 2
            left_indices = self.task_indices[:median_idx]
            right_indices = self.task_indices[median_idx:]

        # Recursively build children
        if len(left_indices) > 0:
            self.left = KDTreeNode(left_indices, self.depth + 1, grads_tensor, self.lora_depth)
        if len(right_indices) > 0:
            self.right = KDTreeNode(right_indices, self.depth + 1, grads_tensor, self.lora_depth)

    def get_orthogonal_complement(self, query_grad: torch.Tensor) -> torch.Tensor:
        """
        Compute the component of query_grad orthogonal to this node's mean vector.

        This is used for OPL to find the gradient direction that doesn't
        interfere with past task knowledge.

        Args:
            query_grad: Gradient vector to orthogonalize

        Returns:
            Orthogonal component of query_grad
        """
        if self.mean_vector is None:
            return query_grad

        mean_norm_sq = torch.dot(self.mean_vector, self.mean_vector)
        if mean_norm_sq < 1e-8:
            return query_grad

        # Project query onto mean vector
        projection_coeff = torch.dot(query_grad, self.mean_vector) / mean_norm_sq
        projection = projection_coeff * self.mean_vector

        # Return orthogonal component
        return query_grad - projection

    def __str__(self, level: int = 0) -> str:
        """Pretty print the tree structure."""
        indent = "  " * level
        if self.is_leaf:
            return f"{indent}Leaf(depth={self.depth}, tasks={self.task_indices})\n"
        else:
            mean_preview = self.mean_vector[:3].tolist() if self.mean_vector is not None else []
            mean_str = ", ".join([f"{x:.4f}" for x in mean_preview])
            result = (
                f"{indent}Node(depth={self.depth}, tasks={self.task_indices}, "
                f"mean=[{mean_str}...], median_sim={self.median_similarity:.4f})\n"
            )
            if self.left:
                result += self.left.__str__(level + 1)
            if self.right:
                result += self.right.__str__(level + 1)
            return result


def tree_lora_loss(current_grad: torch.Tensor, all_grad: torch.Tensor,
                   task_id: int, prev_id_matrix: torch.Tensor,
                   multiple_module: bool = True) -> torch.Tensor:
    """
    Compute the Tree-LoRA regularization loss.

    This loss encourages the current gradient to be similar to selected
    past task gradients, promoting positive transfer while the OPL
    component ensures orthogonality where needed.

    Args:
        current_grad: Current task gradient (lora_depth, feature_dim)
        all_grad: All past task gradients (num_tasks, lora_depth, feature_dim)
        task_id: Current task ID
        prev_id_matrix: Selected past task indices per layer
        multiple_module: Whether to compute per-layer loss

    Returns:
        Regularization loss (negative similarity to maximize)
    """
    reg_loss = None

    if multiple_module:
        # Per-layer loss computation
        for depth_id, prev_task_id in enumerate(prev_id_matrix):
            layer_loss = -(current_grad[depth_id] * all_grad[prev_task_id][depth_id]).sum()
            if reg_loss is None:
                reg_loss = layer_loss
            else:
                reg_loss = reg_loss + layer_loss
    else:
        # Single aggregated loss
        prev_id = prev_id_matrix[0]
        reg_loss = -(current_grad.reshape(-1) * all_grad[prev_id].reshape(-1)).sum()

    return reg_loss


def orthogonal_projection_loss(current_grad: torch.Tensor,
                                past_grads: torch.Tensor,
                                prev_id_matrix: torch.Tensor,
                                normalize: bool = True) -> torch.Tensor:
    """
    Compute Orthogonal Projection Loss (OPL) in gradient space.

    OPL penalizes the component of the current gradient that lies in the
    subspace spanned by past task gradients. This prevents overwriting
    important directions learned for previous tasks.

    Mathematical formulation:
    OPL = ||proj(g_current, span(G_past))||^2 / ||g_current||^2

    where proj(v, S) is the projection of v onto subspace S.

    Args:
        current_grad: Current task gradient (lora_depth, feature_dim)
        past_grads: Past task gradients (num_tasks, lora_depth, feature_dim)
        prev_id_matrix: Selected past task indices per layer
        normalize: Whether to normalize by current gradient norm

    Returns:
        OPL loss value
    """
    device = current_grad.device
    total_projection_norm_sq = torch.tensor(0.0, device=device)
    total_current_norm_sq = torch.tensor(0.0, device=device)

    num_layers = current_grad.shape[0]

    for depth_id in range(num_layers):
        current_layer_grad = current_grad[depth_id]  # (feature_dim,)
        selected_task_id = prev_id_matrix[depth_id].item() if prev_id_matrix.dim() > 0 else prev_id_matrix.item()

        past_layer_grad = past_grads[selected_task_id, depth_id]  # (feature_dim,)

        # Compute projection of current onto past
        past_norm_sq = torch.dot(past_layer_grad, past_layer_grad)

        if past_norm_sq > 1e-8:
            proj_coeff = torch.dot(current_layer_grad, past_layer_grad) / past_norm_sq
            projection = proj_coeff * past_layer_grad
            total_projection_norm_sq = total_projection_norm_sq + torch.dot(projection, projection)

        total_current_norm_sq = total_current_norm_sq + torch.dot(current_layer_grad, current_layer_grad)

    if normalize and total_current_norm_sq > 1e-8:
        opl_loss = total_projection_norm_sq / total_current_norm_sq
    else:
        opl_loss = total_projection_norm_sq

    return opl_loss


def gram_schmidt_orthogonalize(current_grad: torch.Tensor,
                                past_grads: List[torch.Tensor]) -> torch.Tensor:
    """
    Orthogonalize current gradient against all past gradients using Gram-Schmidt.

    This produces a gradient direction that is orthogonal to all past task
    gradient directions, ensuring no interference with previously learned knowledge.

    Args:
        current_grad: Current gradient to orthogonalize (feature_dim,)
        past_grads: List of past gradients to orthogonalize against

    Returns:
        Orthogonalized gradient
    """
    orthogonal_grad = current_grad.clone()

    for past_grad in past_grads:
        past_norm_sq = torch.dot(past_grad, past_grad)
        if past_norm_sq > 1e-8:
            proj_coeff = torch.dot(orthogonal_grad, past_grad) / past_norm_sq
            orthogonal_grad = orthogonal_grad - proj_coeff * past_grad

    return orthogonal_grad


class KD_LoRA_Tree:
    """
    KD-Tree based LoRA gradient management with Orthogonal Projection Loss.

    This class implements:
    1. Tree-structured storage of per-layer gradients from past tasks
    2. UCB/LCB-based exploration for selecting relevant past tasks
    3. Orthogonal Projection Loss computation for forgetting prevention
    4. Dynamic tree updates as new tasks are learned

    The key insight is that different LoRA layers may need different
    past task references, so we maintain per-layer selection and OPL.
    """

    def __init__(self, args):
        """
        Initialize the KD-LoRA Tree.

        Args:
            args: Training arguments containing:
                - num_tasks: Total number of tasks
                - reg: Regularization coefficient
                - global_rank: For distributed training
                - opl_weight: Weight for OPL loss (optional, default 0.1)
                - use_opl: Whether to use OPL (optional, default True)
        """
        self.args = args
        self.root = None

        # Gradient storage
        self.all_accumulate_grads = [None] * getattr(args, 'num_tasks', 8)
        self.current_grad = None
        self.all_grad = None
        self.all_grad_device = None

        # Tree structure
        self.kd_tree_root = None

        # Selection tracking
        self.num_of_selected = None
        self.sim = None

        # Epoch/step tracking
        self.tmp_rounds = -1
        self.total_rounds = 0
        self.tmp_reg = 0

        # Task tracking
        self.last_task_id = -1
        self.mask = None
        self.mask_tensor = None

        # OPL configuration
        self.opl_weight = getattr(args, 'opl_weight', 0.1)
        self.use_opl = getattr(args, 'use_opl', True)
        self.opl_history = []

        # Orthogonal projection matrices (for efficient OPL)
        self.projection_matrices = {}

    def new_epoch_init(self, train_dataloader_len: int):
        """
        Initialize for a new training epoch.

        Args:
            train_dataloader_len: Number of batches in the epoch
        """
        self.current_grad = None
        self.all_grad = None
        self.num_of_selected = None
        self.tmp_rounds = -1
        self.total_rounds = train_dataloader_len
        self.sim = None
        self.tmp_reg = 0

    def step(self):
        """Called at each training step to update regularization schedule."""
        self.tmp_rounds += 1
        # Linear warmup of regularization
        self.tmp_reg = self.args.reg * self.tmp_rounds / self.total_rounds

    def insert_grad(self, _grad_current: torch.Tensor):
        """
        Insert current step's gradient into accumulator.

        Uses running average to aggregate gradients across the epoch.

        Args:
            _grad_current: Current gradient tensor (lora_depth, feature_dim)
        """
        if self.current_grad is None:
            self.current_grad = _grad_current.detach() / self.total_rounds
        else:
            frac = 1.0 / self.total_rounds
            self.current_grad = self.current_grad + _grad_current.detach() * frac

    def end_task(self, task_id: int):
        """
        Finalize gradient storage and update tree structure after task completion.

        Args:
            task_id: ID of the completed task
        """
        if self.args.reg <= 0 or self.current_grad is None:
            return

        # Store accumulated gradients
        self.all_accumulate_grads[task_id] = self.current_grad.clone()

        lora_depth = self.current_grad.shape[0]

        print_rank_0(f"\n[TreeLoRA] Updating KD-Tree after task {task_id}...",
                     self.args.global_rank)

        # Collect valid gradients
        valid_grads = [grad for grad in self.all_accumulate_grads[:task_id + 1]
                       if grad is not None]

        if not valid_grads:
            print_rank_0("[TreeLoRA] No gradients to build tree.", self.args.global_rank)
            return

        # Stack gradients: (num_valid_tasks, lora_depth, feature_dim)
        grads_tensor = copy.deepcopy(torch.stack(valid_grads))

        # Compute gradient differences for better tree structure
        # This captures task-specific directions rather than absolute positions
        for i in range(grads_tensor.shape[0] - 1, 0, -1):
            grads_tensor[i] = grads_tensor[i] - grads_tensor[i - 1]

        # Get task indices for valid gradients
        task_ids = [i for i, grad in enumerate(self.all_accumulate_grads[:task_id + 1])
                    if grad is not None]

        # Build KD-tree
        self.kd_tree_root = KDTreeNode(
            task_indices=task_ids,
            depth=0,
            grads_tensor=grads_tensor,
            lora_depth=lora_depth
        )

        # Update projection matrix for efficient OPL
        self._update_projection_matrix(task_id)

        print_rank_0("[TreeLoRA] KD-Tree updated successfully.", self.args.global_rank)
        print_rank_0(f"[TreeLoRA] Tree structure:\n{self.kd_tree_root}", self.args.global_rank)

    def _update_projection_matrix(self, task_id: int):
        """
        Update projection matrix for efficient OPL computation.

        The projection matrix P = G(G^T G)^{-1} G^T projects onto the
        subspace spanned by past gradients. I - P projects onto the
        orthogonal complement.

        Args:
            task_id: Current task ID
        """
        valid_grads = [grad for grad in self.all_accumulate_grads[:task_id + 1]
                       if grad is not None]

        if len(valid_grads) == 0:
            return

        # For each LoRA layer, compute projection matrix
        lora_depth = valid_grads[0].shape[0]

        self.projection_matrices[task_id] = []

        for layer_idx in range(lora_depth):
            # Collect layer gradients: (num_tasks, feature_dim)
            layer_grads = torch.stack([g[layer_idx] for g in valid_grads])

            # Compute G^T G
            GtG = layer_grads @ layer_grads.T  # (num_tasks, num_tasks)

            # Regularized inverse for numerical stability
            try:
                GtG_inv = torch.linalg.pinv(GtG + 1e-6 * torch.eye(GtG.shape[0], device=GtG.device))
                # Projection matrix: G(G^T G)^{-1} G^T
                proj_matrix = layer_grads.T @ GtG_inv @ layer_grads
                self.projection_matrices[task_id].append(proj_matrix)
            except:
                self.projection_matrices[task_id].append(None)

    def tree_search(self, task_id: int, device: torch.device) -> torch.Tensor:
        """
        Search the tree to select relevant past tasks for each LoRA layer.

        Uses UCB (Upper Confidence Bound) for exploration-exploitation trade-off:
        - Exploits tasks with high gradient similarity
        - Explores less-visited tasks to discover beneficial transfers

        The tree structure biases selection towards related task clusters.

        Args:
            task_id: Current task ID
            device: Computation device

        Returns:
            prev_id_matrix: Selected past task indices per layer (lora_depth,)
        """
        cosine_sim = False  # Use L1 distance by default

        # Initialize all_grad tensor if needed
        if self.all_grad is None:
            self.all_grad = torch.stack(
                self.all_accumulate_grads[:task_id], dim=0
            ).to(device, non_blocking=True)
            self.all_grad_device = self.all_grad

            if cosine_sim:
                # Normalize for cosine similarity
                norms = torch.norm(self.all_grad, dim=2, keepdim=True).mean(dim=0, keepdim=True)
                self.all_grad = self.all_grad / (norms + 1e-5)

            # Initialize tracking tensors
            if self.sim is None:
                self.sim = torch.zeros((task_id, self.all_grad.shape[1]), device=device)
                self.num_of_selected = torch.zeros(
                    self.args.num_tasks, self.all_grad.shape[1]
                ).to(device, non_blocking=True)

        # Clone similarity for modification
        sim = self.sim.clone()

        # Compute average similarity
        valid_mask = self.num_of_selected[:task_id, :] > 0
        if valid_mask.any():
            sim[valid_mask] = sim[valid_mask] / self.num_of_selected[:task_id, :][valid_mask]

        # Add exploration bonus (UCB for cosine sim, LCB for L1 distance)
        if self.num_of_selected is not None:
            exploration_bonus = (
                1.0 / torch.sqrt(2 * self.num_of_selected[:task_id, :] + 1e-5)
                * math.sqrt(math.log(2 * self.total_rounds * (self.tmp_rounds + 1) * (self.tmp_rounds + 2)))
            )
            if cosine_sim:
                sim = sim + exploration_bonus  # UCB: add bonus
            else:
                sim = sim - exploration_bonus  # LCB: subtract bonus
                sim = -sim  # Negate for L1 (minimize distance)

        # Shift to positive values
        sim = sim + torch.min(sim).abs() + 1e-5

        # Tree-guided selection
        # First, sample a task based on overall similarity
        overall_sim = torch.sum(sim, dim=1)
        first_idx = torch.multinomial(
            torch.softmax(overall_sim, dim=0),
            num_samples=1,
            replacement=True
        ).item()

        # Boost similarity for tasks in the same tree branch
        if self.kd_tree_root is not None and self.kd_tree_root.left is not None:
            similarity_boost = 1.0

            if first_idx in self.kd_tree_root.left.task_indices:
                if self.kd_tree_root.left.median_similarity is not None:
                    similarity_boost = min(self.kd_tree_root.left.median_similarity, 1.5)
                sim[self.kd_tree_root.left.task_indices] *= similarity_boost
            else:
                if self.kd_tree_root.right is not None and self.kd_tree_root.right.median_similarity is not None:
                    similarity_boost = min(self.kd_tree_root.right.median_similarity, 1.5)
                    sim[self.kd_tree_root.right.task_indices] *= similarity_boost

            if self.tmp_rounds % 100 == 0:
                print_rank_0(
                    f'\033[34m[TreeLoRA] First selected task: {first_idx}, '
                    f'similarity boost: {similarity_boost:.4f}\033[0m',
                    self.args.global_rank
                )

        # Normalize and mask future tasks
        sim_range = torch.max(sim) - torch.min(sim) + 1e-5
        sim = sim / sim_range
        sim[task_id:, :] = -torch.inf  # Mask future tasks

        # Softmax for probability distribution
        sim_normalized = torch.softmax(sim, dim=0)

        # Sample past task for each layer
        prev_id_matrix = torch.multinomial(
            sim_normalized.T,
            num_samples=1,
            replacement=True
        ).reshape(-1)

        # Update selection counts
        layer_indices = torch.arange(sim.shape[1], device=device)
        self.num_of_selected[prev_id_matrix, layer_indices] += 1

        # Update similarity estimates
        self._update_similarity(prev_id_matrix, device)

        return prev_id_matrix

    def _update_similarity(self, prev_id_matrix: torch.Tensor, device: torch.device):
        """
        Update similarity estimates based on current selection.

        Args:
            prev_id_matrix: Selected task indices per layer
            device: Computation device
        """
        if self.sim is None or self.current_grad is None:
            return

        cosine_sim = False

        for depth_idx, prev_id in enumerate(prev_id_matrix):
            if cosine_sim:
                # Cosine similarity
                self.sim[prev_id, depth_idx] += torch.dot(
                    self.current_grad[depth_idx],
                    self.all_grad[prev_id, depth_idx]
                ).item()
            else:
                # Negative L1 distance (to be maximized)
                self.sim[prev_id, depth_idx] -= torch.sum(
                    torch.abs(self.current_grad[depth_idx] - self.all_grad[prev_id, depth_idx])
                ).item()

    def get_loss(self, _grad_current: torch.Tensor, loss: torch.Tensor,
                 task_id: int, prev_id_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute combined Tree-LoRA regularization loss with OPL.

        The total loss combines:
        1. Tree-LoRA loss: Encourages similarity with selected past tasks
        2. OPL loss: Penalizes non-orthogonal components to past gradients

        Args:
            _grad_current: Current gradient tensor
            loss: Original task loss
            task_id: Current task ID
            prev_id_matrix: Selected past task indices

        Returns:
            Combined regularization loss
        """
        # Base Tree-LoRA loss (similarity-based)
        reg_loss = tree_lora_loss(
            _grad_current,
            self.all_grad_device,
            task_id,
            prev_id_matrix
        )

        # Normalize by loss magnitude and apply warmup
        reg_loss = reg_loss / (reg_loss.detach().clone().abs() + 1e-5) * loss.detach().clone() * self.tmp_reg

        #? Add OPL component if enabled
        if self.use_opl and task_id > 0:
            opl_loss = self._compute_opl_loss(_grad_current, task_id, prev_id_matrix)

            # Scale OPL loss relative to task loss
            if opl_loss.abs() > 1e-8:
                opl_loss = opl_loss * loss.detach().clone() * self.opl_weight
                reg_loss = reg_loss + opl_loss

                # Track OPL for monitoring
                self.opl_history.append(opl_loss.item())

        return reg_loss

    def _compute_opl_loss(self, current_grad: torch.Tensor,
                          task_id: int,
                          prev_id_matrix: torch.Tensor) -> torch.Tensor:
        """
        Compute Orthogonal Projection Loss.

        OPL = sum_l ||proj(g_l, g_l^{prev})||^2 / ||g_l||^2

        where g_l is the current gradient at layer l and g_l^{prev} is the
        selected past task gradient at layer l.

        Args:
            current_grad: Current gradient (lora_depth, feature_dim)
            task_id: Current task ID
            prev_id_matrix: Selected past task indices

        Returns:
            OPL loss value
        """
        return orthogonal_projection_loss(
            current_grad,
            self.all_grad_device,
            prev_id_matrix,
            normalize=True
        )

    def get_orthogonalized_gradient(self, gradient: torch.Tensor,
                                     task_id: int) -> torch.Tensor:
        """
        Get gradient orthogonalized against all past task gradients.

        This can be used for gradient modification during backward pass.

        Args:
            gradient: Gradient to orthogonalize
            task_id: Current task ID

        Returns:
            Orthogonalized gradient
        """
        if task_id == 0 or len(self.projection_matrices) == 0:
            return gradient

        # Get most recent projection matrix
        proj_task_id = task_id - 1
        if proj_task_id not in self.projection_matrices:
            return gradient

        proj_matrices = self.projection_matrices[proj_task_id]
        orthogonal_grad = gradient.clone()

        for layer_idx in range(gradient.shape[0]):
            if layer_idx < len(proj_matrices) and proj_matrices[layer_idx] is not None:
                layer_grad = gradient[layer_idx]
                proj_matrix = proj_matrices[layer_idx].to(gradient.device)

                # Project onto past gradient subspace
                projection = proj_matrix @ layer_grad

                # Subtract projection to get orthogonal component
                orthogonal_grad[layer_idx] = layer_grad - projection

        return orthogonal_grad

    def get_mask(self, class_mask, task_id: int, args, logits: torch.Tensor):
        """
        Get task-specific mask for output masking (if needed).

        Args:
            class_mask: Class masks per task
            task_id: Current task ID
            args: Arguments with nb_classes
            logits: Output logits
        """
        if self.mask is None or task_id != self.last_task_id:
            self.last_task_id = task_id
            self.mask = class_mask[task_id]
            self.mask_tensor = torch.full(
                (args.nb_classes,), False,
                dtype=torch.bool, device=logits.device
            )
            self.mask_tensor[self.mask] = True


class OPLRegularizer(nn.Module):
    """
    Standalone OPL regularizer module for integration with training loops.

    This can be used as a drop-in loss component that enforces gradient
    orthogonality to past task gradients.
    """

    def __init__(self, opl_weight: float = 0.1, normalize: bool = True):
        super().__init__()
        self.opl_weight = opl_weight
        self.normalize = normalize
        self.past_gradients = []

    def add_task_gradient(self, gradient: torch.Tensor):
        """Add a task's gradient to memory."""
        self.past_gradients.append(gradient.detach().clone())

    def forward(self, current_grad: torch.Tensor) -> torch.Tensor:
        """
        Compute OPL loss for current gradient.

        Args:
            current_grad: Current gradient tensor

        Returns:
            OPL loss value
        """
        if len(self.past_gradients) == 0:
            return torch.tensor(0.0, device=current_grad.device)

        total_projection_norm_sq = 0.0
        total_current_norm_sq = torch.sum(current_grad ** 2)

        current_flat = current_grad.reshape(-1)

        for past_grad in self.past_gradients:
            past_flat = past_grad.reshape(-1).to(current_grad.device)

            # Compute projection
            past_norm_sq = torch.dot(past_flat, past_flat)
            if past_norm_sq > 1e-8:
                proj_coeff = torch.dot(current_flat, past_flat) / past_norm_sq
                projection = proj_coeff * past_flat
                total_projection_norm_sq += torch.dot(projection, projection)

        if self.normalize and total_current_norm_sq > 1e-8:
            opl_loss = total_projection_norm_sq / total_current_norm_sq
        else:
            opl_loss = total_projection_norm_sq

        return self.opl_weight * opl_loss