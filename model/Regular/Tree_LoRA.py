"""
Tree_LoRA: Tree-structured LoRA Selection with Orthogonal Projection Loss (OPL)

This module implements the TreeLoRA algorithm for continual learning in LLMs.

=== Algorithm Overview ===

TreeLoRA addresses catastrophic forgetting in continual learning by:

1. **Tree-Structured Gradient Memory**: Organizes past task gradients in a KD-tree
   structure based on similarity at each LoRA layer, enabling efficient retrieval
   of relevant past knowledge.

2. **Orthogonal Projection Loss (OPL)**: Ensures new task gradients are orthogonal
   to past task gradient subspaces, preventing interference with learned knowledge.

   Mathematical formulation:
   L_OPL = ||proj(g_t, G_{<t})||^2 / ||g_t||^2

   where g_t is the current gradient and G_{<t} spans past gradients.

3. **UCB-based Task Selection**: Uses Upper Confidence Bound (UCB) strategy for
   exploration-exploitation trade-off when selecting relevant past tasks.

4. **Per-Layer Regularization**: Applies regularization independently at each
   LoRA layer, allowing fine-grained control over knowledge preservation.

=== Key Components ===

- Tree_LoRA: Main class implementing the continual learning algorithm
- KD_LoRA_Tree: Tree structure for gradient storage and retrieval
- OPL computation: Gradient-level orthogonalization

=== Usage ===

The algorithm is activated when args.reg > 0. Key hyperparameters:
- reg: Overall regularization strength
- lamda_1: Weight for similarity-based loss
- lamda_2: Weight for tree structure regularization
- opl_weight: Weight for OPL loss (set in KD_LoRA_Tree)
"""

import copy
import json
import os
import pickle
import time
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from typing import Dict, List, Optional, Tuple

from model.base_model import CL_Base_Model
from utils.kd_lora_tree import KD_LoRA_Tree, orthogonal_projection_loss, gram_schmidt_orthogonalize
from utils.model.model_utils import TIKTOK
from utils.utils import print_rank_0, to_device, get_all_reduce_mean


class Tree_LoRA(CL_Base_Model):
    """
    Tree-structured LoRA for Continual Learning with Orthogonal Projection Loss.

    This class extends the base continual learning model with:
    - Tree-structured gradient memory for efficient past task retrieval
    - Orthogonal Projection Loss (OPL) for catastrophic forgetting mitigation
    - Per-layer gradient regularization
    - UCB-based exploration for task selection

    Attributes:
        lamda_1: Weight for orthogonal/similarity loss (default: 0.5)
        lamda_2: Weight for tree structure regularization (default: 0)
        kd_lora_tree: Tree structure for gradient management
        tiktok: Timing utility for profiling

    Args:
        model: Base LLM with LoRA layers
        tokenizer: Associated tokenizer
        optimizer: Training optimizer
        train_task_list: Dict of training dataloaders
        eval_task_list: Dict of evaluation dataloaders
        test_task_list: Dict of test dataloaders
        args: Training arguments
        lamda_1: Orthogonal loss weight
        lamda_2: Tree regularization weight
    """

    def __init__(self,
                 model, tokenizer, optimizer,
                 train_task_list, eval_task_list, test_task_list,
                 args,
                 lamda_1: float = 0.5,
                 lamda_2: float = 0):

        super().__init__(model, tokenizer, optimizer,
                         train_task_list, eval_task_list, test_task_list, args)

        # Regularization hyperparameters
        self.lamda_1 = getattr(args, 'lamda_1', lamda_1)
        self.lamda_2 = getattr(args, 'lamda_2', lamda_2)

        # Timing utility
        self.tiktok = TIKTOK(args)

        # Device setup
        if self.args.local_rank == -1:
            self.device = torch.device("cuda")
        else:
            torch.cuda.set_device(self.args.local_rank)
            self.device = torch.device("cuda", self.args.local_rank)

        # Initialize KD-LoRA Tree for gradient management
        num_task = len(self.train_task_list)
        args.num_tasks = num_task
        args.opl_weight = getattr(args, 'opl_weight', 0.1)
        args.use_opl = getattr(args, 'use_opl', True)

        self.kd_lora_tree = KD_LoRA_Tree(args)

        # Statistics tracking
        self.task_losses = []
        self.reg_losses = []
        self.opl_losses = []
        self.forgetting_metrics = {}

        # Gradient projection mode
        self.use_gradient_projection = getattr(args, 'use_gradient_projection', False)

        print_rank_0(
            f"\n[Tree_LoRA] Initialized with:"
            f"\n  - lamda_1 (similarity weight): {self.lamda_1}"
            f"\n  - lamda_2 (tree reg weight): {self.lamda_2}"
            f"\n  - reg: {self.args.reg}"
            f"\n  - opl_weight: {args.opl_weight}"
            f"\n  - use_opl: {args.use_opl}"
            f"\n  - num_tasks: {num_task}",
            self.args.global_rank
        )

    def _extract_lora_gradients(self) -> List[torch.Tensor]:
        """
        Extract LoRA_A parameters for gradient computation.

        In the TreeLoRA algorithm, we use LoRA_A weights as proxies for
        gradient directions, as they capture the input projection learned
        for each task.

        Returns:
            List of LoRA_A parameter tensors
        """
        gradients = []
        for name, param in self.model.named_parameters():
            if "loranew_A" in name:
                gradients.append(param)
        return gradients

    def _compute_gradient_tensor(self, grad_list: List[torch.Tensor]) -> torch.Tensor:
        """
        Stack and reshape gradient list into tensor.

        Args:
            grad_list: List of gradient tensors

        Returns:
            Stacked tensor of shape (lora_depth, flattened_dim)
        """
        if len(grad_list) == 0:
            return None
        return torch.stack([g.reshape(-1) for g in grad_list], dim=0)

    def _apply_gradient_projection(self, task_id: int):
        """
        Apply gradient projection to make gradients orthogonal to past tasks.

        This is an alternative to loss-based OPL that directly modifies
        gradients during the backward pass.

        Args:
            task_id: Current task ID
        """
        if task_id == 0:
            return

        for name, param in self.model.named_parameters():
            if "loranew_A" in name and param.grad is not None:
                grad = param.grad.data.reshape(-1)

                # Orthogonalize against all past task gradients
                for past_task_id in range(task_id):
                    past_grad = self.kd_lora_tree.all_accumulate_grads[past_task_id]
                    if past_grad is not None:
                        # Find matching layer
                        past_flat = past_grad.reshape(-1).to(self.device)

                        if past_flat.shape[0] == grad.shape[0]:
                            # Gram-Schmidt step
                            dot = torch.dot(grad, past_flat)
                            norm_sq = torch.dot(past_flat, past_flat)
                            if norm_sq > 1e-8:
                                grad = grad - (dot / norm_sq) * past_flat

                param.grad.data = grad.reshape(param.grad.shape)


    def train_one_task(self, task: str, task_id: int, epochs: int):
        """
        Train on a single task with TreeLoRA regularization.

        The training loop:
        1. Forward pass to compute task loss
        2. Extract LoRA gradients
        3. Search tree for relevant past tasks
        4. Compute OPL regularization loss
        5. Combined backward pass
        6. Update tree structure after task completion

        Args:
            task: Task name/identifier
            task_id: Numeric task ID (0-indexed)
            epochs: Number of training epochs
        """
        train_dataloader = self.train_task_list[task]
        eval_dataloader = self.eval_task_list[task]

        total_steps = epochs * len(train_dataloader)
        train_dataloader_len = len(train_dataloader)

        progress_bar = tqdm(
            total=total_steps,
            leave=True,
            disable=(self.args.global_rank != 0),
            desc=f"Task {task_id}: {task}"
        )

        print_rank_0(
            f"\n{'='*60}\n"
            f"[Tree_LoRA] Starting Task {task_id}: {task}\n"
            f"  - Epochs: {epochs}\n"
            f"  - Steps per epoch: {train_dataloader_len}\n"
            f"  - Total steps: {total_steps}\n"
            f"{'='*60}",
            self.args.global_rank
        )

        for epoch in range(epochs):
            self.tiktok = TIKTOK(self.args)
            print_rank_0(
                f"\n[Tree_LoRA] Epoch {epoch + 1}/{epochs}, "
                f"Total batches: {train_dataloader_len}",
                self.args.global_rank
            )

            self.model.train()
            self.tiktok.print_time(self.args.global_rank)

            # Initialize tree for new epoch
            self.kd_lora_tree.new_epoch_init(train_dataloader_len)

            epoch_task_loss = 0.0
            epoch_reg_loss = 0.0
            step_count = 0

            for step, batch in enumerate(train_dataloader):
                # Update tree step counter
                if self.args.reg > 0:
                    self.kd_lora_tree.step()

                # Prepare batch
                del batch['sources']
                batch = to_device(batch, self.device)

                # Forward pass
                outputs = self.model(**batch, use_cache=False)
                loss = outputs.loss
                epoch_task_loss += loss.item()

                # Apply TreeLoRA regularization
                if self.args.reg > 0:
                    # === Step 1: Extract LoRA gradients ===
                    self.tiktok.tik()
                    _grad_current = self._extract_lora_gradients()
                    self.tiktok.tok(f"Extract_Grad_@Task{task_id}_Epoch{epoch}")

                    if len(_grad_current) > 0:
                        # === Step 2: Stack into tensor ===
                        self.tiktok.tik()
                        _grad_tensor = self._compute_gradient_tensor(_grad_current)

                        # Insert into tree accumulator
                        self.kd_lora_tree.insert_grad(_grad_tensor)
                        self.tiktok.tok(f"Insert_Grad_@Task{task_id}_Epoch{epoch}")

                        # === Step 3: Apply regularization for subsequent tasks ===
                        if task_id > 0:
                            # Tree search for relevant past tasks
                            self.tiktok.tik()
                            prev_id_matrix = self.kd_lora_tree.tree_search(
                                task_id, device=self.device
                            ) 
                            self.tiktok.tok(f"Tree_Search_@Task{task_id}_Epoch{epoch}")

                            # Compute combined loss (similarity + OPL)
                            self.tiktok.tik()
                            reg_loss = self.kd_lora_tree.get_loss(
                                _grad_tensor, loss, task_id, prev_id_matrix
                            )

                            # Combine: subtract reg_loss to maximize similarity / minimize OPL
                            loss = loss - reg_loss
                            epoch_reg_loss += reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss

                            self.tiktok.tok(f"Compute_Loss_@Task{task_id}_Epoch{epoch}")

                            # Periodic logging
                            if step % 100 == 0:
                                self._log_training_status(
                                    task_id, epoch, step,
                                    reg_loss, prev_id_matrix
                                )

                # Update progress bar
                if self.args.global_rank == 0:
                    progress_bar.update(1)
                    progress_bar.set_description(
                        f"Task {task_id} Epoch {epoch+1} | "
                        f"Loss: {loss.item():.4f}"
                    )

                # Backward pass
                self.tiktok.tik()
                self.model.backward(loss)

                # Optional: Apply gradient projection
                if self.use_gradient_projection and task_id > 0:
                    self._apply_gradient_projection(task_id)

                self.model.step()
                self.tiktok.tok('backward_and_step')

                step_count += 1

                # Periodic timing report
                if self.args.global_rank == 0 and step % 30 == 0:
                    self.tiktok.print_time()

            # Epoch summary
            avg_task_loss = epoch_task_loss / step_count
            avg_reg_loss = epoch_reg_loss / max(step_count, 1)

            self.task_losses.append(avg_task_loss)
            self.reg_losses.append(avg_reg_loss)

            print_rank_0(
                f"\n[Tree_LoRA] Epoch {epoch+1} Summary:"
                f"\n  - Avg Task Loss: {avg_task_loss:.4f}"
                f"\n  - Avg Reg Loss: {avg_reg_loss:.4f}",
                self.args.global_rank
            )

        # === Post-task processing ===
        progress_bar.close()

        # Save model checkpoint
        self._save_task_checkpoint(task_id)

        # Update tree structure
        if self.args.reg > 0:
            self.kd_lora_tree.end_task(task_id=task_id)
            print_rank_0(
                f"[Tree_LoRA] Task {task_id} complete. "
                f"Tree updated with gradient memory.",
                self.args.global_rank
            )

        # Evaluate forgetting (optional)
        if task_id > 0 and self.args.global_rank == 0:
            self._evaluate_forgetting(task_id)

    def _log_training_status(self, task_id: int, epoch: int, step: int,
                             reg_loss, prev_id_matrix):
        """Log detailed training status."""
        sim_info = self.kd_lora_tree.sim
        selected_info = self.kd_lora_tree.num_of_selected[:task_id]

        print_rank_0(
            f"\033[34m[TreeLoRA Status @ Step {step}]\033[0m",
            self.args.global_rank
        )
        print_rank_0(
            f"\033[34m  Similarity matrix:\n{sim_info}\033[0m",
            self.args.global_rank
        )
        print_rank_0(
            f"\033[34m  Selection counts:\n{selected_info}\033[0m",
            self.args.global_rank
        )
        print_rank_0(
            f"\033[34m  Selected tasks: {prev_id_matrix.tolist()}\033[0m",
            self.args.global_rank
        )
        # Fix: Convert tensor to float before formatting
        reg_loss_value = reg_loss.item() if torch.is_tensor(reg_loss) else reg_loss
        print_rank_0(
            f"\033[34m  Reg Loss: {reg_loss_value:.4f}\033[0m",
            self.args.global_rank
        )

        # Log OPL-specific info if available
        if hasattr(self.kd_lora_tree, 'opl_history') and len(self.kd_lora_tree.opl_history) > 0:
            recent_opl = self.kd_lora_tree.opl_history[-1]
            print_rank_0(
                f"\033[34m  Recent OPL Loss: {recent_opl:.4f}\033[0m",
                self.args.global_rank
            )

    def _save_task_checkpoint(self, task_id: int):
        """Save model checkpoint after task completion."""
        if self.args.output_dir is None:
            return

        print_rank_0(
            f'[Tree_LoRA] Saving model checkpoint for task {task_id}...',
            self.args.global_rank
        )

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(task_id))
            os.makedirs(peft_model_id, exist_ok=True)

            # Save model and tokenizer
            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)

            # Save training statistics
            stats_path = os.path.join(peft_model_id, f'training_stats_task_{task_id}.json')
            stats = {
                'task_id': task_id,
                'task_losses': self.task_losses,
                'reg_losses': self.reg_losses,
                'lamda_1': self.lamda_1,
                'lamda_2': self.lamda_2,
                'reg': self.args.reg,
            }
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            # Save tree state for analysis
            if self.args.reg > 0:
                tree_state_path = os.path.join(peft_model_id, f'tree_state_task_{task_id}.pkl')
                tree_state = {
                    'all_accumulate_grads': [
                        g.cpu() if g is not None else None
                        for g in self.kd_lora_tree.all_accumulate_grads
                    ],
                    'num_of_selected': (
                        self.kd_lora_tree.num_of_selected.cpu()
                        if self.kd_lora_tree.num_of_selected is not None else None
                    ),
                    'opl_history': getattr(self.kd_lora_tree, 'opl_history', []),
                }
                with open(tree_state_path, 'wb') as f:
                    pickle.dump(tree_state, f)

            print_rank_0(
                f'[Tree_LoRA] Checkpoint saved to {peft_model_id}',
                self.args.global_rank
            )

    def _evaluate_forgetting(self, current_task_id: int):
        """
        Evaluate catastrophic forgetting on previous tasks.

        Args:
            current_task_id: Current task ID after which to evaluate
        """
        print_rank_0(
            f"\n[Tree_LoRA] Evaluating forgetting after task {current_task_id}...",
            self.args.global_rank
        )

        self.model.eval()
        forgetting_results = {}

        with torch.no_grad():
            for prev_task_id, (task_name, eval_dataloader) in enumerate(self.eval_task_list.items()):
                if prev_task_id >= current_task_id:
                    break

                total_loss = 0.0
                num_batches = 0

                for batch in eval_dataloader:
                    if 'sources' in batch:
                        del batch['sources']
                    batch = to_device(batch, self.device)
                    outputs = self.model(**batch, use_cache=False)
                    total_loss += outputs.loss.item()
                    num_batches += 1

                    # Limit evaluation batches
                    if num_batches >= 50:
                        break

                avg_loss = total_loss / max(num_batches, 1)
                perplexity = math.exp(avg_loss) if avg_loss < 100 else float('inf')

                forgetting_results[task_name] = {
                    'loss': avg_loss,
                    'perplexity': perplexity
                }

                print_rank_0(
                    f"  Task {prev_task_id} ({task_name}): "
                    f"Loss={avg_loss:.4f}, PPL={perplexity:.2f}",
                    self.args.global_rank
                )

        self.forgetting_metrics[current_task_id] = forgetting_results
        self.model.train()

    def save_model(self, round: int):
        """
        Save final model with O_LoRA compatible format.

        Args:
            round: Current round/task number
        """
        if self.args.output_dir is None:
            return

        print_rank_0(
            f'[Tree_LoRA] Saving final model for round {round}...',
            self.args.global_rank
        )

        if self.args.global_rank == 0:
            peft_model_id = os.path.join(self.args.output_dir, str(round))
            os.makedirs(peft_model_id, exist_ok=True)

            self.model.save_pretrained(peft_model_id)
            self.tokenizer.save_pretrained(peft_model_id)

            # Update adapter config for O_LoRA compatibility
            adapter_config_path = os.path.join(peft_model_id, 'adapter_config.json')
            if os.path.exists(adapter_config_path):
                with open(adapter_config_path, 'r') as f:
                    adapter_config = json.load(f)

                # Key compatibility setting
                adapter_config['r_sum'] = 0

                # Add TreeLoRA metadata
                adapter_config['treelora_metadata'] = {
                    'task_id': round,
                    'lamda_1': self.lamda_1,
                    'lamda_2': self.lamda_2,
                    'reg': self.args.reg,
                    'use_opl': getattr(self.args, 'use_opl', True),
                }

                with open(adapter_config_path, 'w') as f:
                    json.dump(adapter_config, f, indent=2)

            print_rank_0(
                f'[Tree_LoRA] Model saved to {peft_model_id}',
                self.args.global_rank
            )


class Tree_LoRA_OPL(Tree_LoRA):
    """
    Extended TreeLoRA with enhanced OPL capabilities.

    This variant provides additional OPL modes:
    1. Loss-based OPL (default): Adds OPL as a loss term
    2. Gradient projection: Directly modifies gradients to be orthogonal
    3. Hybrid: Combines both approaches
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.opl_mode = getattr(self.args, 'opl_mode', 'loss')  # 'loss', 'projection', 'hybrid'

        print_rank_0(
            f"[Tree_LoRA_OPL] OPL mode: {self.opl_mode}",
            self.args.global_rank
        )

    def train_one_task(self, task: str, task_id: int, epochs: int):
        """Training with configurable OPL mode."""
        if self.opl_mode == 'projection':
            self.use_gradient_projection = True
            self.kd_lora_tree.use_opl = False
        elif self.opl_mode == 'hybrid':
            self.use_gradient_projection = True
            self.kd_lora_tree.use_opl = True
        else:  # 'loss'
            self.use_gradient_projection = False
            self.kd_lora_tree.use_opl = True

        super().train_one_task(task, task_id, epochs)