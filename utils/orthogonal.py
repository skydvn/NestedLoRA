# gpm_utils.py - Gradient Projection Memory for Orthogonal LoRA Training
# Ensures each LoRA learns features orthogonal to previously trained LoRAs

import torch
import torch.nn as nn
from typing import List, Dict, Optional
import numpy as np


# ============================================================================
# Gradient Projection Memory (GPM)
# ============================================================================

class GradientProjectionMemory:
    """
    Gradient Projection Memory for Continual Learning

    Used to ensure that when training LoRA_i, gradients don't interfere
    with the learned representations of previously trained LoRAs.

    Key Idea:
    - After training LoRA_i, we compute and store its feature space
    - When training LoRA_{i+1}, we project gradients to be orthogonal
      to the feature spaces of all previous LoRAs

    This ensures: LoRA_1 ⊥ LoRA_2 ⊥ LoRA_3
    """

    def __init__(
            self,
            threshold: float = 0.95,  # Variance threshold for SVD
            memory_strength: float = 1.0,  # How strongly to enforce orthogonality
            device: str = 'cuda'
    ):
        """
        Args:
            threshold: Cumulative variance threshold for selecting principal components
            memory_strength: Multiplier for projection (1.0 = full projection)
            device: Device to store projection matrices
        """
        self.threshold = threshold
        self.memory_strength = memory_strength
        self.device = device

        # Storage for projection matrices
        # Key: (lora_idx, param_name), Value: projection matrix
        self.projection_matrices: Dict[str, torch.Tensor] = {}

        # Storage for feature importance per LoRA
        self.feature_importance: Dict[int, Dict[str, torch.Tensor]] = {}

        # Track which LoRAs have been "locked" (training completed)
        self.locked_loras: List[int] = []

        print(f"Initialized GPM with threshold={threshold}, strength={memory_strength}")

    def get_feature_key(self, lora_idx: int, param_name: str) -> str:
        """Generate unique key for parameter"""
        return f"lora{lora_idx}_{param_name}"

    def compute_feature_importance(
            self,
            model,
            dataloader,
            lora_idx: int,
            num_batches: int = 100
    ):
        """
        Compute feature importance for a trained LoRA using gradient information

        This captures which dimensions of the parameter space are important
        for the current LoRA's learned task.

        Args:
            model: The model with trained LoRA
            dataloader: Data loader to compute gradients
            lora_idx: Index of the LoRA we just finished training
            num_batches: Number of batches to use for computing importance
        """

        print(f"\n{'=' * 70}")
        print(f"Computing Feature Importance for LoRA {lora_idx + 1}")
        print(f"{'=' * 70}")

        model.eval()
        device = next(model.parameters()).device

        # Get parameters for this LoRA
        lora_params = model.get_lora_parameters(lora_idx)

        # Initialize gradient accumulators
        grad_accumulators = {
            id(p): torch.zeros_like(p.data) for p in lora_params
        }

        # Accumulate gradients over multiple batches
        count = 0
        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)

            # Backward pass
            loss.backward()

            # Accumulate gradients
            for p in lora_params:
                if p.grad is not None:
                    grad_accumulators[id(p)] += p.grad.data.abs()

            # Zero gradients
            model.zero_grad()

            count += 1

        # Average the gradients
        for p in lora_params:
            grad_accumulators[id(p)] /= count

        # Store feature importance
        self.feature_importance[lora_idx] = {}

        # Get parameter names and store importance
        param_names = []
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # This is a LoRA layer
                if lora_idx < len(module.lora_A):
                    A_param = module.lora_A[lora_idx]
                    B_param = module.lora_B[lora_idx]

                    A_key = f"{name}.lora_A"
                    B_key = f"{name}.lora_B"

                    if id(A_param) in grad_accumulators:
                        self.feature_importance[lora_idx][A_key] = grad_accumulators[id(A_param)]
                        param_names.append(A_key)

                    if id(B_param) in grad_accumulators:
                        self.feature_importance[lora_idx][B_key] = grad_accumulators[id(B_param)]
                        param_names.append(B_key)

        print(f"  ✓ Computed importance for {len(param_names)} parameters")
        print(f"  ✓ LoRA {lora_idx + 1} is now locked")

        # Mark this LoRA as locked
        if lora_idx not in self.locked_loras:
            self.locked_loras.append(lora_idx)

    def compute_projection_matrices(
            self,
            model,
            dataloader,
            lora_idx: int,
            num_batches: int = 50
    ):
        """
        Compute projection matrices to preserve the feature space of a LoRA

        Uses SVD to find the principal gradient directions for this LoRA,
        which represent the important feature space.

        Args:
            model: Model with trained LoRA
            dataloader: Data loader
            lora_idx: Index of LoRA to compute projections for
            num_batches: Number of batches for computing gradient space
        """

        print(f"\nComputing Projection Matrices for LoRA {lora_idx + 1}...")

        model.eval()
        device = next(model.parameters()).device

        # Get parameters for this LoRA
        lora_params = model.get_lora_parameters(lora_idx)

        # Collect gradients over multiple batches
        gradient_history = {id(p): [] for p in lora_params}

        for batch_idx, (images, labels) in enumerate(dataloader):
            if batch_idx >= num_batches:
                break

            images, labels = images.to(device), labels.to(device)

            # Forward and backward
            outputs = model(images)
            loss = nn.functional.cross_entropy(outputs, labels)
            loss.backward()

            # Store gradients
            for p in lora_params:
                if p.grad is not None:
                    gradient_history[id(p)].append(p.grad.data.clone().flatten())

            model.zero_grad()

        # Compute SVD for each parameter to get projection matrix
        projection_count = 0

        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if lora_idx < len(module.lora_A):
                    for matrix_name, param in [
                        ('lora_A', module.lora_A[lora_idx]),
                        ('lora_B', module.lora_B[lora_idx])
                    ]:
                        if id(param) in gradient_history and len(gradient_history[id(param)]) > 0:
                            # Stack gradients into matrix
                            grad_matrix = torch.stack(gradient_history[id(param)], dim=0)

                            # Perform SVD: G = U @ S @ V^T
                            try:
                                U, S, V = torch.svd(grad_matrix)

                                # Select components based on variance threshold
                                variance = S ** 2
                                cumulative_variance = torch.cumsum(variance, dim=0) / variance.sum()
                                num_components = (cumulative_variance < self.threshold).sum().item() + 1
                                num_components = min(num_components, len(S))

                                # Projection matrix: P = V[:, :num_components] @ V[:, :num_components]^T
                                # This projects onto the important gradient subspace
                                V_important = V[:, :num_components]
                                projection_matrix = V_important @ V_important.T

                                # Store projection matrix
                                key = self.get_feature_key(lora_idx, f"{name}.{matrix_name}")
                                self.projection_matrices[key] = projection_matrix.to(self.device)

                                projection_count += 1

                            except RuntimeError as e:
                                print(f"  Warning: SVD failed for {name}.{matrix_name}: {e}")

        print(f"  ✓ Computed {projection_count} projection matrices")
        print(f"  ✓ Variance threshold: {self.threshold}")

    def project_gradients(
            self,
            model,
            current_lora_idx: int
    ):
        """
        Project gradients of current LoRA to be orthogonal to locked LoRAs

        For each parameter in current LoRA:
            g_projected = g - Σ(P_i @ g) for all locked LoRAs i

        This ensures the current LoRA doesn't interfere with previously learned features.

        Args:
            model: Model being trained
            current_lora_idx: Index of LoRA currently being trained
        """

        if len(self.locked_loras) == 0:
            # No locked LoRAs yet, no projection needed
            return

        # Get parameters for current LoRA
        current_params = model.get_lora_parameters(current_lora_idx)

        # Project gradients for each parameter
        for name, module in model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                if current_lora_idx < len(module.lora_A):
                    for matrix_name, param in [
                        ('lora_A', module.lora_A[current_lora_idx]),
                        ('lora_B', module.lora_B[current_lora_idx])
                    ]:
                        if param.grad is not None:
                            # Flatten gradient
                            original_shape = param.grad.shape
                            grad_flat = param.grad.data.flatten()

                            # Project away from all locked LoRAs
                            for locked_idx in self.locked_loras:
                                key = self.get_feature_key(locked_idx, f"{name}.{matrix_name}")

                                if key in self.projection_matrices:
                                    P = self.projection_matrices[key]

                                    # Project: g = g - strength * (P @ g)
                                    # This removes the component in the locked LoRA's space
                                    projected_component = P @ grad_flat
                                    grad_flat = grad_flat - self.memory_strength * projected_component

                            # Reshape back and update gradient
                            param.grad.data = grad_flat.reshape(original_shape)

    def get_orthogonality_metrics(self, model):
        """
        GPM version: compute orthogonality between locked LoRAs
        Using direct weight comparison: W = A @ B
        """
        if len(self.locked_loras) < 2:
            return {}

        metrics = {}

        # Collect weight vectors for each locked LoRA
        lora_weights = {}

        for lora_idx in sorted(self.locked_loras):
            weights = []

            for name, module in model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    W = get_lora_weight_matrix(module, lora_idx)
                    if W is not None:
                        weights.append(W.flatten())

            if weights:
                lora_weights[lora_idx] = torch.cat(weights, dim=0)

        # Compute pairwise orthogonality
        locked_list = sorted(self.locked_loras)

        for i in range(len(locked_list)):
            for j in range(i + 1, len(locked_list)):
                lora_i = locked_list[i]
                lora_j = locked_list[j]

                if lora_i in lora_weights and lora_j in lora_weights:
                    vec_i = lora_weights[lora_i]
                    vec_j = lora_weights[lora_j]

                    # Cosine similarity
                    dot = (vec_i @ vec_j).item()
                    norm_i = torch.norm(vec_i).item()
                    norm_j = torch.norm(vec_j).item()

                    if norm_i > 0 and norm_j > 0:
                        cosine_sim = dot / (norm_i * norm_j)
                        orthogonality = 1.0 - abs(cosine_sim)

                        metrics[f"orthogonality_L{lora_i + 1}_L{lora_j + 1}"] = orthogonality

        return metrics

    def save(self, path: str):
        """Save GPM state"""
        state = {
            'projection_matrices': {k: v.cpu() for k, v in self.projection_matrices.items()},
            'feature_importance': {
                k: {k2: v2.cpu() for k2, v2 in v.items()}
                for k, v in self.feature_importance.items()
            },
            'locked_loras': self.locked_loras,
            'threshold': self.threshold,
            'memory_strength': self.memory_strength
        }
        torch.save(state, path)
        print(f"  ✓ Saved GPM state to {path}")

    def load(self, path: str):
        """Load GPM state"""
        state = torch.load(path, map_location=self.device)
        self.projection_matrices = {k: v.to(self.device) for k, v in state['projection_matrices'].items()}
        self.feature_importance = {
            k: {k2: v2.to(self.device) for k2, v2 in v.items()}
            for k, v in state['feature_importance'].items()
        }
        self.locked_loras = state['locked_loras']
        self.threshold = state['threshold']
        self.memory_strength = state['memory_strength']
        print(f"  ✓ Loaded GPM state from {path}")


# ============================================================================
# GPM Hook for Automatic Gradient Projection
# ============================================================================

class GPMHook:
    """
    Automatic gradient projection hook

    Registers backward hooks on LoRA parameters to automatically
    project gradients during training.
    """

    def __init__(self, gpm: GradientProjectionMemory, model, current_lora_idx: int):
        """
        Args:
            gpm: GPM instance
            model: Model with LoRA
            current_lora_idx: Index of LoRA being trained
        """
        self.gpm = gpm
        self.model = model
        self.current_lora_idx = current_lora_idx
        self.hooks = []

    def register_hooks(self):
        """Register backward hooks on current LoRA parameters"""

        if len(self.gpm.locked_loras) == 0:
            # No projection needed
            return

        print(f"\n  📌 Registering GPM hooks for LoRA {self.current_lora_idx + 1}")
        print(f"     Projecting away from locked LoRAs: {[i + 1 for i in self.gpm.locked_loras]}")

        # Get current LoRA parameters
        current_params = self.model.get_lora_parameters(self.current_lora_idx)

        def projection_hook(grad):
            """Hook function that projects gradients"""
            if grad is None:
                return grad

            # Project gradient
            original_shape = grad.shape
            grad_flat = grad.flatten()

            # Find which parameter this gradient belongs to
            for name, module in self.model.named_modules():
                if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                    if self.current_lora_idx < len(module.lora_A):
                        for matrix_name, param in [
                            ('lora_A', module.lora_A[self.current_lora_idx]),
                            ('lora_B', module.lora_B[self.current_lora_idx])
                        ]:
                            if param.grad is grad:
                                # Found the parameter, project it
                                for locked_idx in self.gpm.locked_loras:
                                    key = self.gpm.get_feature_key(locked_idx, f"{name}.{matrix_name}")

                                    if key in self.gpm.projection_matrices:
                                        P = self.gpm.projection_matrices[key]
                                        projected_component = P @ grad_flat
                                        grad_flat = grad_flat - self.gpm.memory_strength * projected_component

                                return grad_flat.reshape(original_shape)

            return grad

        # Register hooks
        for param in current_params:
            hook = param.register_hook(projection_hook)
            self.hooks.append(hook)

    def remove_hooks(self):
        """Remove all registered hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []


# ============================================================================
# Helper Functions
# ============================================================================

def get_lora_weight_matrix(module, lora_idx):
    """Get W = A @ B for a specific LoRA"""
    if lora_idx >= len(module.lora_A):
        return None

    A = module.lora_A[lora_idx]
    B = module.lora_B[lora_idx]
    scaling = module.scalings[lora_idx]

    W = A @ B  # [in_features, rank] @ [rank, out_features] = [in_features, out_features]
    W = W * scaling
    return W

def analyze_lora_orthogonality(model, gpm: Optional['GradientProjectionMemory'] = None):
    """
    Analyze orthogonality by comparing W = A @ B matrices directly

    Much simpler than random input approach!
    """
    print(f"\n{'=' * 70}")
    print("LoRA Orthogonality Analysis (Direct Weight Comparison)")
    print(f"{'=' * 70}")

    num_loras = model.num_loras

    print(f"\nModel Info:")
    print(f"  Number of LoRAs: {num_loras}")
    print(f"  Ranks: {model.ranks}")

    # Collect weight matrices for each LoRA
    lora_weights = [[] for _ in range(num_loras)]

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            for lora_idx in range(num_loras):
                W = get_lora_weight_matrix(module, lora_idx)
                if W is not None:
                    lora_weights[lora_idx].append(W.flatten())

    # Concatenate all weights for each LoRA
    lora_vectors = []
    for lora_idx in range(num_loras):
        if lora_weights[lora_idx]:
            vec = torch.cat(lora_weights[lora_idx], dim=0)
            lora_vectors.append(vec)
            print(f"  LoRA{lora_idx + 1} (rank={model.ranks[lora_idx]:2d}): "
                  f"total weight vector size = {vec.shape[0]:,}")

    print()

    # Compute pairwise orthogonality
    orthogonality_scores = []

    for i in range(num_loras):
        for j in range(i + 1, num_loras):
            if i >= len(lora_vectors) or j >= len(lora_vectors):
                continue

            vec_i = lora_vectors[i]
            vec_j = lora_vectors[j]

            # Cosine similarity
            dot = (vec_i @ vec_j).item()
            norm_i = torch.norm(vec_i).item()
            norm_j = torch.norm(vec_j).item()

            if norm_i > 0 and norm_j > 0:
                cosine_sim = dot / (norm_i * norm_j)
                orthogonality = 1.0 - abs(cosine_sim)
                orthogonality_scores.append(orthogonality)

                # Visual indicator
                if orthogonality >= 0.90:
                    indicator = "✅ Excellent"
                elif orthogonality >= 0.80:
                    indicator = "✓  Good"
                elif orthogonality >= 0.70:
                    indicator = "⚠️  Fair"
                else:
                    indicator = "❌ Poor"

                print(f"  LoRA{i + 1}(r={model.ranks[i]:2d}) ⊥ LoRA{j + 1}(r={model.ranks[j]:2d}): "
                      f"{orthogonality:.4f} (cosine_sim: {cosine_sim:+.4f}) {indicator}")

    # Summary
    if orthogonality_scores:
        avg_orth = np.mean(orthogonality_scores)
        print(f"\nSummary:")
        print(f"  Average orthogonality: {avg_orth:.4f}")

        if avg_orth >= 0.90:
            print(f"  Overall quality: ✅ EXCELLENT")
        elif avg_orth >= 0.80:
            print(f"  Overall quality: ✓  GOOD")
        elif avg_orth >= 0.70:
            print(f"  Overall quality: ⚠️  FAIR")
        else:
            print(f"  Overall quality: ❌ POOR")

    print(f"{'=' * 70}\n")

def compute_effective_rank_similarity(model, lora_idx_i, lora_idx_j, num_samples=100):
    """
    Compute similarity between two LoRAs by comparing their effective transformations

    This method doesn't require parameters to have the same shape.
    Instead, it measures how similarly the LoRAs transform random inputs.

    Args:
        model: Model with LoRAs
        lora_idx_i: Index of first LoRA
        lora_idx_j: Index of second LoRA
        num_samples: Number of random samples to use

    Returns:
        cosine_similarity: Similarity between transformation effects
    """
    device = next(model.parameters()).device

    # Generate random inputs
    input_dim = model.config.embed_dimk
    random_inputs = torch.randn(num_samples, input_dim).to(device)

    # Collect outputs from both LoRAs
    outputs_i = []
    outputs_j = []

    for name, module in model.named_modules():
        if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
            if lora_idx_i < len(module.lora_A) and lora_idx_j < len(module.lora_B):
                # LoRA i transformation
                A_i = module.lora_A[lora_idx_i]
                B_i = module.lora_B[lora_idx_i]
                scaling_i = module.scalings[lora_idx_i]

                # LoRA j transformation
                A_j = module.lora_A[lora_idx_j]
                B_j = module.lora_B[lora_idx_j]
                scaling_j = module.scalings[lora_idx_j]

                with torch.no_grad():
                    # Handle different input dimensions
                    if A_i.shape[0] == input_dim and A_j.shape[0] == input_dim:
                        # Transform with LoRA i
                        temp_i = random_inputs @ A_i
                        output_i = temp_i @ B_i * scaling_i
                        outputs_i.append(output_i.flatten())

                        # Transform with LoRA j
                        temp_j = random_inputs @ A_j
                        output_j = temp_j @ B_j * scaling_j
                        outputs_j.append(output_j.flatten())

    if not outputs_i or not outputs_j:
        return 0.0

    # Concatenate and compute similarity
    vec_i = torch.cat(outputs_i, dim=0)
    vec_j = torch.cat(outputs_j, dim=0)

    # Cosine similarity
    dot_product = (vec_i @ vec_j).item()
    norm_i = (vec_i @ vec_i).item()
    norm_j = (vec_j @ vec_j).item()

    if norm_i > 0 and norm_j > 0:
        cosine_sim = dot_product / (np.sqrt(norm_i) * np.sqrt(norm_j))
        return cosine_sim

    return 0.0


def detailed_orthogonality_report(model, gpm=None, save_path=None):
    """
    Generate a detailed orthogonality report

    Args:
        model: Model with LoRAs
        gpm: Optional GPM instance
        save_path: Optional path to save report
    """

    report = []
    report.append("=" * 70)
    report.append("DETAILED LORA ORTHOGONALITY REPORT")
    report.append("=" * 70)
    report.append("")

    # Model info
    report.append("Model Configuration:")
    report.append(f"  Number of LoRAs: {model.num_loras}")
    report.append(f"  Ranks: {model.ranks}")
    report.append(f"  Alphas: {model.lora_alphas}")
    report.append("")

    # Compute all pairwise similarities
    report.append("Pairwise LoRA Similarities:")
    report.append("-" * 70)

    num_loras = model.num_loras
    similarities = np.zeros((num_loras, num_loras))

    for i in range(num_loras):
        for j in range(i + 1, num_loras):
            sim = compute_effective_rank_similarity(model, i, j)
            similarities[i, j] = sim
            similarities[j, i] = sim

            orthogonality = 1.0 - abs(sim)

            report.append(f"  LoRA{i + 1}(rank={model.ranks[i]:2d}) ⊥ "
                          f"LoRA{j + 1}(rank={model.ranks[j]:2d}): "
                          f"orthogonality={orthogonality:.4f}, "
                          f"similarity={sim:+.4f}")

    report.append("")

    # Overall statistics
    upper_triangle = []
    for i in range(num_loras):
        for j in range(i + 1, num_loras):
            upper_triangle.append(abs(similarities[i, j]))

    if upper_triangle:
        avg_sim = np.mean(upper_triangle)
        max_sim = np.max(upper_triangle)
        min_sim = np.min(upper_triangle)

        report.append("Overall Statistics:")
        report.append(f"  Average |similarity|: {avg_sim:.4f}")
        report.append(f"  Maximum |similarity|: {max_sim:.4f}")
        report.append(f"  Minimum |similarity|: {min_sim:.4f}")
        report.append(f"  Average orthogonality: {1.0 - avg_sim:.4f}")
        report.append("")

    # GPM status
    if gpm is not None:
        report.append("GPM Status:")
        report.append(f"  Locked LoRAs: {[i + 1 for i in gpm.locked_loras]}")
        report.append(f"  Projection matrices: {len(gpm.projection_matrices)}")
        report.append("")

    # Quality assessment
    report.append("Quality Assessment:")
    if upper_triangle:
        avg_orth = 1.0 - avg_sim
        if avg_orth >= 0.90:
            quality = "EXCELLENT ✅"
        elif avg_orth >= 0.80:
            quality = "GOOD ✓"
        elif avg_orth >= 0.70:
            quality = "FAIR ⚠️"
        else:
            quality = "POOR ❌"

        report.append(f"  Overall Quality: {quality}")
        report.append(f"  Average Orthogonality: {avg_orth:.4f}")

    report.append("=" * 70)

    # Print report
    full_report = "\n".join(report)
    print(full_report)

    # Save if requested
    if save_path is not None:
        with open(save_path, 'w') as f:
            f.write(full_report)
        print(f"\n✓ Report saved to {save_path}")

    return similarities


def print_gpm_status(gpm: GradientProjectionMemory):
    """Print current GPM status"""

    print(f"\n{'=' * 70}")
    print("GPM Status")
    print(f"{'=' * 70}")
    print(f"  Locked LoRAs: {[i + 1 for i in gpm.locked_loras]}")
    print(f"  Projection matrices stored: {len(gpm.projection_matrices)}")
    print(f"  Variance threshold: {gpm.threshold}")
    print(f"  Memory strength: {gpm.memory_strength}")
    print(f"{'=' * 70}\n")


if __name__ == '__main__':
    print("Orthogonality analysis utilities loaded successfully!")
    print("\nAvailable functions:")
    print("  • analyze_lora_orthogonality(model, gpm)")
    print("  • compute_effective_rank_similarity(model, i, j)")
    print("  • detailed_orthogonality_report(model, gpm, save_path)")
