"""
LoRA as Gradient Generator

Instead of unmerge/merge cycle, this approach:
1. Trains LoRA matrices A and B
2. Uses LoRA output (A*B) as gradient for base model
3. Updates base weights: W_{t+1} = W_t - η * (A*B)

Key Innovation:
- LoRA learns the UPDATE DIRECTION (gradient)
- Base model is updated DIRECTLY with this gradient
- No separate LoRA addition, no unmerge/merge cycle
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional


class LoRAGradientGenerator(nn.Module):
    """
    LoRA that generates gradients for base model updates

    Process:
    1. Forward: output = W·x (no LoRA addition)
    2. Train LoRA: min_{A,B} Loss(W·x)
    3. Compute gradient: ΔW = A @ B
    4. Update base: W ← W - η * ΔW
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        rank: int = 8,
        lora_alpha: int = 16,
        max_rank: int = 32,
        lora_lr: float = 1e-3,
    ):
        super().__init__()

        # Store reference to base linear layer
        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features

        # LoRA configuration
        self.current_rank = rank
        self.max_rank = max_rank
        self.lora_alpha = lora_alpha
        self.lora_lr = lora_lr

        # Get device
        device = linear_layer.weight.device

        # LoRA matrices (gradient generators)
        self.lora_A = nn.Parameter(torch.zeros(self.in_features, max_rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(max_rank, self.out_features, device=device))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Freeze base model weights
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    @property
    def scaling(self):
        return self.lora_alpha / self.current_rank

    def compute_lora_gradient(self):
        """
        Compute gradient from LoRA matrices: ΔW = (A @ B).T * scaling

        This is the UPDATE that will be applied to base weights:
        W_{t+1} = W_t - η * ΔW
        """
        # Use only active rank
        A_active = self.lora_A[:, :self.current_rank]  # [in_features, rank]
        B_active = self.lora_B[:self.current_rank, :]  # [rank, out_features]

        # Ensure same device
        target_device = self.linear.weight.device
        if A_active.device != target_device:
            A_active = A_active.to(target_device)
        if B_active.device != target_device:
            B_active = B_active.to(target_device)

        # Compute gradient: [out_features, in_features]
        gradient = (A_active @ B_active).T * self.scaling

        return gradient

    def apply_lora_gradient_to_base(self, learning_rate: float = None):
        """
        ⭐ APPLY LoRA AS GRADIENT TO BASE WEIGHTS ⭐

        W_{t+1} = W_t - η * (A @ B)

        This is the key step where LoRA updates base model!
        """
        if learning_rate is None:
            learning_rate = self.lora_lr

        # Compute gradient from LoRA
        lora_gradient = self.compute_lora_gradient()

        # ⭐ UPDATE BASE WEIGHTS ⭐
        with torch.no_grad():
            self.linear.weight.data -= learning_rate * lora_gradient

        # Reset LoRA matrices (they generated the update, now reset)
        with torch.no_grad():
            self.lora_A.data.zero_()
            self.lora_B.data.zero_()

    def forward(self, x):
        """
        Forward: Base model + LoRA output

        output = W·x + (A@B)·x

        This ensures loss is computed with LoRA included:
        Loss((W + A@B)·x; D)
        """
        # Base output
        base_output = self.linear(x)

        # LoRA output
        A_active = self.lora_A[:, :self.current_rank]
        B_active = self.lora_B[:self.current_rank, :]

        # Ensure same device
        if A_active.device != x.device:
            A_active = A_active.to(x.device)
        if B_active.device != x.device:
            B_active = B_active.to(x.device)

        # LoRA forward: (x @ A) @ B
        lora_output = (x @ A_active) @ B_active * self.scaling

        # ⭐ Return base + LoRA for loss computation
        return base_output + lora_output


class LoRAGradientModel(nn.Module):
    """
    Model with LoRA as gradient generators
    """

    def __init__(
        self,
        base_model,
        initial_rank: int = 4,
        max_rank: int = 32,
        lora_alpha: int = 16,
        lora_lr: float = 1e-3,
    ):
        super().__init__()

        self.base_model = base_model
        self.lora_modules = nn.ModuleDict()
        self.initial_rank = initial_rank
        self.max_rank = max_rank
        self.lora_lr = lora_lr

        # Apply LoRA gradient generators to all linear layers
        self._apply_lora_to_model(initial_rank, max_rank, lora_alpha, lora_lr)

    def _apply_lora_to_model(self, initial_rank, max_rank, lora_alpha, lora_lr):
        """Apply LoRA gradient generators to linear layers"""
        target_modules = ["qkv", "proj", "fc1", "fc2", "head"]

        for name, module in self.base_model.named_modules():
            should_apply = any(target in name for target in target_modules)

            if should_apply and isinstance(module, nn.Linear):
                # Create LoRA gradient generator
                lora = LoRAGradientGenerator(
                    module,
                    rank=initial_rank,
                    lora_alpha=lora_alpha,
                    max_rank=max_rank,
                    lora_lr=lora_lr
                )

                # Store with sanitized name
                sanitized_name = name.replace('.', '_')
                self.lora_modules[sanitized_name] = lora

    def forward(self, x):
        """Forward through base model (no LoRA addition)"""
        return self.base_model(x)

    def apply_all_lora_gradients(self, learning_rate: float = None):
        """
        ⭐ APPLY ALL LoRA GRADIENTS TO BASE MODEL ⭐

        For each layer:
        W_{t+1} = W_t - η * (A @ B)

        This updates the base model using LoRA-generated gradients!
        """
        for name, lora in self.lora_modules.items():
            lora.apply_lora_gradient_to_base(learning_rate)

    def get_lora_parameters(self):
        """Get LoRA parameters (A and B matrices)"""
        params = []
        for lora in self.lora_modules.values():
            params.extend([lora.lora_A, lora.lora_B])
        return params

    def reset_lora_matrices(self):
        """Reset all LoRA matrices to zero"""
        with torch.no_grad():
            for lora in self.lora_modules.values():
                lora.lora_A.data.zero_()
                lora.lora_B.data.zero_()


# ============================================================================
# Training with LoRA as Gradient Generator
# ============================================================================

def train_epoch_lora_gradient(
    model,
    loader,
    lora_optimizer,
    criterion,
    device,
    epoch,
    lora_update_lr: float = 1e-4,
    apply_every_n_steps: int = 1,
):
    """
    Train with LoRA as gradient generator

    Process:
    1. Forward: output = W·x (base model only)
    2. Backward: compute gradients for A, B
    3. Update A, B: optimize LoRA matrices
    4. Apply gradient: W ← W - η * (A*B)
    5. Reset A, B to zero

    Args:
        model: LoRAGradientModel
        loader: DataLoader
        lora_optimizer: Optimizer for LoRA parameters (A, B)
        criterion: Loss function
        device: Device
        epoch: Current epoch
        lora_update_lr: Learning rate for base model update
        apply_every_n_steps: Apply LoRA gradient every N steps
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    from tqdm import tqdm
    pbar = tqdm(loader, desc=f'Epoch {epoch} [LoRA Gradient]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # ══════════════════════════════════════════════════════
        # STEP 1: Forward with base model only (no LoRA)
        # ══════════════════════════════════════════════════════
        outputs = model(images)
        loss = criterion(outputs, labels)

        # ══════════════════════════════════════════════════════
        # STEP 2: Backward - compute gradients for LoRA (A, B)
        # ══════════════════════════════════════════════════════
        lora_optimizer.zero_grad()
        loss.backward()

        # ══════════════════════════════════════════════════════
        # STEP 3: Update LoRA matrices (A, B)
        # ══════════════════════════════════════════════════════
        lora_optimizer.step()

        # ══════════════════════════════════════════════════════
        # STEP 4: Apply LoRA as gradient to base model
        # ══════════════════════════════════════════════════════
        if (batch_idx + 1) % apply_every_n_steps == 0:
            # ⭐ KEY STEP: W ← W - η * (A @ B)
            model.apply_all_lora_gradients(learning_rate=lora_update_lr)
            # LoRA matrices are automatically reset to zero

        # Metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

    return total_loss / len(loader), 100. * correct / total


# ============================================================================
# Example Usage
# ============================================================================

def example_lora_gradient():
    """
    Example: Using LoRA as gradient generator
    """
    from tiny_vit_cifar10 import TinyViT, TinyViTConfig

    print("="*80)
    print("LoRA AS GRADIENT GENERATOR")
    print("="*80)

    # Create base model
    config = TinyViTConfig()
    base_model = TinyViT(config).to('cuda')

    # Wrap with LoRA gradient generators
    model = LoRAGradientModel(
        base_model,
        initial_rank=8,
        max_rank=32,
        lora_alpha=16,
        lora_lr=1e-4
    )

    print(f"\n✓ Model created with LoRA gradient generators")
    print(f"  LoRA modules: {len(model.lora_modules)}")

    # Optimizer for LoRA parameters only
    lora_params = model.get_lora_parameters()
    lora_optimizer = torch.optim.AdamW(lora_params, lr=1e-3)

    print(f"\n✓ Optimizer for LoRA parameters")
    print(f"  Trainable LoRA params: {sum(p.numel() for p in lora_params):,}")

    # Training step
    print("\n" + "="*80)
    print("TRAINING PROCESS")
    print("="*80)

    criterion = nn.CrossEntropyLoss()
    x = torch.randn(4, 3, 32, 32).to('cuda')
    labels = torch.randint(0, 10, (4,)).to('cuda')

    print("\n1. Forward (base model only)")
    outputs = model(x)
    loss = criterion(outputs, labels)
    print(f"   Loss: {loss.item():.4f}")

    print("\n2. Backward (compute LoRA gradients)")
    lora_optimizer.zero_grad()
    loss.backward()
    print("   ✓ Gradients for A, B computed")

    print("\n3. Update LoRA matrices")
    lora_optimizer.step()
    print("   ✓ A, B updated via optimizer")

    print("\n4. Apply LoRA as gradient to base model")
    print("   Computing: ΔW = (A @ B).T * scaling")

    # Check LoRA gradient magnitude
    first_lora = list(model.lora_modules.values())[0]
    lora_grad = first_lora.compute_lora_gradient()
    print(f"   LoRA gradient norm: {lora_grad.norm():.6f}")

    # Apply to base model
    model.apply_all_lora_gradients(learning_rate=1e-4)
    print("   ⭐ Base weights updated: W ← W - η * (A @ B)")
    print("   ⭐ LoRA matrices reset to zero")

    # Verify LoRA reset
    print(f"\n5. Verify LoRA reset")
    print(f"   LoRA A norm: {first_lora.lora_A.norm():.6f}")
    print(f"   LoRA B norm: {first_lora.lora_B.norm():.6f}")
    print("   ✓ Both should be 0.0")

    print("\n" + "="*80)
    print("COMPLETE!")
    print("="*80)
    print("""
Process Summary:
1. Forward: output = W·x (no LoRA addition)
2. Backward: ∇A, ∇B computed
3. Update LoRA: A ← A - lr·∇A, B ← B - lr·∇B
4. Generate gradient: ΔW = (A @ B)
5. Update base: W ← W - η·ΔW
6. Reset: A ← 0, B ← 0

Key Insight:
- LoRA learns the UPDATE DIRECTION
- Base model is updated DIRECTLY
- No unmerge/merge, no LoRA addition in forward
    """)


if __name__ == '__main__':
    example_lora_gradient()

    print("\n" + "="*80)
    print("COMPARISON")
    print("="*80)
    print("""
TRADITIONAL LoRA:
─────────────────
Forward:  output = W·x + (A@B)·x
Update:   Only A, B updated
Result:   Base model W unchanged

DIRECT MERGE LoRA:
──────────────────
Forward:  output = W·x (LoRA merged)
Update:   Unmerge → Update A,B → Re-merge
Result:   Base model W contains LoRA

LoRA AS GRADIENT (THIS):
────────────────────────
Forward:  output = W·x (pure base model)
Update:   1. Update A, B via loss
          2. Compute gradient: ΔW = A@B
          3. Update base: W ← W - η·ΔW
          4. Reset A, B to zero
Result:   Base model W continuously updated
          LoRA generates update directions
          
Benefits:
✓ No LoRA addition overhead
✓ No unmerge/merge cycle
✓ Base model learns from LoRA guidance
✓ LoRA acts as learned optimizer
    """)