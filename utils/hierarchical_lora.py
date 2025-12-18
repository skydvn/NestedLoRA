"""
Hierarchical LoRA - Normal Fine-Tuning

W = W_0 + LoRA_1 + LoRA_2 + ... + LoRA_L

Key differences from gradient generator:
- W_0 stays FROZEN (never updated)
- Each LoRA is a permanent adaptation
- Different frequencies → different update rates
- Higher frequency → higher rank
- Forward always includes all LoRAs

Training:
1. Forward: y = (W_0 + ΔW_1 + ΔW_2 + ΔW_3)·x
2. Optimize: Update active LoRAs based on their frequency
3. NO reset, NO merging to base
"""

import torch
import torch.nn as nn
import math
from typing import List, Dict, Optional


class HierarchicalLoRALayer(nn.Module):
    """
    Single LoRA layer in hierarchy

    Normal fine-tuning: LoRA stays as additive component
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        lora_alpha: int = 16,
        update_frequency: int = 1,
        device: Optional[torch.device] = None,
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.update_frequency = update_frequency

        # LoRA matrices
        if device is None:
            device = torch.device('cpu')

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank, device=device))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features, device=device))

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    @property
    def scaling(self):
        return self.lora_alpha / self.rank

    def should_update(self, global_batch_idx: int) -> bool:
        """Check if this LoRA should update at this batch"""
        return (global_batch_idx % self.update_frequency) == 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward: LoRA contribution

        output = (x @ A) @ B * scaling
        """
        lora_output = (x @ self.lora_A) @ self.lora_B * self.scaling
        return lora_output


class HierarchicalLoRAModule(nn.Module):
    """
    Hierarchical LoRA for a single linear layer

    Normal fine-tuning:
      output = base·x + LoRA_1·x + LoRA_2·x + ... + LoRA_L·x

    All LoRAs are KEPT (not merged into base)
    """

    def __init__(
        self,
        linear_layer: nn.Linear,
        ranks: List[int],
        frequencies: List[int],
        lora_alpha: int = 16,
    ):
        super().__init__()

        self.linear = linear_layer
        self.in_features = linear_layer.in_features
        self.out_features = linear_layer.out_features
        self.num_levels = len(ranks)

        assert len(ranks) == len(frequencies), "Must have same number of ranks and frequencies"

        # Create hierarchy of LoRA layers
        self.lora_layers = nn.ModuleList()
        device = linear_layer.weight.device

        for level, (rank, freq) in enumerate(zip(ranks, frequencies)):
            lora = HierarchicalLoRALayer(
                in_features=self.in_features,
                out_features=self.out_features,
                rank=rank,
                lora_alpha=lora_alpha,
                update_frequency=freq,
                device=device
            )
            self.lora_layers.append(lora)

        # Freeze base model (normal LoRA fine-tuning)
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward with hierarchical LoRA

        output = W_0·x + ΔW_1·x + ΔW_2·x + ... + ΔW_L·x

        All LoRAs always included!
        """
        # Base output
        output = self.linear(x)

        # Add all LoRA contributions
        for lora in self.lora_layers:
            output = output + lora(x)

        return output

    def get_active_loras(self, global_batch_idx: int) -> List[int]:
        """Get indices of LoRAs that should update this batch"""
        active = []
        for i, lora in enumerate(self.lora_layers):
            if lora.should_update(global_batch_idx):
                active.append(i)
        return active

    def get_lora_parameters(self, level: Optional[int] = None) -> List[nn.Parameter]:
        """Get parameters for specific level or all levels"""
        if level is not None:
            return [self.lora_layers[level].lora_A, self.lora_layers[level].lora_B]
        else:
            params = []
            for lora in self.lora_layers:
                params.extend([lora.lora_A, lora.lora_B])
            return params


class HierarchicalLoRAModel(nn.Module):
    """
    Model with hierarchical LoRA - Normal Fine-Tuning

    Key: LoRAs stay as additive components (NOT merged into base)
    """

    def __init__(
        self,
        base_model: nn.Module,
        ranks: List[int] = [4, 8, 16],
        frequencies: List[int] = [100, 10, 1],
        lora_alpha: int = 16,
        target_modules: List[str] = ["qkv", "proj", "fc1", "fc2"],
    ):
        super().__init__()

        self.base_model = base_model
        self.hierarchical_modules = nn.ModuleDict()
        self.ranks = ranks
        self.frequencies = frequencies
        self.num_levels = len(ranks)

        # Apply hierarchical LoRA
        self._apply_hierarchical_lora(ranks, frequencies, lora_alpha, target_modules)

        print(f"Hierarchical LoRA Model (Normal Fine-Tuning):")
        print(f"  Levels: {self.num_levels}")
        print(f"  Ranks: {ranks}")
        print(f"  Frequencies: {frequencies}")
        print(f"  Total modules: {len(self.hierarchical_modules)}")
        print(f"\n  ⭐ Base model FROZEN (LoRAs stay additive)")

    def _apply_hierarchical_lora(self, ranks, frequencies, lora_alpha, target_modules):
        """Apply hierarchical LoRA to linear layers"""
        for name, module in self.base_model.named_modules():
            should_apply = any(target in name for target in target_modules)

            if should_apply and isinstance(module, nn.Linear):
                hier_lora = HierarchicalLoRAModule(
                    module,
                    ranks=ranks,
                    frequencies=frequencies,
                    lora_alpha=lora_alpha
                )

                sanitized_name = name.replace('.', '_')
                self.hierarchical_modules[sanitized_name] = hier_lora

    def forward(self, x):
        """
        Forward through model with hierarchical LoRA

        All LoRAs always included in forward pass
        """
        return self.base_model(x)

    def get_lora_parameters_by_level(self, level: int) -> List[nn.Parameter]:
        """Get all parameters for specific LoRA level"""
        params = []
        for hier_module in self.hierarchical_modules.values():
            params.extend(hier_module.get_lora_parameters(level))
        return params

    def get_all_lora_parameters(self) -> List[nn.Parameter]:
        """Get all LoRA parameters across all levels"""
        params = []
        for hier_module in self.hierarchical_modules.values():
            params.extend(hier_module.get_lora_parameters())
        return params

    def get_active_loras(self, global_batch_idx: int) -> Dict[str, List[int]]:
        """Get active LoRA levels for each module at this batch"""
        active = {}
        for name, hier_module in self.hierarchical_modules.items():
            active[name] = hier_module.get_active_loras(global_batch_idx)
        return active

    def get_hierarchy_info(self) -> Dict:
        """Get information about hierarchy structure"""
        total_params = sum(p.numel() for p in self.get_all_lora_parameters())

        return {
            'num_levels': self.num_levels,
            'ranks': self.ranks,
            'frequencies': self.frequencies,
            'num_modules': len(self.hierarchical_modules),
            'total_lora_params': total_params,
        }


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch_hierarchical(
    model: HierarchicalLoRAModel,
    loader,
    optimizers: Dict[int, torch.optim.Optimizer],
    criterion,
    device,
    epoch: int,
):
    """
    Train with hierarchical LoRA - Normal Fine-Tuning

    Process:
    1. Forward: y = (W_0 + ΔW_1 + ΔW_2 + ... + ΔW_L)·x
    2. Determine which LoRAs should update this batch
    3. Update only active LoRAs
    4. NO merging, NO reset

    Args:
        model: HierarchicalLoRAModel
        loader: DataLoader
        optimizers: Dict mapping level → optimizer
        criterion: Loss function
        device: Device
        epoch: Current epoch
    """
    from tqdm import tqdm

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    global_batch_idx = (epoch - 1) * len(loader)

    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)
        global_batch_idx += 1

        # ═══════════════════════════════════════════════════════
        # Determine which LoRAs are active this batch
        # ═══════════════════════════════════════════════════════
        active_levels = set()
        for name, hier_module in model.hierarchical_modules.items():
            active = hier_module.get_active_loras(global_batch_idx)
            active_levels.update(active)

        active_levels = sorted(list(active_levels))

        # ═══════════════════════════════════════════════════════
        # Forward (all LoRAs included)
        # ═══════════════════════════════════════════════════════
        outputs = model(images)  # W_0 + ΔW_1 + ΔW_2 + ΔW_3
        loss = criterion(outputs, labels)

        # ═══════════════════════════════════════════════════════
        # Backward and update only active LoRAs
        # ═══════════════════════════════════════════════════════
        # Zero gradients for active optimizers
        for level in active_levels:
            optimizers[level].zero_grad()

        # Backward
        loss.backward()

        # Update only active LoRAs
        for level in active_levels:
            optimizers[level].step()

        # ═══════════════════════════════════════════════════════
        # Metrics
        # ═══════════════════════════════════════════════════════
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        active_str = ','.join([f'L{l}' for l in active_levels]) if active_levels else 'None'
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'active': active_str
        })

    return total_loss / len(loader), 100. * correct / total


# ============================================================================
# Example Usage
# ============================================================================

def example_hierarchical_lora():
    """Example: Normal hierarchical LoRA fine-tuning"""
    from tiny_vit_cifar10 import TinyViT, TinyViTConfig

    print("="*80)
    print("HIERARCHICAL LoRA - NORMAL FINE-TUNING")
    print("="*80)

    # Configuration
    ranks = [4, 8, 16]
    frequencies = [100, 10, 1]

    print(f"\nHierarchy configuration:")
    print(f"  Level 0: rank={ranks[0]}, update every {frequencies[0]} batches")
    print(f"  Level 1: rank={ranks[1]}, update every {frequencies[1]} batches")
    print(f"  Level 2: rank={ranks[2]}, update every {frequencies[2]} batches")

    print(f"\n⭐ Key difference from gradient generator:")
    print(f"  - Base W_0 stays FROZEN")
    print(f"  - LoRAs are permanent adaptations")
    print(f"  - Forward: W_0·x + ΔW_1·x + ΔW_2·x + ΔW_3·x")
    print(f"  - NO merging, NO reset")

    # Create model
    config = TinyViTConfig()
    base_model = TinyViT(config).to('cuda')

    model = HierarchicalLoRAModel(
        base_model,
        ranks=ranks,
        frequencies=frequencies,
        lora_alpha=16
    ).to('cuda')

    print(f"\n✓ Model created")

    # Show parameter counts
    info = model.get_hierarchy_info()
    print(f"\nParameter breakdown:")
    for level in range(len(ranks)):
        params = model.get_lora_parameters_by_level(level)
        count = sum(p.numel() for p in params)
        print(f"  Level {level} (rank={ranks[level]}): {count:,} parameters")

    print(f"  Total LoRA: {info['total_lora_params']:,} parameters")

    # Simulate updates
    print(f"\n{'='*80}")
    print("UPDATE SCHEDULE: First 10 batches")
    print(f"{'='*80}")

    for batch in range(1, 11):
        # Get first module as example
        first_module = list(model.hierarchical_modules.values())[0]
        active = first_module.get_active_loras(batch)

        active_str = ', '.join([f"Level {l} (r={ranks[l]})" for l in active])
        if not active_str:
            active_str = "None"

        print(f"Batch {batch:3d}: {active_str}")

    print(f"\n{'='*80}")
    print("KEY POINTS")
    print(f"{'='*80}")
    print("""
Normal Fine-Tuning (This Implementation):
  ✓ Base model W_0 frozen
  ✓ LoRAs stay as additive components
  ✓ Forward: W_0·x + Σᵢ ΔWᵢ·x
  ✓ Different update frequencies
  ✓ No merging, no reset
  
Result:
  W = W_0 + LoRA_1 + LoRA_2 + ... + LoRA_L
  
Each LoRA captures patterns at its timescale:
  - Slow LoRA (low rank): Long-term patterns
  - Medium LoRA: Mid-term patterns  
  - Fast LoRA (high rank): Short-term patterns
    """)


if __name__ == '__main__':
    example_hierarchical_lora()