"""
Dynamic LoRA with Progressive Rank Adjustment

This implementation allows LoRA rank to change during training at fixed intervals.

Strategies:
1. Progressive Growth: Start small, grow larger (r=4 → 8 → 16)
2. Progressive Shrinkage: Start large, shrink down (r=16 → 8 → 4)
3. Curriculum: Vary rank based on training stage
4. Adaptive: Adjust based on performance
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from copy import deepcopy


class DynamicLoRALayer(nn.Module):
    """
    LoRA layer with dynamically adjustable rank
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            initial_rank: int = 4,
            max_rank: int = 32,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.current_rank = initial_rank
        self.max_rank = max_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout_p = lora_dropout

        # Initialize with max rank (we'll only use current_rank dimensions)
        self.lora_A = nn.Parameter(torch.zeros(in_features, max_rank))
        self.lora_B = nn.Parameter(torch.zeros(max_rank, out_features))
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Initialize all dimensions
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # Track rank history
        self.rank_history = [initial_rank]

    @property
    def scaling(self):
        """Dynamic scaling based on current rank"""
        return self.lora_alpha / self.current_rank

    def forward(self, x):
        """Forward pass using only current_rank dimensions"""
        # Device handling
        if x.device != self.lora_A.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)

        # Use only active rank dimensions
        dropout_x = self.lora_dropout(x)
        lora_A_active = self.lora_A[:, :self.current_rank]
        lora_B_active = self.lora_B[:self.current_rank, :]

        lora_out = (dropout_x @ lora_A_active) @ lora_B_active
        return lora_out * self.scaling

    def set_rank(self, new_rank):
        """
        Change the active rank

        Args:
            new_rank: New rank to use (must be <= max_rank)
        """
        if new_rank > self.max_rank:
            raise ValueError(f"new_rank {new_rank} exceeds max_rank {self.max_rank}")

        old_rank = self.current_rank
        self.current_rank = new_rank
        self.rank_history.append(new_rank)

        # If growing, initialize new dimensions
        if new_rank > old_rank:
            with torch.no_grad():
                # Initialize new rows in A
                nn.init.kaiming_uniform_(
                    self.lora_A[:, old_rank:new_rank],
                    a=math.sqrt(5)
                )
                # Initialize new rows in B (keep as zeros)
                self.lora_B[old_rank:new_rank, :] = 0

        return old_rank, new_rank

    def grow_rank(self, increment=1):
        """Increase rank by increment"""
        new_rank = min(self.current_rank + increment, self.max_rank)
        return self.set_rank(new_rank)

    def shrink_rank(self, decrement=1):
        """Decrease rank by decrement"""
        new_rank = max(1, self.current_rank - decrement)
        return self.set_rank(new_rank)

    def get_active_parameters(self):
        """Get number of active parameters"""
        return self.in_features * self.current_rank + self.current_rank * self.out_features


class DynamicLoRALinear(nn.Module):
    """Linear layer with dynamic LoRA"""

    def __init__(
            self,
            original_linear: nn.Linear,
            initial_rank: int = 4,
            max_rank: int = 32,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.lora = DynamicLoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            initial_rank=initial_rank,
            max_rank=max_rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_linear(x) + self.lora(x)

    def set_rank(self, new_rank):
        """Change LoRA rank"""
        return self.lora.set_rank(new_rank)


class RankScheduler:
    """
    Schedule rank changes during training
    """

    def __init__(
            self,
            initial_rank: int = 4,
            final_rank: int = 16,
            change_epochs: list = None,
            strategy: str = 'progressive_growth',
            num_stages: int = 8,
            total_epochs: int = 100
    ):
        """
        Args:
            initial_rank: Starting rank
            final_rank: Ending rank
            change_epochs: List of epochs to change rank (if None, auto-calculate)
            strategy: 'progressive_growth', 'progressive_shrinkage', 'cyclic', 'adaptive'
            num_stages: Number of rank change stages
            total_epochs: Total training epochs
        """
        self.initial_rank = initial_rank
        self.final_rank = final_rank
        self.strategy = strategy
        self.num_stages = num_stages
        self.total_epochs = total_epochs

        # Auto-calculate change epochs if not provided
        if change_epochs is None:
            self.change_epochs = self._calculate_change_epochs()
        else:
            self.change_epochs = sorted(change_epochs)
        print(f"Epoch Scheduler: {self.change_epochs}")

        # Calculate rank schedule
        self.rank_schedule = self._create_schedule()

    def _calculate_change_epochs(self):
        """Auto-calculate when to change ranks"""
        if self.strategy == 'cyclic':
            # Change every few epochs
            interval = self.total_epochs // (self.num_stages * 2)
            return list(range(interval, self.total_epochs, interval))
        else:
            # Evenly spaced changes
            interval = self.total_epochs // self.num_stages
            return [interval * i for i in range(1, self.num_stages)]

    def _create_schedule(self):
        """Create rank schedule based on strategy"""
        schedule = {}
        # print(f"{self.strategy}")

        if self.strategy == 'progressive_growth':
            # Start small, grow larger: 4 → 8 → 12 → 16
            ranks = [
                self.initial_rank + i * (self.final_rank - self.initial_rank) // self.num_stages
                for i in range(self.num_stages + 1)
            ]
            schedule[0] = ranks[0]
            for i, epoch in enumerate(self.change_epochs):
                schedule[epoch] = ranks[i + 1]
            print(f"schedule: {schedule}")

        elif self.strategy == 'progressive_shrinkage':
            # Start large, shrink down: 16 → 12 → 8 → 4
            ranks = [
                self.final_rank - i * (self.final_rank - self.initial_rank) // self.num_stages
                for i in range(self.num_stages + 1)
            ]
            schedule[0] = ranks[0]
            for i, epoch in enumerate(self.change_epochs):
                schedule[epoch] = ranks[i + 1]

        elif self.strategy == 'exponential_growth':
            # Exponential growth: 4 → 8 → 16 → 32 → 64
            # Double the rank at each stage
            ranks = []
            current = self.initial_rank
            ranks.append(current)
            for i in range(self.num_stages):
                print(f"float: {current * 1.5} | int: {int(current * 1.2)}")
                current = min(int(current * 1.5), self.final_rank)
                ranks.append(current)

            schedule[0] = ranks[0]
            for i, epoch in enumerate(self.change_epochs[:len(ranks) - 1]):
                schedule[epoch] = ranks[i + 1]

        elif self.strategy == 'power_of_two':
            # Use exact powers of two: 4, 8, 16, 32, 64
            # Calculate which powers of 2 to use
            import math
            start_power = int(math.log2(self.initial_rank))
            end_power = int(math.log2(self.final_rank))

            ranks = [2 ** i for i in range(start_power, end_power + 1)]

            schedule[0] = ranks[0]
            if len(ranks) > 1:
                # Distribute epochs evenly
                epoch_interval = self.total_epochs // (len(ranks) - 1)
                for i in range(1, len(ranks)):
                    epoch = i * epoch_interval
                    if epoch < self.total_epochs:
                        schedule[epoch] = ranks[i]

        elif self.strategy == 'cyclic':
            # Cycle between small and large: 4 → 16 → 4 → 16
            schedule[0] = self.initial_rank
            for i, epoch in enumerate(self.change_epochs):
                schedule[epoch] = self.final_rank if i % 2 == 0 else self.initial_rank

        elif self.strategy == 'warm_start':
            # Start large, stabilize to medium: 16 → 16 → 8 → 8
            schedule[0] = self.final_rank
            mid_epoch = self.total_epochs // 2
            schedule[mid_epoch] = (self.initial_rank + self.final_rank) // 2

        return schedule

    def get_rank(self, epoch):
        """Get rank for given epoch"""
        # Find the most recent rank change
        applicable_ranks = {e: r for e, r in self.rank_schedule.items() if e <= epoch}
        if applicable_ranks:
            latest_epoch = max(applicable_ranks.keys())
            return applicable_ranks[latest_epoch]
        return self.initial_rank

    def should_change_rank(self, epoch):
        """Check if rank should change at this epoch"""
        return epoch in self.rank_schedule

    def get_schedule_info(self):
        """Get human-readable schedule"""
        info = []
        info.append(f"Strategy: {self.strategy}")
        info.append(f"Initial rank: {self.initial_rank}")
        info.append(f"Final rank: {self.final_rank}")
        info.append(f"\nRank Schedule:")
        for epoch in sorted(self.rank_schedule.keys()):
            rank = self.rank_schedule[epoch]
            info.append(f"  Epoch {epoch:3d}: rank = {rank}")
        return '\n'.join(info)


def apply_dynamic_lora(
        model,
        initial_rank: int = 4,
        max_rank: int = 32,
        lora_alpha: int = 16,
        target_modules: list = None
):
    """
    Apply dynamic LoRA to model

    Args:
        model: Base model
        initial_rank: Starting rank
        max_rank: Maximum rank
        lora_alpha: LoRA scaling
        target_modules: Module names to apply LoRA to
    """
    if target_modules is None:
        target_modules = ["qkv", "proj", "fc1", "fc2"]

    device = next(model.parameters()).device

    for name, module in list(model.named_modules()):
        should_replace = any(target in name for target in target_modules)

        if should_replace and isinstance(module, nn.Linear):
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            if parent_name:
                parent = dict(model.named_modules())[parent_name]
            else:
                parent = model

            # Replace with dynamic LoRA
            dynamic_lora = DynamicLoRALinear(
                module,
                initial_rank=initial_rank,
                max_rank=max_rank,
                lora_alpha=lora_alpha,
            ).to(device)

            setattr(parent, attr_name, dynamic_lora)

    return model


def update_model_rank(model, new_rank):
    """
    Update all LoRA ranks in model

    Args:
        model: Model with dynamic LoRA
        new_rank: New rank to set

    Returns:
        Number of modules updated
    """
    count = 0
    changes = []

    for name, module in model.named_modules():
        if isinstance(module, DynamicLoRALinear):
            old_rank, new_rank_actual = module.set_rank(new_rank)
            count += 1
            changes.append({
                'module': name,
                'old_rank': old_rank,
                'new_rank': new_rank_actual
            })

    return count, changes


def print_rank_info(model):
    """Print rank information for all LoRA modules"""
    print("\n" + "=" * 70)
    print("DYNAMIC LoRA RANK INFORMATION")
    print("=" * 70)

    total_active = 0
    total_max = 0

    for name, module in model.named_modules():
        if isinstance(module, DynamicLoRALinear):
            lora = module.lora
            active_params = lora.get_active_parameters()
            max_params = lora.in_features * lora.max_rank + lora.max_rank * lora.out_features

            print(f"\n{name}:")
            print(f"  Current rank: {lora.current_rank}/{lora.max_rank}")
            print(f"  Active params: {active_params:,}")
            print(f"  Max params: {max_params:,}")
            print(f"  Utilization: {100 * active_params / max_params:.1f}%")

            total_active += active_params
            total_max += max_params

    print(f"\n{'=' * 70}")
    print(f"Total Active LoRA Parameters: {total_active:,}")
    print(f"Total Max LoRA Parameters: {total_max:,}")
    print(f"Overall Utilization: {100 * total_active / total_max:.1f}%")
    print(f"{'=' * 70}\n")


# Example usage and visualization

def visualize_rank_schedule():
    """Visualize different rank schedules"""

    strategies = [
        ('progressive_growth', 4, 16),
        ('progressive_shrinkage', 16, 4),
        ('cyclic', 4, 16),
        ('warm_start', 16, 4),
    ]

    total_epochs = 100

    print("=" * 70)
    print("RANK SCHEDULE VISUALIZATION")
    print("=" * 70)

    for strategy, initial, final in strategies:
        scheduler = RankScheduler(
            initial_rank=initial,
            final_rank=final,
            strategy=strategy,
            num_stages=4,
            total_epochs=total_epochs
        )

        print(f"\n{scheduler.get_schedule_info()}")

        # Show rank at key epochs
        print(f"\nRank progression:")
        for epoch in [0, 10, 25, 50, 75, 90, 99]:
            rank = scheduler.get_rank(epoch)
            print(f"  Epoch {epoch:2d}: rank = {rank}")


def example_training_integration():
    """Show how to integrate with training"""

    print("\n" + "=" * 70)
    print("TRAINING INTEGRATION EXAMPLE")
    print("=" * 70)


if __name__ == '__main__':
    visualize_rank_schedule()
    example_training_integration()

    print("\n" + "=" * 70)
    print("KEY FEATURES")
    print("=" * 70)