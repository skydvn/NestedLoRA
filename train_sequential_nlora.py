# train_epoch_sequential_lora.py - Epoch-Based Sequential LoRA Training
# Train LoRA1 for N epochs, then LoRA2 for N epochs, then LoRA3 for N epochs, then repeat

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime

# Import W&B if available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

# Import model components
from models.vit_small import TinyViTConfig
from trainer import get_data_loaders

from utils.nested_lora import (
    TinyViTMultiRankLoRA,
    create_multi_rank_optimizers,
    print_multi_rank_parameter_stats,
    compute_grad_norm
)


# ============================================================================
# Configuration for Epoch-Based Sequential Training
# ============================================================================

class EpochSequentialConfig:
    """
    Configuration for epoch-based sequential multi-rank LoRA training

    Training proceeds in epoch blocks:
    - Epochs 1-N: Train ONLY LoRA 1
    - Epochs (N+1)-2N: Train ONLY LoRA 2
    - Epochs (2N+1)-3N: Train ONLY LoRA 3
    - Epochs (3N+1)-4N: Train ONLY LoRA 1 (cycle repeats)
    """

    # Model architecture
    ranks = [4, 16, 64]
    lora_alphas = [4, 16, 64]
    lora_dropout = 0.1

    # Learning rates for each LoRA
    lr_lora1 = 3e-3  # Rank 4
    lr_lora2 = 1e-3  # Rank 8
    lr_lora3 = 5e-4  # Rank 16

    # Epoch-based cycling
    epochs_per_lora = 10  # Train each LoRA for this many epochs before switching
    num_cycles = 3  # How many complete cycles (LoRA1->LoRA2->LoRA3)

    # Total epochs = epochs_per_lora * num_loras * num_cycles
    # Example: 10 epochs/LoRA * 3 LoRAs * 3 cycles = 90 total epochs

    # Alternative: Custom epoch allocation
    use_custom_allocation = False
    # epochs_lora1 = 15  # Train LoRA1 for 15 epochs
    # epochs_lora2 = 10  # Train LoRA2 for 10 epochs
    # epochs_lora3 = 15  # Train LoRA3 for 15 epochs
    # num_cycles = 2     # Repeat this pattern 2 times
    custom_epoch_pattern = [10, 10, 10]  # Epochs per LoRA in one cycle

    # Training
    batch_size = 256
    weight_decay = 0.01
    grad_clip = 1.0

    # Scheduler (per LoRA, resets when switching LoRAs)
    use_scheduler = True
    scheduler_type = 'cosine'
    restart_scheduler_on_switch = True  # Restart scheduler when switching LoRAs


# ============================================================================
# Training Function for Single LoRA
# ============================================================================

def train_epoch_single_lora(
        model,
        loader,
        optimizer,
        criterion,
        device,
        epoch,
        lora_idx,
        lora_name
):
    """
    Train a single LoRA for one epoch

    Args:
        model: TinyViT model
        loader: Data loader
        optimizer: Optimizer for the active LoRA
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        lora_idx: Index of the LoRA being trained (0, 1, or 2)
        lora_name: Name for display ('LoRA1', 'LoRA2', 'LoRA3')

    Returns:
        avg_loss, accuracy, grad_norm
    """

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [{lora_name}]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Zero gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Compute gradient norm
        grad_norm = compute_grad_norm(model.get_lora_parameters(lora_idx))
        grad_norms.append(grad_norm)

        # Gradient clipping
        if hasattr(model, 'config') and hasattr(model.config, 'grad_clip'):
            if model.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_lora_parameters(lora_idx),
                    max_norm=model.config.grad_clip
                )

        # Update
        optimizer.step()

        # Track metrics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'grad': f'{grad_norm:.3f}'
        })

    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total
    avg_grad_norm = sum(grad_norms) / len(grad_norms) if grad_norms else 0

    return avg_loss, accuracy, avg_grad_norm


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    # Per-class accuracy - initialize outside loop
    num_classes = None
    class_correct = None
    class_total = None

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            # Initialize per-class tracking on first batch
            if num_classes is None:
                num_classes = outputs.size(1)
                class_correct = [0] * num_classes
                class_total = [0] * num_classes

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    # Calculate per-class accuracy
    per_class_acc = {}
    class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                   'dog', 'frog', 'horse', 'ship', 'truck']

    for i, name in enumerate(class_names):
        if class_total[i] > 0:
            per_class_acc[f'test_acc_{name}'] = 100. * class_correct[i] / class_total[i]

    return avg_loss, accuracy, per_class_acc


# ============================================================================
# Main Training Function
# ============================================================================

def main(
        project_name='epoch-sequential-lora',
        experiment_name=None,
        config=None,
        use_wandb=True,
        save_dir='./checkpoints'
):
    """Main training function with epoch-based sequential LoRA training"""

    if config is None:
        config = EpochSequentialConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Calculate total epochs
    if config.use_custom_allocation:
        epochs_in_cycle = sum(config.custom_epoch_pattern)
        total_epochs = epochs_in_cycle * config.num_cycles
    else:
        epochs_in_cycle = config.epochs_per_lora * 3  # 3 LoRAs
        total_epochs = epochs_in_cycle * config.num_cycles

    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if config.use_custom_allocation:
            pattern_str = '_'.join(map(str, config.custom_epoch_pattern))
            experiment_name = f'epoch_seq_custom_{pattern_str}x{config.num_cycles}_{timestamp}'
        else:
            experiment_name = f'epoch_seq_{config.epochs_per_lora}x{config.num_cycles}_{timestamp}'

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'architecture': 'TinyViT-MultiRankLoRA-EpochSequential',
                'ranks': config.ranks,
                'lora_alphas': config.lora_alphas,
                'lr_lora1': config.lr_lora1,
                'lr_lora2': config.lr_lora2,
                'lr_lora3': config.lr_lora3,
                'epochs_per_lora': config.epochs_per_lora if not config.use_custom_allocation else None,
                'custom_epoch_pattern': config.custom_epoch_pattern if config.use_custom_allocation else None,
                'num_cycles': config.num_cycles,
                'total_epochs': total_epochs,
                'batch_size': config.batch_size,
                'update_strategy': 'epoch_sequential'
            }
        )
        print(f"\n✓ W&B initialized: {wandb.run.url}\n")

    # Create model
    print(f"\n{'=' * 70}")
    print(f"Creating TinyViT with Epoch-Based Sequential Multi-Rank LoRA")
    print(f"{'=' * 70}")
    print(f"Training Strategy:")

    if config.use_custom_allocation:
        print(f"  Custom Pattern per Cycle:")
        for i, epochs in enumerate(config.custom_epoch_pattern):
            print(f"    LoRA {i + 1} (rank={config.ranks[i]:2d}): {epochs} epochs")
        print(f"  Number of Cycles: {config.num_cycles}")
        print(f"  Total Epochs: {total_epochs}")
    else:
        print(f"  Epochs per LoRA: {config.epochs_per_lora}")
        print(f"  Number of Cycles: {config.num_cycles}")
        print(f"  Total Epochs: {total_epochs}")
        print(f"  Pattern:")
        epoch_counter = 1
        for cycle in range(config.num_cycles):
            print(f"    Cycle {cycle + 1}:")
            for i in range(3):
                start = epoch_counter
                end = epoch_counter + config.epochs_per_lora - 1
                print(f"      Epochs {start:3d}-{end:3d}: Train LoRA{i + 1} (rank={config.ranks[i]})")
                epoch_counter += config.epochs_per_lora

    vit_config = TinyViTConfig()
    model = TinyViTMultiRankLoRA(
        vit_config,
        ranks=config.ranks,
        lora_alphas=config.lora_alphas,
        lora_dropout=config.lora_dropout
    ).to(device)

    # Store config in model for access in train function
    model.config = config

    # Print parameter statistics
    trainable_params, total_params = print_multi_rank_parameter_stats(model)

    # Create optimizers
    print(f"\n{'=' * 70}")
    print("Creating Optimizers")
    print(f"{'=' * 70}")

    optimizer1, optimizer2, optimizer3 = create_multi_rank_optimizers(
        model,
        lrs=[config.lr_lora1, config.lr_lora2, config.lr_lora3],
        weight_decay=config.weight_decay
    )
    optimizers = [optimizer1, optimizer2, optimizer3]

    # Create schedulers (will be reset when switching LoRAs)
    if config.use_scheduler:
        if config.use_custom_allocation:
            scheduler_epochs = config.custom_epoch_pattern
        else:
            scheduler_epochs = [config.epochs_per_lora] * 3

        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[i], T_max=scheduler_epochs[i]
            ) for i in range(3)
        ]
    else:
        schedulers = [None, None, None]

    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=config.batch_size)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    best_acc = 0
    global_epoch = 0

    # Track cumulative epochs per LoRA
    total_epochs_per_lora = [0, 0, 0]

    print(f"\n{'=' * 70}")
    print(f"Starting Training")
    print(f"{'=' * 70}\n")

    # ========================================================================
    # Main Training Loop: Cycle through LoRAs by epochs
    # ========================================================================

    for cycle in range(config.num_cycles):
        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1}/{config.num_cycles}")
        print(f"{'=' * 70}\n")

        # Determine epoch allocation for this cycle
        if config.use_custom_allocation:
            epoch_allocation = config.custom_epoch_pattern
        else:
            epoch_allocation = [config.epochs_per_lora] * 3

        # Train each LoRA for its allocated epochs
        for lora_idx in range(3):
            lora_name = f"LoRA{lora_idx + 1}"
            num_epochs_for_this_lora = epoch_allocation[lora_idx]

            print(f"\n{'─' * 70}")
            print(f"Training {lora_name} (rank={config.ranks[lora_idx]}) for {num_epochs_for_this_lora} epochs")
            print(f"{'─' * 70}\n")

            # Reset scheduler if configured
            if config.use_scheduler and config.restart_scheduler_on_switch:
                schedulers[lora_idx] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[lora_idx], T_max=num_epochs_for_this_lora
                )

            # Train this LoRA for the allocated epochs
            for epoch_in_phase in range(num_epochs_for_this_lora):
                global_epoch += 1
                total_epochs_per_lora[lora_idx] += 1

                print(f'\n--- Global Epoch {global_epoch}/{total_epochs} '
                      f'({lora_name} epoch {epoch_in_phase + 1}/{num_epochs_for_this_lora}) ---')

                # Train
                train_loss, train_acc, grad_norm = train_epoch_single_lora(
                    model,
                    train_loader,
                    optimizers[lora_idx],
                    criterion,
                    device,
                    global_epoch,
                    lora_idx,
                    lora_name
                )

                # Evaluate
                test_loss, test_acc, per_class_acc = evaluate(
                    model, test_loader, criterion, device
                )

                # Step scheduler
                if config.use_scheduler and schedulers[lora_idx] is not None:
                    schedulers[lora_idx].step()

                # Get current learning rate
                current_lr = optimizers[lora_idx].param_groups[0]['lr']

                # Print summary
                print(f'\nEpoch {global_epoch} Summary:')
                print(f'  Active LoRA: {lora_name} (rank={config.ranks[lora_idx]})')
                print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
                print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
                print(f'  Grad Norm: {grad_norm:.4f}')
                print(f'  Learning Rate: {current_lr:.6f}')
                print(f'  Cumulative epochs per LoRA: L1={total_epochs_per_lora[0]}, '
                      f'L2={total_epochs_per_lora[1]}, L3={total_epochs_per_lora[2]}')

                # Log to W&B
                if use_wandb and WANDB_AVAILABLE:
                    log_dict = {
                        'global_epoch': global_epoch,
                        'cycle': cycle + 1,
                        'active_lora': lora_idx + 1,
                        'active_lora_name': lora_name,
                        'epoch_in_phase': epoch_in_phase + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'grad_norm': grad_norm,
                        'learning_rate': current_lr,
                        f'lora{lora_idx + 1}_cumulative_epochs': total_epochs_per_lora[lora_idx],
                    }
                    log_dict.update(per_class_acc)
                    wandb.log(log_dict)

                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc

                    checkpoint = {
                        'global_epoch': global_epoch,
                        'cycle': cycle,
                        'active_lora': lora_idx,
                        'model_state_dict': model.state_dict(),
                        'optimizer1_state_dict': optimizer1.state_dict(),
                        'optimizer2_state_dict': optimizer2.state_dict(),
                        'optimizer3_state_dict': optimizer3.state_dict(),
                        'best_acc': best_acc,
                        'total_epochs_per_lora': total_epochs_per_lora,
                        'config': vars(config)
                    }

                    checkpoint_path = os.path.join(save_dir, 'best_epoch_sequential_model.pth')
                    torch.save(checkpoint, checkpoint_path)
                    print(f'  ✓ Saved best model: {best_acc:.2f}%')

                    if use_wandb and WANDB_AVAILABLE:
                        wandb.run.summary['best_accuracy'] = best_acc
                        wandb.run.summary['best_epoch'] = global_epoch
                        wandb.run.summary['best_active_lora'] = lora_name

        # End of cycle summary
        print(f"\n{'=' * 70}")
        print(f"End of Cycle {cycle + 1}")
        print(f"{'=' * 70}")
        print(f"Cumulative Epochs per LoRA:")
        for i in range(3):
            print(f"  LoRA{i + 1} (rank={config.ranks[i]:2d}): {total_epochs_per_lora[i]} epochs")
        print(f"Best Test Accuracy so far: {best_acc:.2f}%")

    # Final summary
    print(f'\n{"=" * 70}')
    print(f'Training Complete!')
    print(f'{"=" * 70}')
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'Total Epochs: {global_epoch}')
    print(f'Final Epoch Distribution:')
    for i in range(3):
        print(f'  LoRA{i + 1} (rank={config.ranks[i]:2d}): {total_epochs_per_lora[i]} epochs')
    print(f'Checkpoints saved to: {save_dir}')

    if use_wandb and WANDB_AVAILABLE:
        print(f'W&B Dashboard: {wandb.run.url}')
        wandb.finish()

    print(f'{"=" * 70}\n')

    return model, best_acc


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    # Create configuration
    config = EpochSequentialConfig()

    # ========================================================================
    # Example 1: Balanced - Each LoRA gets equal epochs
    # ========================================================================
    config.use_custom_allocation = False
    config.epochs_per_lora = 100  # Each LoRA trains for 50 epochs
    config.num_cycles = 1  # Repeat 3 times
    # Total: 50 epochs × 3 LoRAs × 1 cycles = 150 epochs
    # Pattern: L1(10) -> L2(10) -> L3(10) -> L1(10) -> L2(10) -> L3(10) -> L1(10) -> L2(10) -> L3(10)

    # ========================================================================
    # Example 2: Emphasize Coarse Features (LoRA1)
    # ========================================================================
    # config.use_custom_allocation = True
    # config.custom_epoch_pattern = [15, 10, 10]  # LoRA1: 15, LoRA2: 10, LoRA3: 10
    # config.num_cycles = 2
    # Total: (15+10+10) × 2 = 70 epochs

    # ========================================================================
    # Example 3: Emphasize Fine Features (LoRA3)
    # ========================================================================
    # config.use_custom_allocation = True
    # config.custom_epoch_pattern = [8, 8, 15]  # LoRA3 gets more epochs
    # config.num_cycles = 3

    # ========================================================================
    # Example 4: Progressive Focus
    # ========================================================================
    # config.use_custom_allocation = True
    # config.custom_epoch_pattern = [5, 10, 15]  # More epochs for higher ranks
    # config.num_cycles = 2

    # ========================================================================
    # Example 5: Quick Cycles
    # ========================================================================
    # config.use_custom_allocation = False
    # config.epochs_per_lora = 3   # Quick switches
    # config.num_cycles = 10       # Many cycles
    # Total: 3 × 3 × 10 = 90 epochs

    print("\n" + "=" * 70)
    print("Epoch-Based Sequential Multi-Rank LoRA Training")
    print("=" * 70)

    if config.use_custom_allocation:
        print(f"\nCustom Epoch Allocation:")
        for i, epochs in enumerate(config.custom_epoch_pattern):
            print(f"  LoRA{i + 1} (rank={config.ranks[i]:2d}): {epochs} epochs per cycle")
        print(f"Number of Cycles: {config.num_cycles}")
        total = sum(config.custom_epoch_pattern) * config.num_cycles
        print(f"Total Epochs: {total}")
    else:
        print(f"\nBalanced Allocation:")
        print(f"  Each LoRA: {config.epochs_per_lora} epochs per cycle")
        print(f"  Number of Cycles: {config.num_cycles}")
        print(f"  Total Epochs: {config.epochs_per_lora * 3 * config.num_cycles}")

    print("=" * 70 + "\n")

    project_name = "tiny-vit-lora-cifar10"

    # Run training
    model, best_acc = main(
        project_name=project_name,
        experiment_name=f'epoch_seq_{config.epochs_per_lora}epochs_x{config.num_cycles}cycles',
        config=config,
        use_wandb=True,
        save_dir='./checkpoints'
    )

    print(f'\n🎉 Training finished! Best accuracy: {best_acc:.2f}%')
    print(f'   Strategy: Epoch-Based Sequential')