# train_epoch_sequential_gpm.py - Epoch-Sequential LoRA with GPM for Orthogonality
# Train each LoRA sequentially while ensuring orthogonality using Gradient Projection Memory

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

from utils.orthogonal import (
    GradientProjectionMemory,
    GPMHook,
    analyze_lora_orthogonality,
    print_gpm_status
)


# ============================================================================
# Configuration for GPM-Enhanced Epoch-Sequential Training
# ============================================================================

class GPMEpochSequentialConfig:
    """
    Configuration for GPM-enhanced epoch-sequential training

    Each LoRA is trained sequentially, with GPM ensuring orthogonality
    to previously trained LoRAs.
    """

    # Model architecture
    ranks = [16, 32, 64] # [4, 16, 64]
    lora_alphas = [16, 32, 64] # [4, 16, 64]
    lora_dropout = 0.1

    # Learning rates for each LoRA
    lr_lora1 = 3e-3  # Rank 4
    lr_lora2 = 1e-3  # Rank 8
    lr_lora3 = 5e-4  # Rank 16

    # Epoch-based cycling
    epochs_per_lora = 33  # Train each LoRA for this many epochs
    num_cycles = 1  # How many complete cycles

    # GPM Configuration
    use_gpm = True  # Enable GPM
    gpm_threshold = 0.95  # Variance threshold for SVD (0.90-0.99)
    gpm_memory_strength = 1.0  # Projection strength (0.0-1.0)
    gpm_num_batches_importance = 100  # Batches for computing importance
    gpm_num_batches_projection = 50  # Batches for computing projections

    # Alternative: Use hooks for automatic projection
    use_gpm_hooks = True  # If True, use hooks; if False, manual projection

    # Training
    batch_size = 128
    weight_decay = 0.01
    grad_clip = 1.0

    # Scheduler
    use_scheduler = True
    restart_scheduler_on_switch = True

    # Analysis
    analyze_orthogonality_every_n_epochs = 5  # Analyze orthogonality


# ============================================================================
# Training Function with GPM
# ============================================================================

def train_epoch_with_gpm(
        model,
        loader,
        optimizer,
        criterion,
        device,
        epoch,
        lora_idx,
        lora_name,
        gpm: GradientProjectionMemory = None,
        use_hooks: bool = True
):
    """
    Train a single LoRA for one epoch with GPM projection

    Args:
        model: TinyViT model
        loader: Data loader
        optimizer: Optimizer for the active LoRA
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        lora_idx: Index of the LoRA being trained (0, 1, or 2)
        lora_name: Name for display ('LoRA1', 'LoRA2', 'LoRA3')
        gpm: GPM instance (None if GPM disabled)
        use_hooks: Whether to use automatic hooks or manual projection

    Returns:
        avg_loss, accuracy, grad_norm
    """

    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []

    # Register GPM hooks if using automatic projection
    gpm_hook = None
    if gpm is not None and use_hooks and len(gpm.locked_loras) > 0:
        gpm_hook = GPMHook(gpm, model, lora_idx)
        gpm_hook.register_hooks()

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

        # Manual GPM projection (if not using hooks)
        if gpm is not None and not use_hooks:
            gpm.project_gradients(model, lora_idx)

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
        locked_info = f"🔒{gpm.locked_loras}" if gpm and len(gpm.locked_loras) > 0 else ""
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'grad': f'{grad_norm:.3f}',
            'locked': locked_info
        })

    # Remove hooks
    if gpm_hook is not None:
        gpm_hook.remove_hooks()

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

    # Per-class accuracy
    class_correct = [0] * 10
    class_total = [0] * 10

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Per-class accuracy
            for i in range(labels.size(0)):
                label = labels[i]
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
# Main Training Function with GPM
# ============================================================================

def main(
        project_name='gpm-epoch-sequential-lora',
        experiment_name=None,
        config=None,
        use_wandb=True,
        save_dir='./checkpoints'
):
    """Main training function with GPM-enhanced orthogonality"""

    if config is None:
        config = GPMEpochSequentialConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Calculate total epochs
    epochs_in_cycle = config.epochs_per_lora * 3
    total_epochs = epochs_in_cycle * config.num_cycles

    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        gpm_str = f'gpm{config.gpm_threshold:.2f}' if config.use_gpm else 'nogpm'
        experiment_name = f'epoch_seq_{gpm_str}_{config.epochs_per_lora}x{config.num_cycles}_{timestamp}'

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'architecture': 'TinyViT-MultiRankLoRA-GPM',
                'ranks': config.ranks,
                'lora_alphas': config.lora_alphas,
                'use_gpm': config.use_gpm,
                'gpm_threshold': config.gpm_threshold,
                'gpm_memory_strength': config.gpm_memory_strength,
                'use_gpm_hooks': config.use_gpm_hooks,
                'epochs_per_lora': config.epochs_per_lora,
                'num_cycles': config.num_cycles,
                'total_epochs': total_epochs,
                'batch_size': config.batch_size,
                'update_strategy': 'epoch_sequential_gpm'
            }
        )
        print(f"\n✓ W&B initialized: {wandb.run.url}\n")

    # Create model
    print(f"\n{'=' * 70}")
    print(f"Creating TinyViT with GPM-Enhanced Epoch-Sequential LoRA")
    print(f"{'=' * 70}")
    print(f"GPM Configuration:")
    print(f"  Enabled: {config.use_gpm}")
    if config.use_gpm:
        print(f"  Variance Threshold: {config.gpm_threshold}")
        print(f"  Memory Strength: {config.gpm_memory_strength}")
        print(f"  Use Hooks: {config.use_gpm_hooks}")
    print(f"\nTraining Strategy:")
    print(f"  Epochs per LoRA: {config.epochs_per_lora}")
    print(f"  Number of Cycles: {config.num_cycles}")
    print(f"  Total Epochs: {total_epochs}")

    vit_config = TinyViTConfig()
    model = TinyViTMultiRankLoRA(
        vit_config,
        ranks=config.ranks,
        lora_alphas=config.lora_alphas,
        lora_dropout=config.lora_dropout
    ).to(device)

    model.config = config

    # Print parameter statistics
    trainable_params, total_params = print_multi_rank_parameter_stats(model)

    # Initialize GPM
    gpm = None
    if config.use_gpm:
        gpm = GradientProjectionMemory(
            threshold=config.gpm_threshold,
            memory_strength=config.gpm_memory_strength,
            device=device
        )

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

    # Create schedulers
    if config.use_scheduler:
        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[i], T_max=config.epochs_per_lora
            ) for i in range(3)
        ]
    else:
        schedulers = [None, None, None]

    # Get data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=config.batch_size)

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    best_acc = 0
    global_epoch = 0
    total_epochs_per_lora = [0, 0, 0]

    print(f"\n{'=' * 70}")
    print(f"Starting Training with GPM")
    print(f"{'=' * 70}\n")

    # ========================================================================
    # Main Training Loop: Cycle through LoRAs with GPM
    # ========================================================================

    for cycle in range(config.num_cycles):
        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1}/{config.num_cycles}")
        print(f"{'=' * 70}\n")

        if gpm is not None:
            print_gpm_status(gpm)

        # Train each LoRA sequentially
        for lora_idx in range(3):
            lora_name = f"LoRA{lora_idx + 1}"
            num_epochs_for_this_lora = config.epochs_per_lora

            print(f"\n{'─' * 70}")
            print(f"Training {lora_name} (rank={config.ranks[lora_idx]})")
            print(f"{'─' * 70}")

            if gpm is not None and len(gpm.locked_loras) > 0:
                print(f"  📌 GPM Protection: Projecting away from LoRAs {[i + 1 for i in gpm.locked_loras]}")
            else:
                print(f"  🆕 First LoRA or GPM disabled - no projection")

            print()

            # Reset scheduler if configured
            if config.use_scheduler and config.restart_scheduler_on_switch:
                schedulers[lora_idx] = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizers[lora_idx], T_max=num_epochs_for_this_lora
                )

            # Train this LoRA for allocated epochs
            for epoch_in_phase in range(num_epochs_for_this_lora):
                global_epoch += 1
                total_epochs_per_lora[lora_idx] += 1

                print(f'\n--- Global Epoch {global_epoch}/{total_epochs} '
                      f'({lora_name} epoch {epoch_in_phase + 1}/{num_epochs_for_this_lora}) ---')

                # Train with GPM
                train_loss, train_acc, grad_norm = train_epoch_with_gpm(
                    model,
                    train_loader,
                    optimizers[lora_idx],
                    criterion,
                    device,
                    global_epoch,
                    lora_idx,
                    lora_name,
                    gpm=gpm,
                    use_hooks=config.use_gpm_hooks
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

                # Analyze orthogonality periodically
                if (global_epoch % config.analyze_orthogonality_every_n_epochs == 0 and
                        len(gpm.locked_loras) > 0):
                    analyze_lora_orthogonality(model, gpm)

                # Log to W&B
                if use_wandb and WANDB_AVAILABLE:
                    log_dict = {
                        'global_epoch': global_epoch,
                        'cycle': cycle + 1,
                        'active_lora': lora_idx + 1,
                        'active_lora_name': lora_name,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'grad_norm': grad_norm,
                        'learning_rate': current_lr,
                        'num_locked_loras': len(gpm.locked_loras) if gpm else 0,
                    }

                    # Add orthogonality metrics
                    if gpm is not None:
                        orth_metrics = gpm.get_orthogonality_metrics(model)
                        log_dict.update(orth_metrics)

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

                    checkpoint_path = os.path.join(save_dir, 'best_gpm_model.pth')
                    torch.save(checkpoint, checkpoint_path)

                    # Save GPM state
                    if gpm is not None:
                        gpm_path = os.path.join(save_dir, 'gpm_state.pth')
                        gpm.save(gpm_path)

                    print(f'  ✓ Saved best model: {best_acc:.2f}%')

                    if use_wandb and WANDB_AVAILABLE:
                        wandb.run.summary['best_accuracy'] = best_acc
                        wandb.run.summary['best_epoch'] = global_epoch

            # ================================================================
            # After training this LoRA: Lock it with GPM
            # ================================================================

            if gpm is not None and lora_idx not in gpm.locked_loras:
                print(f"\n{'=' * 70}")
                print(f"Locking {lora_name} with GPM")
                print(f"{'=' * 70}")

                # Compute feature importance
                gpm.compute_feature_importance(
                    model,
                    train_loader,
                    lora_idx,
                    num_batches=config.gpm_num_batches_importance
                )

                # Compute projection matrices
                gpm.compute_projection_matrices(
                    model,
                    train_loader,
                    lora_idx,
                    num_batches=config.gpm_num_batches_projection
                )

                print(f"\n  ✓ {lora_name} is now locked and protected")
                print(f"  ✓ Future LoRAs will be projected to be orthogonal to {lora_name}")

                # Analyze orthogonality after locking
                if len(gpm.locked_loras) > 1:
                    analyze_lora_orthogonality(model, gpm)

        # End of cycle summary
        print(f"\n{'=' * 70}")
        print(f"End of Cycle {cycle + 1}")
        print(f"{'=' * 70}")
        if gpm is not None:
            print_gpm_status(gpm)
        print(f"Best Test Accuracy: {best_acc:.2f}%")

    # Final analysis
    print(f"\n{'=' * 70}")
    print(f"FINAL ORTHOGONALITY ANALYSIS")
    print(f"{'=' * 70}")
    analyze_lora_orthogonality(model, gpm)

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

    return model, best_acc, gpm


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    # Create configuration
    config = GPMEpochSequentialConfig()

    # ========================================================================
    # GPM Configuration Options
    # ========================================================================

    # Option 1: Standard GPM (recommended)
    config.use_gpm = True
    config.gpm_threshold = 0.95  # Keep 95% of variance
    config.gpm_memory_strength = 1.0  # Full projection
    config.use_gpm_hooks = True  # Automatic projection

    # Option 2: Lighter GPM
    # config.use_gpm = True
    # config.gpm_threshold = 0.90        # Keep 90% of variance (fewer components)
    # config.gpm_memory_strength = 0.8   # Partial projection

    # Option 3: No GPM (baseline)
    # config.use_gpm = False

    # ========================================================================
    # Training Configuration
    # ========================================================================

    config.epochs_per_lora = 100
    config.num_cycles = 1
    config.analyze_orthogonality_every_n_epochs = 5

    print("\n" + "=" * 70)
    print("GPM-Enhanced Epoch-Sequential Multi-Rank LoRA Training")
    print("=" * 70)
    print(f"\nGPM Configuration:")
    print(f"  Enabled: {config.use_gpm}")
    if config.use_gpm:
        print(f"  Variance Threshold: {config.gpm_threshold}")
        print(f"  Memory Strength: {config.gpm_memory_strength}")
        print(f"  Use Hooks: {config.use_gpm_hooks}")
    print(f"\nTraining:")
    print(f"  Epochs per LoRA: {config.epochs_per_lora}")
    print(f"  Cycles: {config.num_cycles}")
    print(f"  Total Epochs: {config.epochs_per_lora * 3 * config.num_cycles}")
    print("=" * 70 + "\n")

    project_name = "tiny-vit-lora-cifar10"

    # Run training
    model, best_acc, gpm = main(
        project_name=project_name,
        experiment_name=f'gpm_epoch_seq_{config.epochs_per_lora}x{config.num_cycles}',
        config=config,
        use_wandb=True,
        save_dir='./checkpoints'
    )

    print(f'\n🎉 Training finished! Best accuracy: {best_acc:.2f}%')
    print(f'   Strategy: GPM-Enhanced Epoch-Sequential')
    print(f'   LoRAs are orthogonal: {config.use_gpm}')