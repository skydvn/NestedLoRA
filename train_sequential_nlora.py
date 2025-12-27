# train_epoch_sequential_lora.py - Adaptive Epoch-Based Sequential LoRA Training
# Automatically adapts to any number of LoRAs based on ranks array

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
from datetime import datetime

# Import W&B if available
try:
    import wandb
    WANDB_AVAILABLE = False
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

# Import model components
from models.vit_small import TinyViTConfig, load_pretrained_vit_tiny
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
    """Configuration for epoch-based sequential multi-rank LoRA training"""

    # Model architecture - AUTOMATICALLY ADAPTS TO LENGTH
    ranks = [4, 8, 16]
    lora_alphas = [4, 8, 16]
    lora_dropout = 0.1

    # Learning rates - AUTOMATICALLY ADAPTS TO LENGTH
    learning_rates = [3e-3, 1e-3, 5e-4]

    # Epoch-based cycling
    epochs_per_lora = 10
    num_cycles = 5

    # Alternative: Custom epoch allocation
    use_custom_allocation = False
    custom_epoch_pattern = [10, 10, 10, 10, 10]  # Must match length of ranks

    # Training
    batch_size = 128
    weight_decay = 0.01
    grad_clip = 1.0

    # Scheduler
    use_scheduler = True
    scheduler_type = 'cosine'
    restart_scheduler_on_switch = True

    def __post_init__(self):
        """Validate configuration"""
        num_loras = len(self.ranks)
        assert len(self.lora_alphas) == num_loras, \
            f"lora_alphas length ({len(self.lora_alphas)}) must match ranks length ({num_loras})"
        assert len(self.learning_rates) == num_loras, \
            f"learning_rates length ({len(self.learning_rates)}) must match ranks length ({num_loras})"
        if self.use_custom_allocation:
            assert len(self.custom_epoch_pattern) == num_loras, \
                f"custom_epoch_pattern length ({len(self.custom_epoch_pattern)}) must match ranks length ({num_loras})"


# ============================================================================
# Training Function for Single LoRA
# ============================================================================

def train_epoch_single_lora(model, loader, optimizer, criterion, device, epoch, lora_idx, lora_name):
    """Train a single LoRA for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    grad_norms = []

    pbar = tqdm(loader, desc=f'Epoch {epoch} [{lora_name}]')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()

        grad_norm = compute_grad_norm(model.get_lora_parameters(lora_idx))
        grad_norms.append(grad_norm)

        if hasattr(model, 'config') and hasattr(model.config, 'grad_clip'):
            if model.config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_lora_parameters(lora_idx),
                    max_norm=model.config.grad_clip
                )

        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'grad': f'{grad_norm:.3f}'
        })

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
    num_classes = None
    class_correct = None
    class_total = None

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            if num_classes is None:
                num_classes = outputs.size(1)
                class_correct = [0] * num_classes
                class_total = [0] * num_classes

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            for i in range(labels.size(0)):
                label = labels[i].item()
                class_correct[label] += (predicted[i] == label).item()
                class_total[label] += 1

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

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

def main(project_name='epoch-sequential-lora', experiment_name=None, config=None, 
         use_wandb=True, save_dir='./checkpoints'):
    """Main training function with adaptive epoch-based sequential LoRA training"""

    if config is None:
        config = EpochSequentialConfig()
    
    # Validate configuration
    config.__post_init__()
    
    num_loras = len(config.ranks)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Calculate total epochs
    if config.use_custom_allocation:
        epochs_in_cycle = sum(config.custom_epoch_pattern)
        total_epochs = epochs_in_cycle * config.num_cycles
    else:
        epochs_in_cycle = config.epochs_per_lora * num_loras
        total_epochs = epochs_in_cycle * config.num_cycles

    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if config.use_custom_allocation:
            pattern_str = '_'.join(map(str, config.custom_epoch_pattern))
            experiment_name = f'ImageNet-epoch_seq_custom_{pattern_str}x{config.num_cycles}_{timestamp}'
        else:
            experiment_name = f'ImageNet-epoch_seq_{config.epochs_per_lora}x{config.num_cycles}_{timestamp}'

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb_config = {
            'architecture': 'TinyViT-MultiRankLoRA-EpochSequential',
            'num_loras': num_loras,
            'ranks': config.ranks,
            'lora_alphas': config.lora_alphas,
            'learning_rates': config.learning_rates,
            'epochs_per_lora': config.epochs_per_lora if not config.use_custom_allocation else None,
            'custom_epoch_pattern': config.custom_epoch_pattern if config.use_custom_allocation else None,
            'num_cycles': config.num_cycles,
            'total_epochs': total_epochs,
            'batch_size': config.batch_size,
            'update_strategy': 'epoch_sequential'
        }
        wandb.init(project=project_name, name=experiment_name, config=wandb_config)
        print(f"\n✓ W&B initialized: {wandb.run.url}\n")

    # Create model
    print(f"\n{'=' * 70}")
    print(f"Creating TinyViT with Epoch-Based Sequential Multi-Rank LoRA")
    print(f"Number of LoRAs: {num_loras}")
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

    vit_config = TinyViTConfig()
    model = TinyViTMultiRankLoRA(
        vit_config,
        ranks=config.ranks,
        lora_alphas=config.lora_alphas,
        lora_dropout=config.lora_dropout
    ).to(device)

    # Load pretrained weights
    model = load_pretrained_vit_tiny(model, pretrained_model_name='vit_tiny_patch16_224')

    model.config = config
    trainable_params, total_params = print_multi_rank_parameter_stats(model)

    # Create optimizers - DYNAMICALLY BASED ON NUMBER OF LORAS
    print(f"\n{'=' * 70}")
    print("Creating Optimizers")
    print(f"{'=' * 70}")

    optimizers = create_multi_rank_optimizers(
        model,
        lrs=config.learning_rates,
        weight_decay=config.weight_decay
    )

    # Create schedulers
    if config.use_scheduler:
        if config.use_custom_allocation:
            scheduler_epochs = config.custom_epoch_pattern
        else:
            scheduler_epochs = [config.epochs_per_lora] * num_loras

        schedulers = [
            torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizers[i], T_max=scheduler_epochs[i]
            ) for i in range(num_loras)
        ]
    else:
        schedulers = [None] * num_loras

    # Get data loaders
    train_loader, test_loader = get_data_loaders(batch_size=config.batch_size)
    criterion = nn.CrossEntropyLoss()

    # Training tracking
    best_acc = 0
    global_epoch = 0
    total_epochs_per_lora = [0] * num_loras

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
            epoch_allocation = [config.epochs_per_lora] * num_loras

        # Train each LoRA for its allocated epochs
        for lora_idx in range(num_loras):
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
                    model, train_loader, optimizers[lora_idx], criterion,
                    device, global_epoch, lora_idx, lora_name
                )

                # Evaluate
                test_loss, test_acc, per_class_acc = evaluate(
                    model, test_loader, criterion, device
                )

                # Step scheduler
                if config.use_scheduler and schedulers[lora_idx] is not None:
                    schedulers[lora_idx].step()

                current_lr = optimizers[lora_idx].param_groups[0]['lr']

                # Print summary
                print(f'\nEpoch {global_epoch} Summary:')
                print(f'  Active LoRA: {lora_name} (rank={config.ranks[lora_idx]})')
                print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
                print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
                print(f'  Grad Norm: {grad_norm:.4f}')
                print(f'  Learning Rate: {current_lr:.6f}')
                
                # Print cumulative epochs for all LoRAs
                epochs_str = ', '.join([f'L{i+1}={total_epochs_per_lora[i]}' 
                                       for i in range(num_loras)])
                print(f'  Cumulative epochs per LoRA: {epochs_str}')

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
                    }
                    # Add per-LoRA cumulative epochs
                    for i in range(num_loras):
                        log_dict[f'lora{i+1}_cumulative_epochs'] = total_epochs_per_lora[i]
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
                        'optimizer_state_dicts': [opt.state_dict() for opt in optimizers],
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
        for i in range(num_loras):
            print(f"  LoRA{i + 1} (rank={config.ranks[i]:2d}): {total_epochs_per_lora[i]} epochs")
        print(f"Best Test Accuracy so far: {best_acc:.2f}%")

    # Final summary
    print(f'\n{"=" * 70}')
    print(f'Training Complete!')
    print(f'{"=" * 70}')
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'Total Epochs: {global_epoch}')
    print(f'Final Epoch Distribution:')
    for i in range(num_loras):
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
    config = EpochSequentialConfig()

    # Configuration automatically adapts to the number of ranks
    config.use_custom_allocation = False
    config.epochs_per_lora = 200
    config.num_cycles = 1

    print("\n" + "=" * 70)
    print("Adaptive Epoch-Based Sequential Multi-Rank LoRA Training")
    print("=" * 70)
    print(f"\nNumber of LoRAs: {len(config.ranks)}")
    print(f"Ranks: {config.ranks}")

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
        print(f"  Total Epochs: {config.epochs_per_lora * len(config.ranks) * config.num_cycles}")

    print("=" * 70 + "\n")

    project_name = "tiny-vit-lora-cifar10"

    model, best_acc = main(
        project_name=project_name,
        experiment_name=f'ImageNet-epoch_seq_{config.epochs_per_lora}epochs_x{config.num_cycles}cycles',
        config=config,
        use_wandb=True,
        save_dir='./checkpoints'
    )

    print(f'\n🎉 Training finished! Best accuracy: {best_acc:.2f}%')
    print(f'   Strategy: Epoch-Based Sequential')