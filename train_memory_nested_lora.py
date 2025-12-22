# train_memory_efficient_sequential.py - Memory-Efficient Sequential LoRA Training
# Only keeps active LoRA in GPU memory, offloads others to CPU

import torch
import torch.nn as nn
from tqdm import tqdm
import os
from datetime import datetime
import gc

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

from utils.memory_efficient_lora import (
    MemoryEfficientTinyViTMultiRankLoRA,
    print_memory_efficient_lora_stats,
    compute_grad_norm
)


# ============================================================================
# Configuration
# ============================================================================

class MemoryEfficientSequentialConfig:
    """
    Configuration for memory-efficient epoch-based sequential multi-rank LoRA training

    Memory Strategy:
    - Only 1 LoRA in GPU memory at a time
    - Other 2 LoRAs stored on CPU
    - ~66% reduction in GPU memory for LoRA parameters
    - Fast switching between LoRAs (~100ms overhead)
    """

    # Model architecture
    ranks = [4, 8, 16]
    lora_alphas = [4, 8, 16]
    lora_dropout = 0.1

    # Learning rates for each LoRA
    lr_lora1 = 3e-3  # Rank 4 - coarse features
    lr_lora2 = 1e-3  # Rank 8 - medium features
    lr_lora3 = 5e-4  # Rank 16 - fine features

    # Epoch-based cycling
    epochs_per_lora = 10  # Train each LoRA for this many epochs before switching
    num_cycles = 3  # How many complete cycles (LoRA1->LoRA2->LoRA3)

    # Alternative: Custom epoch allocation
    use_custom_allocation = False
    custom_epoch_pattern = [10, 10, 10]  # Epochs per LoRA in one cycle

    # Training
    batch_size = 128
    weight_decay = 0.01
    grad_clip = 1.0

    # Scheduler
    use_scheduler = True
    scheduler_type = 'cosine'
    restart_scheduler_on_switch = True

    # Memory management
    clear_cache_on_switch = True  # Force GPU cache clear when switching LoRAs
    print_memory_stats = True  # Print memory stats after each switch


# ============================================================================
# Training Function
# ============================================================================

def train_epoch_single_lora(
        model,
        loader,
        optimizer,
        criterion,
        device,
        epoch,
        lora_idx,
        lora_name,
        grad_clip=1.0
):
    """Train a single LoRA for one epoch"""

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

        # Forward pass (only active LoRA is computed)
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        loss.backward()

        # Compute gradient norm
        grad_norm = compute_grad_norm(model.get_active_lora_parameters())
        grad_norms.append(grad_norm)

        # Gradient clipping
        if grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                model.get_active_lora_parameters(),
                max_norm=grad_clip
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
# Main Training Function
# ============================================================================

def main(
        project_name='memory-efficient-sequential-lora',
        experiment_name=None,
        config=None,
        use_wandb=True,
        save_dir='./checkpoints'
):
    """Main training function with memory-efficient sequential LoRA training"""

    if config is None:
        config = MemoryEfficientSequentialConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Calculate total epochs
    if config.use_custom_allocation:
        epochs_in_cycle = sum(config.custom_epoch_pattern)
        total_epochs = epochs_in_cycle * config.num_cycles
    else:
        epochs_in_cycle = config.epochs_per_lora * 3
        total_epochs = epochs_in_cycle * config.num_cycles

    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'mem_efficient_seq_{config.epochs_per_lora}x{config.num_cycles}_{timestamp}'

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'architecture': 'TinyViT-MemoryEfficientMultiRankLoRA',
                'ranks': config.ranks,
                'lora_alphas': config.lora_alphas,
                'lr_lora1': config.lr_lora1,
                'lr_lora2': config.lr_lora2,
                'lr_lora3': config.lr_lora3,
                'epochs_per_lora': config.epochs_per_lora,
                'num_cycles': config.num_cycles,
                'total_epochs': total_epochs,
                'batch_size': config.batch_size,
                'update_strategy': 'memory_efficient_sequential',
                'memory_strategy': 'only_1_lora_in_gpu'
            }
        )
        print(f"\n✓ W&B initialized: {wandb.run.url}\n")

    # Create model
    print(f"\n{'=' * 70}")
    print(f"Creating Memory-Efficient TinyViT with Multi-Rank LoRA")
    print(f"{'=' * 70}")

    vit_config = TinyViTConfig()
    model = MemoryEfficientTinyViTMultiRankLoRA(
        vit_config,
        ranks=config.ranks,
        lora_alphas=config.lora_alphas,
        lora_dropout=config.lora_dropout
    ).to(device)

    # Print parameter statistics
    active_params, total_lora_params = print_memory_efficient_lora_stats(model)

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
    print(f"Starting Memory-Efficient Sequential Training")
    print(f"{'=' * 70}")
    print(f"Memory Strategy: Only 1 LoRA in GPU at a time (~66% memory savings)")
    print(f"Total Epochs: {total_epochs}")
    print(f"{'=' * 70}\n")

    # ========================================================================
    # Main Training Loop
    # ========================================================================

    for cycle in range(config.num_cycles):
        print(f"\n{'=' * 70}")
        print(f"CYCLE {cycle + 1}/{config.num_cycles}")
        print(f"{'=' * 70}\n")

        # Determine epoch allocation
        if config.use_custom_allocation:
            epoch_allocation = config.custom_epoch_pattern
        else:
            epoch_allocation = [config.epochs_per_lora] * 3

        # Train each LoRA
        for lora_idx in range(3):
            lora_name = f"LoRA{lora_idx + 1}"
            num_epochs_for_this_lora = epoch_allocation[lora_idx]

            print(f"\n{'─' * 70}")
            print(f"Training {lora_name} (rank={config.ranks[lora_idx]}) for {num_epochs_for_this_lora} epochs")
            print(f"{'─' * 70}\n")

            # 🔥 SWITCH TO THIS LORA (offload others to CPU)
            model.switch_active_lora(lora_idx)

            # Print memory stats
            if config.print_memory_stats:
                model.print_memory_stats()

            # Force garbage collection
            if config.clear_cache_on_switch:
                gc.collect()
                torch.cuda.empty_cache()

            # Create optimizer for ONLY active LoRA
            learning_rate = [config.lr_lora1, config.lr_lora2, config.lr_lora3][lora_idx]

            optimizer = torch.optim.AdamW(
                model.get_active_lora_parameters(),
                lr=learning_rate,
                betas=(0.9, 0.999),
                weight_decay=config.weight_decay
            )

            print(f"\n✓ Optimizer created for {lora_name}")
            print(f"  Learning Rate: {learning_rate:.6f}")
            print(f"  Trainable Params: {sum(p.numel() for p in model.get_active_lora_parameters()):,}")

            # Create scheduler
            if config.use_scheduler:
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=num_epochs_for_this_lora
                )
            else:
                scheduler = None

            # Train this LoRA
            for epoch_in_phase in range(num_epochs_for_this_lora):
                global_epoch += 1
                total_epochs_per_lora[lora_idx] += 1

                print(f'\n--- Global Epoch {global_epoch}/{total_epochs} '
                      f'({lora_name} epoch {epoch_in_phase + 1}/{num_epochs_for_this_lora}) ---')

                # Train
                train_loss, train_acc, grad_norm = train_epoch_single_lora(
                    model,
                    train_loader,
                    optimizer,
                    criterion,
                    device,
                    global_epoch,
                    lora_idx,
                    lora_name,
                    grad_clip=config.grad_clip
                )

                # Evaluate
                test_loss, test_acc, per_class_acc = evaluate(
                    model, test_loader, criterion, device
                )

                # Step scheduler
                if scheduler is not None:
                    scheduler.step()

                # Get current learning rate
                current_lr = optimizer.param_groups[0]['lr']

                # GPU memory
                gpu_memory_mb = torch.cuda.memory_allocated() / 1024 ** 2

                # Print summary
                print(f'\nEpoch {global_epoch} Summary:')
                print(f'  Active LoRA: {lora_name} (rank={config.ranks[lora_idx]})')
                print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
                print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
                print(f'  Grad Norm: {grad_norm:.4f}')
                print(f'  Learning Rate: {current_lr:.6f}')
                print(f'  GPU Memory: {gpu_memory_mb:.2f} MB')
                print(f'  Cumulative epochs: L1={total_epochs_per_lora[0]}, '
                      f'L2={total_epochs_per_lora[1]}, L3={total_epochs_per_lora[2]}')

                # Log to W&B
                if use_wandb and WANDB_AVAILABLE:
                    log_dict = {
                        'global_epoch': global_epoch,
                        'cycle': cycle + 1,
                        'active_lora': lora_idx + 1,
                        'active_lora_name': lora_name,
                        'active_lora_rank': config.ranks[lora_idx],
                        'epoch_in_phase': epoch_in_phase + 1,
                        'train_loss': train_loss,
                        'train_acc': train_acc,
                        'test_loss': test_loss,
                        'test_acc': test_acc,
                        'grad_norm': grad_norm,
                        'learning_rate': current_lr,
                        'gpu_memory_mb': gpu_memory_mb,
                        f'lora{lora_idx + 1}_cumulative_epochs': total_epochs_per_lora[lora_idx],
                    }
                    log_dict.update(per_class_acc)
                    wandb.log(log_dict)

                # Save best model
                if test_acc > best_acc:
                    best_acc = test_acc

                    # Get all LoRA states (including offloaded ones)
                    all_lora_states = {}

                    # Current active LoRA
                    all_lora_states[model.active_lora_idx] = {
                        'A': model.head.lora_A.data.cpu().clone(),
                        'B': model.head.lora_B.data.cpu().clone()
                    }

                    # Offloaded LoRAs (already on CPU)
                    for i in range(3):
                        if i != model.active_lora_idx:
                            all_lora_states[i] = model.head.lora_states[i]

                    checkpoint = {
                        'global_epoch': global_epoch,
                        'cycle': cycle,
                        'active_lora': lora_idx,
                        'model_state_dict': model.state_dict(),
                        'all_lora_states': all_lora_states,
                        'best_acc': best_acc,
                        'total_epochs_per_lora': total_epochs_per_lora,
                        'config': vars(config)
                    }

                    checkpoint_path = os.path.join(save_dir, 'best_memory_efficient_model.pth')
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

        # Print final memory stats
        if config.print_memory_stats:
            print(f"\nMemory Statistics at End of Cycle:")
            model.print_memory_stats()

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
    config = MemoryEfficientSequentialConfig()

    # Example configurations:

    # Balanced - Equal epochs for each LoRA
    config.use_custom_allocation = False
    config.epochs_per_lora = 200
    config.num_cycles = 1

    # Custom allocation examples:
    # config.use_custom_allocation = True
    # config.custom_epoch_pattern = [15, 10, 10]  # Emphasize coarse features
    # config.custom_epoch_pattern = [8, 8, 15]    # Emphasize fine features
    # config.custom_epoch_pattern = [5, 10, 15]   # Progressive focus

    print("\n" + "=" * 70)
    print("Memory-Efficient Epoch-Based Sequential Multi-Rank LoRA Training")
    print("=" * 70)
    print("\n🚀 Memory Strategy:")
    print("   • Only 1 LoRA in GPU memory at a time")
    print("   • Other 2 LoRAs offloaded to CPU")
    print("   • ~66% reduction in GPU memory for LoRA parameters")
    print("   • No accuracy loss - full states preserved")
    print("   • Fast switching (~100ms overhead)")

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
        experiment_name=f'mem_efficient_seq_{config.epochs_per_lora}epochs_x{config.num_cycles}cycles',
        config=config,
        use_wandb=True,
        save_dir='./checkpoints'
    )

    print(f'\n🎉 Training finished! Best accuracy: {best_acc:.2f}%')
    print(f'   Strategy: Memory-Efficient Sequential (66% memory savings)')