# train_multi_lora.py - Training script for Multi-Optimizer Additive LoRA

import torch
import torch.nn as nn
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
# Configuration with Iteration-Based Steps
# ============================================================================

class IterationBasedConfig:
    """
    Configuration for iteration-based multi-rank LoRA training

    Each LoRA updates when: global_iteration % step_lora_i == 0
    """

    # Model architecture
    ranks = [4, 8, 16]
    lora_alphas = [4, 8, 16]
    lora_dropout = 0.1

    # Learning rates
    lr_lora1 = 3e-2  # Rank 4
    lr_lora2 = 1e-2  # Rank 8
    lr_lora3 = 5e-3  # Rank 16

    # Iteration-based update steps
    # LoRA i updates when: iteration % step_lora_i == 0
    # step_lora1 = 1  # Update every 1 iteration (every batch)
    # step_lora2 = 2  # Update every 2 iterations (every 2 batches)
    # step_lora3 = 4  # Update every 4 iterations (every 4 batches)

    # Alternative strategies:
    # Strategy 1: All update every batch
    # step_lora1 = 1, step_lora2 = 1, step_lora3 = 1

    # Strategy 2: Progressive (fine features update less often)
    # step_lora1 = 1, step_lora2 = 2, step_lora3 = 4

    # Strategy 3: Inverse (fine features update more often)
    step_lora1 = 4
    step_lora2 = 2
    step_lora3 = 1

    # Training
    batch_size = 128
    num_epochs = 100
    weight_decay = 0.01
    grad_clip = 1.0

    # Scheduler
    use_scheduler = True
    scheduler_type = 'cosine'


# ============================================================================
# Iteration-Based Training Function
# ============================================================================

def train_epoch_iteration_based(
        model,
        loader,
        optimizers,
        criterion,
        device,
        epoch,
        config,
        global_iteration
):
    """
    Train with iteration-based LoRA updates

    Each LoRA updates when: global_iteration % step_lora_i == 0

    Args:
        model: TinyViT model
        loader: Data loader
        optimizers: Tuple of 3 optimizers
        criterion: Loss function
        device: Device to train on
        epoch: Current epoch number
        config: Training configuration
        global_iteration: Starting global iteration count

    Returns:
        avg_loss, accuracy, lora_losses, grad_norms, new_global_iteration
    """

    optimizer1, optimizer2, optimizer3 = optimizers

    model.train()
    total_loss = 0
    correct = 0
    total = 0

    # Track per-LoRA metrics
    losses = {'lora1': 0, 'lora2': 0, 'lora3': 0}
    update_counts = {'lora1': 0, 'lora2': 0, 'lora3': 0}
    grad_norms = {'lora1': [], 'lora2': [], 'lora3': []}

    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Calculate current iteration
        iteration = global_iteration + batch_idx

        # ================================================================
        # Check which LoRAs should update this iteration
        # ================================================================

        update_lora1 = (iteration % config.step_lora1 == 0)
        update_lora2 = (iteration % config.step_lora2 == 0)
        update_lora3 = (iteration % config.step_lora3 == 0)

        # ================================================================
        # LORA 1: Update if iteration % step_lora1 == 0
        # ================================================================
        if update_lora1:
            optimizer1.zero_grad()
            outputs = model(images)
            loss1 = criterion(outputs, labels)
            loss1.backward()

            grad_norm1 = compute_grad_norm(model.get_lora_parameters(0))
            grad_norms['lora1'].append(grad_norm1)

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_lora_parameters(0),
                    max_norm=config.grad_clip
                )

            optimizer1.step()
            losses['lora1'] += loss1.item()
            update_counts['lora1'] += 1

        # ================================================================
        # LORA 2: Update if iteration % step_lora2 == 0
        # ================================================================
        if update_lora2:
            optimizer2.zero_grad()
            outputs = model(images)
            loss2 = criterion(outputs, labels)
            loss2.backward()

            grad_norm2 = compute_grad_norm(model.get_lora_parameters(1))
            grad_norms['lora2'].append(grad_norm2)

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_lora_parameters(1),
                    max_norm=config.grad_clip
                )

            optimizer2.step()
            losses['lora2'] += loss2.item()
            update_counts['lora2'] += 1

        ================================================================
        LORA 3: Update if iteration % step_lora3 == 0
        ================================================================
        if update_lora3:
            optimizer3.zero_grad()
            outputs = model(images)
            loss3 = criterion(outputs, labels)
            loss3.backward()

            grad_norm3 = compute_grad_norm(model.get_lora_parameters(2))
            grad_norms['lora3'].append(grad_norm3)

            if config.grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    model.get_lora_parameters(2),
                    max_norm=config.grad_clip
                )

            optimizer3.step()
            losses['lora3'] += loss3.item()
            update_counts['lora3'] += 1

        # ================================================================
        # Track overall metrics (after updates)
        # ================================================================
        with torch.no_grad():
            outputs = model(images)
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

        # Update progress bar
        update_str = f"U:[{'1' if update_lora1 else '-'}{'2' if update_lora2 else '-'}{'3' if update_lora3 else '-'}]"
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%',
            'iter': iteration,
            'updates': update_str,
            'L1': f'{losses["lora1"] / (update_counts["lora1"] + 1e-8):.3f}',
            'L2': f'{losses["lora2"] / (update_counts["lora2"] + 1e-8):.3f}',
            'L3': f'{losses["lora3"] / (update_counts["lora3"] + 1e-8):.3f}'
        })

    # Calculate metrics
    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    # Average losses per LoRA (only for batches where they updated)
    avg_losses = {
        'lora1': losses['lora1'] / max(update_counts['lora1'], 1),
        'lora2': losses['lora2'] / max(update_counts['lora2'], 1),
        'lora3': losses['lora3'] / max(update_counts['lora3'], 1)
    }

    # Average gradient norms
    avg_grad_norms = {
        'lora1': sum(grad_norms['lora1']) / len(grad_norms['lora1']) if grad_norms['lora1'] else 0,
        'lora2': sum(grad_norms['lora2']) / len(grad_norms['lora2']) if grad_norms['lora2'] else 0,
        'lora3': sum(grad_norms['lora3']) / len(grad_norms['lora3']) if grad_norms['lora3'] else 0
    }

    # New global iteration count
    new_global_iteration = global_iteration + len(loader)

    return avg_loss, accuracy, avg_losses, avg_grad_norms, update_counts, new_global_iteration


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
        project_name='iteration-based-lora',
        experiment_name=None,
        config=None,
        use_wandb=True,
        save_dir='./checkpoints'
):
    """Main training function with iteration-based LoRA updates"""

    if config is None:
        config = IterationBasedConfig()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        step_str = f's{config.step_lora1}_{config.step_lora2}_{config.step_lora3}'
        experiment_name = f'iter_based_{step_str}_{timestamp}'

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.login(key="b1d6eed8871c7668a889ae74a621b5dbd2f3b070")
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'architecture': 'TinyViT-MultiRankLoRA-IterationBased',
                'ranks': config.ranks,
                'lora_alphas': config.lora_alphas,
                'lr_lora1': config.lr_lora1,
                'lr_lora2': config.lr_lora2,
                'lr_lora3': config.lr_lora3,
                'step_lora1': config.step_lora1,
                'step_lora2': config.step_lora2,
                'step_lora3': config.step_lora3,
                'batch_size': config.batch_size,
                'num_epochs': config.num_epochs,
                'update_strategy': 'iteration_based'
            }
        )
        print(f"\n✓ W&B initialized: {wandb.run.url}\n")

    # Create model
    print(f"\n{'=' * 70}")
    print(f"Creating TinyViT with Iteration-Based Multi-Rank LoRA")
    print(f"{'=' * 70}")
    print(f"Update Schedule:")
    print(f"  LoRA 1 (rank=4):  Every {config.step_lora1} iteration(s)")
    print(f"  LoRA 2 (rank=8):  Every {config.step_lora2} iteration(s)")
    print(f"  LoRA 3 (rank=16): Every {config.step_lora3} iteration(s)")

    vit_config = TinyViTConfig()
    model = TinyViTMultiRankLoRA(
        vit_config,
        ranks=config.ranks,
        lora_alphas=config.lora_alphas,
        lora_dropout=config.lora_dropout
    ).to(device)

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

    # Create schedulers
    if config.use_scheduler:
        scheduler1 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer1, T_max=config.num_epochs
        )
        scheduler2 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer2, T_max=config.num_epochs
        )
        scheduler3 = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer3, T_max=config.num_epochs
        )

    # Get data loaders
    print("\nLoading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=config.batch_size)

    # Calculate iterations per epoch
    iterations_per_epoch = len(train_loader)
    total_iterations = iterations_per_epoch * config.num_epochs

    print(f"\nTraining Details:")
    print(f"  Batches per epoch:  {iterations_per_epoch}")
    print(f"  Total iterations:   {total_iterations}")
    print(f"  LoRA 1 updates:     ~{total_iterations // config.step_lora1}")
    print(f"  LoRA 2 updates:     ~{total_iterations // config.step_lora2}")
    print(f"  LoRA 3 updates:     ~{total_iterations // config.step_lora3}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0
    global_iteration = 0

    print(f"\n{'=' * 70}")
    print(f"Starting Training")
    print(f"{'=' * 70}\n")

    for epoch in range(1, config.num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{config.num_epochs} (Iteration {global_iteration}) ---')

        # Train
        train_loss, train_acc, lora_losses, grad_norms, update_counts, global_iteration = train_epoch_iteration_based(
            model,
            train_loader,
            (optimizer1, optimizer2, optimizer3),
            criterion,
            device,
            epoch,
            config,
            global_iteration
        )

        # Evaluate
        test_loss, test_acc, per_class_acc = evaluate(
            model, test_loader, criterion, device
        )

        # Step schedulers
        if config.use_scheduler:
            scheduler1.step()
            scheduler2.step()
            scheduler3.step()

        # Get current learning rates
        lr1 = optimizer1.param_groups[0]['lr']
        lr2 = optimizer2.param_groups[0]['lr']
        lr3 = optimizer3.param_groups[0]['lr']

        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
        print(f'  Global Iteration: {global_iteration}')
        print(f'  Update Counts:')
        print(f'    LoRA 1 (rank=4):  {update_counts["lora1"]:3d} updates, loss={lora_losses["lora1"]:.4f}')
        print(f'    LoRA 2 (rank=8):  {update_counts["lora2"]:3d} updates, loss={lora_losses["lora2"]:.4f}')
        print(f'    LoRA 3 (rank=16): {update_counts["lora3"]:3d} updates, loss={lora_losses["lora3"]:.4f}')
        print(f'  Grad Norms:')
        print(f'    LoRA 1: {grad_norms["lora1"]:.4f}')
        print(f'    LoRA 2: {grad_norms["lora2"]:.4f}')
        print(f'    LoRA 3: {grad_norms["lora3"]:.4f}')
        print(f'  Learning Rates:')
        print(f'    LoRA 1: {lr1:.6f}')
        print(f'    LoRA 2: {lr2:.6f}')
        print(f'    LoRA 3: {lr3:.6f}')

        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'global_iteration': global_iteration,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'lora1_loss': lora_losses['lora1'],
                'lora2_loss': lora_losses['lora2'],
                'lora3_loss': lora_losses['lora3'],
                'lora1_updates': update_counts['lora1'],
                'lora2_updates': update_counts['lora2'],
                'lora3_updates': update_counts['lora3'],
                'lora1_grad_norm': grad_norms['lora1'],
                'lora2_grad_norm': grad_norms['lora2'],
                'lora3_grad_norm': grad_norms['lora3'],
                'lr_lora1': lr1,
                'lr_lora2': lr2,
                'lr_lora3': lr3
            }
            log_dict.update(per_class_acc)
            wandb.log(log_dict)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc

            checkpoint = {
                'epoch': epoch,
                'global_iteration': global_iteration,
                'model_state_dict': model.state_dict(),
                'optimizer1_state_dict': optimizer1.state_dict(),
                'optimizer2_state_dict': optimizer2.state_dict(),
                'optimizer3_state_dict': optimizer3.state_dict(),
                'best_acc': best_acc,
                'config': vars(config)
            }

            checkpoint_path = os.path.join(save_dir, 'best_iteration_based_model.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f'  ✓ Saved best model: {best_acc:.2f}%')

            if use_wandb and WANDB_AVAILABLE:
                wandb.run.summary['best_accuracy'] = best_acc
                wandb.run.summary['best_epoch'] = epoch
                wandb.run.summary['best_iteration'] = global_iteration

    # Final summary
    print(f'\n{"=" * 70}')
    print(f'Training Complete!')
    print(f'{"=" * 70}')
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'Total Iterations: {global_iteration}')
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
    config = IterationBasedConfig()

    # Example configurations:

    # Config 1: Progressive updates (default)
    # Coarse features (rank=4) update most often
    # Fine features (rank=16) update least often
    config.step_lora1 = 1  # Every iteration
    config.step_lora2 = 2  # Every 2 iterations
    config.step_lora3 = 4  # Every 4 iterations

    # Config 2: Inverse strategy
    # Fine features update most often
    # config.step_lora1 = 4
    # config.step_lora2 = 2
    # config.step_lora3 = 1

    # Config 3: Sparse updates
    # All LoRAs update infrequently but at different rates
    # config.step_lora1 = 3
    # config.step_lora2 = 5
    # config.step_lora3 = 7

    # Config 4: All same frequency
    # config.step_lora1 = 1
    # config.step_lora2 = 1
    # config.step_lora3 = 1

    print("\n" + "=" * 70)
    print("Iteration-Based Multi-Rank LoRA Training")
    print("=" * 70)
    print(f"\nUpdate Schedule:")
    print(f"  LoRA 1 (rank=4):  iteration % {config.step_lora1} == 0")
    print(f"  LoRA 2 (rank=8):  iteration % {config.step_lora2} == 0")
    print(f"  LoRA 3 (rank=16): iteration % {config.step_lora3} == 0")
    print("=" * 70 + "\n")
    project_name = "tiny-vit-lora-cifar10"  # "'dynamic-lora',
    # Run training
    model, best_acc = main(
        project_name=project_name,
        experiment_name=f'iter_steps_{config.step_lora1}_{config.step_lora2}_{config.step_lora3}',
        config=config,
        use_wandb=True,
        save_dir='checkpoints'
    )

    print(f'\n🎉 Training finished! Best accuracy: {best_acc:.2f}%')
    print(f'   Update steps: {config.step_lora1}, {config.step_lora2}, {config.step_lora3}')
