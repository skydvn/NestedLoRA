"""
TinyViT Training with Dynamic LoRA Rank

Train with LoRA rank that changes during training based on a schedule.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import os
import json
from datetime import datetime

# Import the original TinyViT components
from models.vit_small import (
    TinyViT, TinyViTConfig, PatchEmbedding,
    TransformerBlock
)
from trainer import get_data_loaders

from utils.dynamic_lora import (
    apply_dynamic_lora, RankScheduler, update_model_rank,
    print_rank_info, DynamicLoRALinear
)
from utils.experiment_naming import create_experiment_name

# Try to import wandb
try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False
    print("⚠️  wandb not available - training will continue with local metrics only")


class DynamicLoRATracker:
    """Track dynamic LoRA rank changes and metrics"""

    def __init__(self, save_dir='./metrics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics = {
            'epoch': [],
            'current_rank': [],
            'active_params': [],
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rate': []
        }

    def update(self, epoch, current_rank, active_params, train_loss, train_acc,
               test_loss, test_acc, lr):
        """Update metrics"""
        self.metrics['epoch'].append(epoch)
        self.metrics['current_rank'].append(current_rank)
        self.metrics['active_params'].append(active_params)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['learning_rate'].append(lr)

    def save(self, filename='dynamic_lora_metrics.json'):
        """Save metrics to JSON"""
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"✓ Metrics saved to {path}")

    def get_rank_changes(self):
        """Get list of when rank changed"""
        changes = []
        prev_rank = None
        for i, rank in enumerate(self.metrics['current_rank']):
            if rank != prev_rank:
                changes.append({
                    'epoch': self.metrics['epoch'][i],
                    'old_rank': prev_rank,
                    'new_rank': rank
                })
                prev_rank = rank
        return changes


def get_current_rank(model):
    """Get current rank from model"""
    for module in model.modules():
        if isinstance(module, DynamicLoRALinear):
            return module.lora.current_rank
    return 0


def get_active_lora_params(model):
    """Get number of active LoRA parameters"""
    total = 0
    for module in model.modules():
        if isinstance(module, DynamicLoRALinear):
            total += module.lora.get_active_parameters()
    return total


def train_epoch(model, loader, optimizer, criterion, device, epoch, use_wandb=False):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} - Training')
    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%'
        })

        # # Log to wandb
        # if use_wandb and WANDB_AVAILABLE and batch_idx % 50 == 0:
        #     try:
        #         wandb.log({
        #             'batch_train_loss': loss.item(),
        #             'batch_train_acc': 100. * correct / total,
        #         }, step=epoch * len(loader) + batch_idx)
        #     except:
        #         pass

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
    """Evaluate model"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    return total_loss / len(loader), 100. * correct / total


def main(
    # LoRA configuration
    initial_rank=4,
    max_rank=32,
    final_rank=16,
    lora_alpha=16,
    rank_strategy='progressive_growth',
    num_rank_stages=8,

    # Training configuration
    learning_rate=1e-3,
    batch_size=128,
    num_epochs=100,

    # W&B configuration
    use_wandb=True,
    project_name= "tiny-vit-lora-cifar10", #"'dynamic-lora',

    # Other
    save_dir='./checkpoints_dynamic'
):
    """
    Train with dynamic LoRA rank

    Args:
        initial_rank: Starting LoRA rank
        max_rank: Maximum LoRA rank (pre-allocate this much)
        final_rank: Target final rank
        lora_alpha: LoRA scaling factor
        rank_strategy: 'progressive_growth', 'progressive_shrinkage', 'cyclic', 'warm_start'
        num_rank_stages: Number of rank change stages
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Total training epochs
        use_wandb: Whether to use W&B
        project_name: W&B project name
        save_dir: Directory to save checkpoints
    """

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Generate experiment name
    experiment_name = create_experiment_name(
        model_name='tinyvit',
        dataset_name='cifar10',
        rank=initial_rank,  # Use initial rank in name
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        format='detailed',
        include_timestamp=True
    )
    # Add dynamic indicator
    experiment_name = f"dynamic_{rank_strategy}_{experiment_name}"

    print(f"\n{'='*70}")
    print(f"Dynamic LoRA Training")
    print(f"{'='*70}")
    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"Strategy: {rank_strategy}")
    print(f"Rank: {initial_rank} → {final_rank}")

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config={
                    'architecture': 'TinyViT',
                    'dataset': 'CIFAR-10',
                    'lora_type': 'dynamic',
                    'initial_rank': initial_rank,
                    'max_rank': max_rank,
                    'final_rank': final_rank,
                    'rank_strategy': rank_strategy,
                    'num_rank_stages': num_rank_stages,
                    'lora_alpha': lora_alpha,
                    'learning_rate': learning_rate,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                }
            )
            print(f"\n✓ W&B initialized: {wandb.run.url}")
            use_wandb = True
        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            use_wandb = False
    else:
        use_wandb = False

    # Create model
    print(f"\n{'='*70}")
    print("Creating TinyViT with Dynamic LoRA")
    print(f"{'='*70}")

    config = TinyViTConfig()
    base_model = TinyViT(config).to(device)

    # Apply dynamic LoRA
    model = apply_dynamic_lora(
        base_model,
        initial_rank=initial_rank,
        max_rank=max_rank,
        lora_alpha=lora_alpha,
        target_modules=["qkv", "proj", "fc1", "fc2"]
    )

    # Print initial rank info
    print_rank_info(model)

    # Create rank scheduler
    rank_scheduler = RankScheduler(
        initial_rank=initial_rank,
        final_rank=final_rank,
        strategy=rank_strategy,
        num_stages=num_rank_stages,
        total_epochs=num_epochs
    )

    print(f"\n{rank_scheduler.get_schedule_info()}\n")

    # Get data
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Metrics tracker
    tracker = DynamicLoRATracker(save_dir=save_dir)

    # Training loop
    best_acc = 0

    print(f"\n{'='*70}")
    print(f"Starting Training")
    print(f"{'='*70}\n")

    for epoch in range(1, num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{num_epochs} ---')

        # Check if rank should change
        if rank_scheduler.should_change_rank(epoch):
            new_rank = rank_scheduler.get_rank(epoch)
            count, changes = update_model_rank(model, new_rank)

            print(f"\n🔄 RANK CHANGED at Epoch {epoch}")
            print(f"  Updated {count} modules to rank {new_rank}")
            for change in changes[:3]:  # Show first 3
                print(f"    {change['module']}: {change['old_rank']} → {change['new_rank']}")

            # Log to wandb
            if use_wandb:
                try:
                    wandb.log({
                        'rank_change': new_rank,
                        'epoch': epoch
                    })
                except:
                    pass

        # Get current state
        current_rank = get_current_rank(model)
        active_params = get_active_lora_params(model)

        print(f"Current LoRA rank: {current_rank}")
        print(f"Active LoRA params: {active_params:,}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_wandb
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Get learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update tracker
        tracker.update(
            epoch, current_rank, active_params,
            train_loss, train_acc, test_loss, test_acc, current_lr
        )

        # Log to wandb
        if use_wandb:
            try:
                wandb.log({
                    'epoch': epoch,
                    'current_rank': current_rank,
                    'active_lora_params': active_params,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'learning_rate': current_lr,
                })
            except:
                pass

        # Step scheduler
        lr_scheduler.step()

        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Rank: {current_rank}, Active Params: {active_params:,}')
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_acc': best_acc,
                'current_rank': current_rank,
                'rank_scheduler': rank_scheduler.rank_schedule,
            }

            torch.save(checkpoint, os.path.join(save_dir, 'best_model_dynamic.pth'))
            print(f'  ✓ Saved best model: {best_acc:.2f}%')

            if use_wandb:
                try:
                    wandb.run.summary['best_accuracy'] = best_acc
                    wandb.run.summary['best_epoch'] = epoch
                except:
                    pass

    # Save metrics
    tracker.save()

    # Show rank changes
    print(f"\n{'='*70}")
    print("RANK CHANGES DURING TRAINING")
    print(f"{'='*70}")

    changes = tracker.get_rank_changes()
    for change in changes:
        print(f"Epoch {change['epoch']:3d}: "
              f"{change['old_rank']} → {change['new_rank']}")

    # Final summary
    print(f"\n{'='*70}")
    print(f'Training Complete!')
    print(f"{'='*70}")
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'Final Rank: {get_current_rank(model)}')
    print(f'Strategy: {rank_strategy}')
    print(f'Checkpoints: {save_dir}')
    print(f"{'='*70}\n")

    if use_wandb:
        try:
            wandb.finish()
        except:
            pass

    return model, tracker


if __name__ == '__main__':
    # # Example 1: Progressive Growth (recommended)
    # print("="*70)
    # print("EXAMPLE 1: PROGRESSIVE GROWTH")
    # print("="*70)
    # print("Start with small rank, gradually increase")
    # print("Good for: Better convergence, curriculum learning")
    #
    # model, tracker = main(
    #     initial_rank=4,
    #     max_rank=32,
    #     final_rank=16,
    #     rank_strategy='progressive_growth',
    #     num_rank_stages=4,
    #     num_epochs=50,
    #     use_wandb=True
    # )

    # Uncomment to try other strategies:

    # Example 2: Progressive Shrinkage
    # print("\n\nEXAMPLE 2: PROGRESSIVE SHRINKAGE")
    # model, tracker = main(
    #     initial_rank=16,
    #     final_rank=4,
    #     rank_strategy='progressive_shrinkage',
    #     num_epochs=50
    # )

    # Example 3: Cyclic
    # print("\n\nEXAMPLE 3: CYCLIC RANKS")
    # model, tracker = main(
    #     initial_rank=4,
    #     final_rank=16,
    #     rank_strategy='cyclic',
    #     num_epochs=50
    # )

    # Example 4: Exponential Growth (recommended)
    print("="*70)
    print("EXAMPLE 1: EXPONENTIAL GROWTH")
    print("="*70)
    print("Start with small rank, gradually increase")
    print("Good for: Better convergence, curriculum learning")

    model, tracker = main(
        initial_rank=4,
        max_rank=32,
        final_rank=32,
        rank_strategy='progressive_growth',
        num_rank_stages=5,
        num_epochs=100,
        use_wandb=True
    )