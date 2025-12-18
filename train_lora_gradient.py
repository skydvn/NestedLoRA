"""
LoRA as Gradient Generator

Process:
1. Train LoRA: A, B = argmin_{A,B} Loss((W + A@B)·x; D)
   - Forward includes LoRA: output = W·x + (A@B)·x
   - Loss computed on full model: W + A@B
2. Compute gradient: ΔW = A @ B
3. Update base: W_{t+1} = W_t - η * ΔW
4. Reset LoRA: A, B ← 0

Key Innovation:
- LoRA learns optimal update via: min Loss((W + A@B)·x; D)
- Then update is applied: W ← W - η·(A@B)
- LoRA reset and learns next update

This is different from traditional LoRA where A,B stay fixed!
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
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

from utils.lora_gradient_generator import LoRAGradientModel

try:
    import wandb
    WANDB_AVAILABLE = True
except:
    WANDB_AVAILABLE = False


def train_epoch(
    model,
    loader,
    lora_optimizer,
    criterion,
    device,
    epoch,
    base_update_lr=1e-4,
    inner_steps=5,  # Number of gradient steps for A, B
    apply_every_n_batches=1,
):
    """
    Train one epoch with LoRA gradient generator

    Process:
    - Inner loop: Optimize A, B over multiple gradient steps
    - Outer loop: Apply learned gradient to base, then reset

    Args:
        model: LoRAGradientModel
        loader: DataLoader
        lora_optimizer: Optimizer for LoRA (A, B)
        criterion: Loss function
        device: Device
        epoch: Current epoch
        base_update_lr: Learning rate for base model update (η)
        inner_steps: Number of gradient steps for optimizing A, B
        apply_every_n_batches: Apply LoRA gradient every N batches
    """
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # ═══════════════════════════════════════════════════════
        # INNER LOOP: Optimize A, B (multiple gradient steps)
        # ═══════════════════════════════════════════════════════
        for inner_step in range(inner_steps):
            # Forward with LoRA
            outputs = model(images)  # W·x + (A@B)·x
            loss = criterion(outputs, labels)

            # Backward
            lora_optimizer.zero_grad()
            loss.backward()

            # Update A, B
            lora_optimizer.step()

            # Track metrics (only last inner step)
            if inner_step == inner_steps - 1:
                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

        # ═══════════════════════════════════════════════════════
        # OUTER LOOP: Apply LoRA gradient to base model
        # ═══════════════════════════════════════════════════════
        if (batch_idx + 1) % apply_every_n_batches == 0:
            # After inner_steps, A and B are optimized
            # Apply to base: W ← W - η * (A @ B)
            model.apply_all_lora_gradients(learning_rate=base_update_lr)
            # Note: apply_all_lora_gradients also resets A, B to zero

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100.*correct/total:.2f}%',
            'inner': inner_steps
        })

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
    initial_rank=8,
    max_rank=32,
    lora_alpha=16,

    # Learning rates
    lora_lr=1e-3,          # LR for LoRA matrices (A, B)
    base_update_lr=1e-4,   # LR for base model update (η)

    # Training configuration
    batch_size=128,
    num_epochs=100,
    inner_steps=5,  # Number of gradient steps for A, B per batch
    apply_every_n_batches=1,  # Apply gradient every N batches

    # W&B configuration
    use_wandb=True,
    project_name='lora-gradient-generator',

    # Other
    save_dir='./checkpoints_lora_gradient'
):
    """
    Train with LoRA as gradient generator

    Process:
    - Inner loop: Optimize A, B with multiple gradient steps
    - Outer loop: Apply optimized gradient to W, then reset

    Args:
        initial_rank: LoRA rank
        max_rank: Maximum LoRA rank
        lora_alpha: LoRA scaling factor
        lora_lr: Learning rate for LoRA matrices
        base_update_lr: Learning rate for base model update (η)
        batch_size: Batch size
        num_epochs: Number of epochs
        inner_steps: Number of gradient steps to optimize A, B
        apply_every_n_batches: Apply LoRA gradient every N batches
        use_wandb: Use W&B
        project_name: W&B project name
        save_dir: Save directory
    """

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Experiment name
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    experiment_name = f'lora_grad_r{initial_rank}_llr{lora_lr}_blr{base_update_lr}_{timestamp}'

    print("="*80)
    print("TRAINING WITH LoRA AS GRADIENT GENERATOR")
    print("="*80)
    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"\nMethod:")
    print(f"  1. Train LoRA: min_{{A,B}} Loss")
    print(f"  2. Compute gradient: ΔW = (A @ B)")
    print(f"  3. Update base: W ← W - η * ΔW")
    print(f"  4. Reset LoRA: A, B ← 0")
    print(f"\nKey: NO unmerge/merge, NO LoRA addition!")
    print("="*80)

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        try:
            wandb.init(
                project=project_name,
                name=experiment_name,
                config={
                    'architecture': 'TinyViT',
                    'dataset': 'CIFAR-10',
                    'method': 'lora_gradient_generator',
                    'initial_rank': initial_rank,
                    'max_rank': max_rank,
                    'lora_alpha': lora_alpha,
                    'lora_lr': lora_lr,
                    'base_update_lr': base_update_lr,
                    'batch_size': batch_size,
                    'num_epochs': num_epochs,
                    'apply_every_n_batches': apply_every_n_batches,
                }
            )
            print(f"\n✓ W&B initialized: {wandb.run.url}")
            use_wandb = True
        except Exception as e:
            print(f"\n⚠️  W&B failed: {e}")
            use_wandb = False
    else:
        use_wandb = False

    # Create model
    print("\n" + "="*80)
    print("Creating Model")
    print("="*80)

    config = TinyViTConfig()
    base_model = TinyViT(config).to(device)

    # Wrap with LoRA gradient generators
    model = LoRAGradientModel(
        base_model,
        initial_rank=initial_rank,
        max_rank=max_rank,
        lora_alpha=lora_alpha,
        lora_lr=base_update_lr
    ).to(device)

    # Count parameters
    lora_params = model.get_lora_parameters()
    lora_param_count = sum(p.numel() for p in lora_params)
    base_params = sum(p.numel() for p in model.base_model.parameters())

    print(f"\n✓ Model created")
    print(f"  LoRA modules: {len(model.lora_modules)}")
    print(f"  LoRA parameters: {lora_param_count:,}")
    print(f"  Base parameters: {base_params:,}")
    print(f"  LoRA trainable: {100*lora_param_count/base_params:.2f}%")

    # Get data
    print("\nLoading CIFAR-10...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Training setup
    criterion = nn.CrossEntropyLoss()

    # Optimizer for LoRA parameters only
    lora_optimizer = torch.optim.AdamW(lora_params, lr=lora_lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(lora_optimizer, T_max=num_epochs)

    print(f"\n✓ Training setup")
    print(f"  LoRA optimizer: AdamW (lr={lora_lr})")
    print(f"  Base update lr: {base_update_lr}")
    print(f"  Apply gradient every: {apply_every_n_batches} steps")

    # Metrics
    metrics = {
        'epoch': [],
        'train_loss': [],
        'train_acc': [],
        'test_loss': [],
        'test_acc': [],
        'lora_lr': [],
    }

    # Training loop
    best_acc = 0

    print("\n" + "="*80)
    print("TRAINING START")
    print("="*80)

    for epoch in range(1, num_epochs + 1):
        print(f'\n{"="*80}')
        print(f'EPOCH {epoch}/{num_epochs}')
        print(f'{"="*80}')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, lora_optimizer, criterion, device, epoch,
            base_update_lr=base_update_lr,
            inner_steps=inner_steps,
            apply_every_n_batches=apply_every_n_batches
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Get LR
        current_lora_lr = lora_optimizer.param_groups[0]['lr']

        # Store metrics
        metrics['epoch'].append(epoch)
        metrics['train_loss'].append(train_loss)
        metrics['train_acc'].append(train_acc)
        metrics['test_loss'].append(test_loss)
        metrics['test_acc'].append(test_acc)
        metrics['lora_lr'].append(current_lora_lr)

        # Log to W&B
        if use_wandb:
            try:
                wandb.log({
                    'epoch': epoch,
                    'train_loss': train_loss,
                    'train_acc': train_acc,
                    'test_loss': test_loss,
                    'test_acc': test_acc,
                    'lora_lr': current_lora_lr,
                    'base_update_lr': base_update_lr,
                })
            except:
                pass

        # Step scheduler
        scheduler.step()

        # Print summary
        print(f'\n{"─"*80}')
        print(f'EPOCH {epoch} SUMMARY')
        print(f'{"─"*80}')
        print(f'Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
        print(f'LoRA LR: {current_lora_lr:.6f}')
        print(f'Base LR: {base_update_lr:.6f}')
        print(f'{"─"*80}')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.base_model.state_dict(),  # Save base model
                'best_acc': best_acc,
            }

            torch.save(checkpoint, os.path.join(save_dir, 'best_model_lora_grad.pth'))
            print(f'\n✅ BEST MODEL SAVED: {best_acc:.2f}%')

            if use_wandb:
                try:
                    wandb.run.summary['best_accuracy'] = best_acc
                    wandb.run.summary['best_epoch'] = epoch
                except:
                    pass

    # Save metrics
    metrics_path = os.path.join(save_dir, 'lora_gradient_metrics.json')
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print(f"\n{'='*80}")
    print("TRAINING COMPLETE!")
    print(f"{'='*80}")
    print(f"Best Test Accuracy: {best_acc:.2f}%")
    print(f"Checkpoints saved to: {save_dir}")
    print(f"Metrics saved to: {metrics_path}")

    print(f"\n{'='*80}")
    print("METHOD SUMMARY")
    print(f"{'='*80}")
    print("""
LoRA as Gradient Generator:

Process (each training step):
  1. Forward: output = W·x (no LoRA)
  2. Backward: compute ∇A, ∇B
  3. Update LoRA: A ← A - lr·∇A, B ← B - lr·∇B
  4. Generate gradient: ΔW = (A @ B)
  5. Update base: W ← W - η·ΔW
  6. Reset: A ← 0, B ← 0

Key Innovation:
  ✓ LoRA learns UPDATE DIRECTION
  ✓ Base model updated DIRECTLY
  ✓ No unmerge/merge cycle
  ✓ No LoRA addition in forward
  ✓ LoRA acts as learned optimizer
    """)

    if use_wandb:
        try:
            wandb.finish()
        except:
            pass

    return model, metrics


if __name__ == '__main__':
    # Train with LoRA gradient generator
    model, metrics = main(
        initial_rank=8,
        max_rank=32,
        lora_alpha=16,
        lora_lr=1e-3,          # LR for LoRA
        base_update_lr=1e-4,   # LR for base model
        batch_size=128,
        num_epochs=100,
        apply_every_n_batches=1,
        use_wandb=True,
        project_name='lora-gradient-generator'
    )