"""
TinyViT with LoRA Training - Complete Metrics Tracking with Weights & Biases

Features:
- Tracks all training metrics (loss, accuracy, learning rate)
- Logs to wandb for experiment tracking
- Saves checkpoints with metrics
- Visualizes training progress
- Supports model comparison
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import math
import json
import os
from datetime import datetime

# Import wandb
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")

# Import the original TinyViT components
from models.vit_small import (
    TinyViT, TinyViTConfig, PatchEmbedding,
    TransformerBlock
)
from trainer import get_data_loaders


# ============================================================================
# LoRA Components (Same as before)
# ============================================================================

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / rank

        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(lora_dropout)

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        if x.device != self.lora_A.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)

        dropout_x = self.lora_dropout(x)
        lora_out = (dropout_x @ self.lora_A) @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation"""

    def __init__(
            self,
            original_linear: nn.Linear,
            rank: int = 8,
            lora_alpha: int = 16,
            lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_linear = original_linear
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        return self.original_linear(x) + self.lora(x)


class LoRAMultiHeadSelfAttention(nn.Module):
    """MultiHeadSelfAttention with LoRA"""

    def __init__(self, config, rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads

        base_qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        base_proj = nn.Linear(config.embed_dim, config.embed_dim)

        self.qkv = LoRALinear(base_qkv, rank, lora_alpha, lora_dropout)
        self.proj = LoRALinear(base_proj, rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        return x


class LoRAMLP(nn.Module):
    """MLP with LoRA"""

    def __init__(self, config, rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)

        base_fc1 = nn.Linear(config.embed_dim, hidden_dim)
        base_fc2 = nn.Linear(hidden_dim, config.embed_dim)

        self.fc1 = LoRALinear(base_fc1, rank, lora_alpha, lora_dropout)
        self.fc2 = LoRALinear(base_fc2, rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class LoRATransformerBlock(nn.Module):
    """TransformerBlock with LoRA"""

    def __init__(self, config, rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = LoRAMultiHeadSelfAttention(config, rank, lora_alpha, lora_dropout)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = LoRAMLP(config, rank, lora_alpha, lora_dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViTLoRA(nn.Module):
    """TinyViT with LoRA adaptation"""

    def __init__(
            self,
            config,
            rank=8,
            lora_alpha=16,
            lora_dropout=0.1,
    ):
        super().__init__()
        self.config = config
        self.rank = rank
        self.lora_alpha = lora_alpha

        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        self.blocks = nn.ModuleList([
            LoRATransformerBlock(config, rank, lora_alpha, lora_dropout)
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        base_head = nn.Linear(config.embed_dim, config.num_classes)
        self.head = LoRALinear(base_head, rank, lora_alpha, lora_dropout)

        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)
        return logits


# ============================================================================
# Metrics Tracker
# ============================================================================

class MetricsTracker:
    """Track and save training metrics"""

    def __init__(self, save_dir='./metrics'):
        self.save_dir = save_dir
        os.makedirs(save_dir, exist_ok=True)

        self.metrics = {
            'train_loss': [],
            'train_acc': [],
            'test_loss': [],
            'test_acc': [],
            'learning_rate': [],
            'epoch': []
        }

    def update(self, epoch, train_loss, train_acc, test_loss, test_acc, lr):
        """Update metrics for current epoch"""
        self.metrics['epoch'].append(epoch)
        self.metrics['train_loss'].append(train_loss)
        self.metrics['train_acc'].append(train_acc)
        self.metrics['test_loss'].append(test_loss)
        self.metrics['test_acc'].append(test_acc)
        self.metrics['learning_rate'].append(lr)

    def save(self, filename='metrics.json'):
        """Save metrics to JSON file"""
        path = os.path.join(self.save_dir, filename)
        with open(path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"Metrics saved to {path}")

    def get_best_accuracy(self):
        """Get best test accuracy"""
        if self.metrics['test_acc']:
            return max(self.metrics['test_acc'])
        return 0.0


# ============================================================================
# Training Functions with Metrics
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch, use_wandb=True):
    """Train for one epoch and return metrics"""
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

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

        # Log to wandb every 50 batches
        if use_wandb and WANDB_AVAILABLE and batch_idx % 50 == 0:
            wandb.log({
                'batch_train_loss': loss.item(),
                'batch_train_acc': 100. * correct / total,
                'batch': epoch * len(loader) + batch_idx
            })

    avg_loss = total_loss / len(loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, use_wandb=True):
    """Evaluate model and return metrics"""
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


def print_trainable_parameters(model):
    """Print trainable parameters statistics"""
    trainable_params = 0
    all_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"\n{'=' * 60}")
    print(f"Parameter Statistics:")
    print(f"{'=' * 60}")
    print(f"Trainable params:  {trainable_params:,}")
    print(f"All params:        {all_params:,}")
    print(f"Trainable %:       {100 * trainable_params / all_params:.2f}%")
    print(f"{'=' * 60}\n")

    return trainable_params, all_params


# ============================================================================
# Main Training Function with W&B
# ============================================================================

def main(
        project_name='tiny-vit-lora',
        experiment_name=None,
        rank=8,
        lora_alpha=16,
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=50,
        use_wandb=True,
        save_dir='./checkpoints'
):
    """
    Main training function with wandb integration

    Args:
        project_name: W&B project name
        experiment_name: W&B run name (default: auto-generated)
        rank: LoRA rank
        lora_alpha: LoRA alpha scaling
        learning_rate: Learning rate
        batch_size: Batch size
        num_epochs: Number of epochs
        use_wandb: Whether to use wandb
        save_dir: Directory to save checkpoints
    """

    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Create experiment name if not provided
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'lora_r{rank}_lr{learning_rate}_{timestamp}'

    # Configuration
    config = TinyViTConfig()

    # Initialize wandb
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'architecture': 'TinyViT',
                'dataset': 'CIFAR-10',
                'lora_rank': rank,
                'lora_alpha': lora_alpha,
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'optimizer': 'AdamW',
                'scheduler': 'CosineAnnealingLR',
                'embed_dim': config.embed_dim,
                'num_heads': config.num_heads,
                'num_layers': config.num_layers,
                'patch_size': config.patch_size,
            }
        )
        print(f"\n✓ Weights & Biases initialized")
        print(f"  Project: {project_name}")
        print(f"  Run: {experiment_name}")
        print(f"  URL: {wandb.run.url}\n")
    elif use_wandb and not WANDB_AVAILABLE:
        print("\n⚠️  W&B requested but not installed. Install with: pip install wandb")
        use_wandb = False

    # Create model
    print(f"\n{'=' * 60}")
    print(f"Creating TinyViT with LoRA")
    print(f"{'=' * 60}")

    model = TinyViTLoRA(
        config,
        rank=rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1
    ).to(device)

    # Print and log parameters
    trainable_params, all_params = print_trainable_parameters(model)

    if use_wandb and WANDB_AVAILABLE:
        wandb.config.update({
            'trainable_params': trainable_params,
            'total_params': all_params,
            'trainable_percent': 100 * trainable_params / all_params
        })

    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=batch_size)

    # Training setup
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=learning_rate,
        weight_decay=0.01
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

    # Initialize metrics tracker
    metrics_tracker = MetricsTracker(save_dir=save_dir)

    # Training loop
    best_acc = 0

    print(f"\n{'=' * 60}")
    print(f"Starting Training")
    print(f"{'=' * 60}\n")

    for epoch in range(1, num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{num_epochs} ---')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch, use_wandb
        )

        # Evaluate
        test_loss, test_acc, per_class_acc = evaluate(
            model, test_loader, criterion, device, use_wandb
        )

        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']

        # Update metrics tracker
        metrics_tracker.update(epoch, train_loss, train_acc, test_loss, test_acc, current_lr)

        # Log to wandb
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'learning_rate': current_lr,
            }
            # Add per-class accuracy
            log_dict.update(per_class_acc)
            wandb.log(log_dict)

        # Step scheduler
        scheduler.step()

        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')
        print(f'  LR: {current_lr:.6f}')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc

            # Save checkpoint
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'best_acc': best_acc,
                'config': {
                    'rank': rank,
                    'lora_alpha': lora_alpha,
                    'learning_rate': learning_rate,
                }
            }

            checkpoint_path = os.path.join(save_dir, 'best_model.pth')
            torch.save(checkpoint, checkpoint_path)

            # Save LoRA weights only
            lora_state = {k: v for k, v in model.state_dict().items() if 'lora' in k}
            lora_path = os.path.join(save_dir, 'best_lora_weights.pth')
            torch.save(lora_state, lora_path)

            print(f'  ✓ Saved best model: {best_acc:.2f}%')

            # Log best model to wandb
            if use_wandb and WANDB_AVAILABLE:
                wandb.run.summary['best_accuracy'] = best_acc
                wandb.run.summary['best_epoch'] = epoch

        # Save checkpoint every 10 epochs
        if epoch % 10 == 0:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'test_acc': test_acc,
            }, checkpoint_path)
            print(f'  ✓ Saved checkpoint at epoch {epoch}')

    # Save final metrics
    metrics_tracker.save()

    # Final summary
    print(f'\n{"=" * 60}')
    print(f'Training Complete!')
    print(f'{"=" * 60}')
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'Checkpoints saved to: {save_dir}')
    print(f'Metrics saved to: {os.path.join(save_dir, "metrics.json")}')

    if use_wandb and WANDB_AVAILABLE:
        print(f'W&B Dashboard: {wandb.run.url}')

        # Save final model to wandb
        artifact = wandb.Artifact('tiny-vit-lora-model', type='model')
        artifact.add_file(os.path.join(save_dir, 'best_model.pth'))
        artifact.add_file(os.path.join(save_dir, 'best_lora_weights.pth'))
        wandb.log_artifact(artifact)

        wandb.finish()

    print(f'{"=" * 60}\n')

    return model, metrics_tracker


if __name__ == '__main__':
    # Example usage with different configurations
    model = "tinyvit"
    dataset = "cifar10"
    rank = 32
    # Configuration 1: Default
    model, metrics = main(
        project_name='tiny-vit-lora-cifar10',
        experiment_name=f'{model}_{dataset}_{rank}',
        rank=rank,
        lora_alpha=16,
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=50,
        use_wandb=True
    )

    # Uncomment to try different configurations:

    # Configuration 2: Larger rank
    # model, metrics = main(
    #     experiment_name='large_rank',
    #     rank=16,
    #     lora_alpha=32,
    # )

    # Configuration 3: Smaller rank
    # model, metrics = main(
    #     experiment_name='small_rank',
    #     rank=4,
    #     lora_alpha=8,
    # )