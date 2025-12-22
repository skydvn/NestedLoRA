"""
Baseline Training Script for TinyViT on CIFAR-10
No LoRA - Just standard fine-tuning of all parameters

This serves as a baseline to compare against:
- Multi-rank LoRA training
- GPM-enhanced sequential LoRA training
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb
from tqdm import tqdm
import argparse
from dataclasses import dataclass
import os


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class BaselineConfig:
    """Configuration for baseline training"""

    # Model
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 2
    num_classes: int = 10
    dropout: float = 0.1

    # Training
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Optimizer
    optimizer: str = 'adamw'  # 'adamw' or 'sgd'
    momentum: float = 0.9  # for SGD

    # Learning rate schedule
    lr_schedule: str = 'cosine'  # 'cosine', 'step', or 'constant'
    lr_min: float = 1e-5

    # Data augmentation
    use_augmentation: bool = True

    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    seed: int = 42

    # Logging
    use_wandb: bool = True
    project_name: str = "tiny-vit-cifar10"
    save_dir: str = 'checkpoints/baseline'
    save_best_only: bool = True
    log_interval: int = 50


# ═══════════════════════════════════════════════════════════════════
# TinyViT Model (Same as before, no LoRA)
# ═══════════════════════════════════════════════════════════════════

class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = (config.img_size // config.patch_size) ** 2

        self.projection = nn.Conv2d(
            3, config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        return x


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads

        assert config.embed_dim % config.num_heads == 0

        self.qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        self.proj = nn.Linear(config.embed_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = torch.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = torch.nn.functional.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.attn = MultiHeadSelfAttention(config)
        self.norm2 = nn.LayerNorm(config.embed_dim)
        self.mlp = MLP(config)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class TinyViT(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Patch embedding
        self.patch_embed = PatchEmbedding(config)

        # Class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Positional embedding
        num_patches = (config.img_size // config.patch_size) ** 2
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, config.embed_dim))

        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(config) for _ in range(config.num_layers)
        ])

        # Classification head
        self.norm = nn.LayerNorm(config.embed_dim)
        self.head = nn.Linear(config.embed_dim, config.num_classes)

        # Initialize weights
        self._init_weights()

    def _init_weights(self):
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits


# ═══════════════════════════════════════════════════════════════════
# Data Loading
# ═══════════════════════════════════════════════════════════════════

def get_dataloaders(config):
    """Get CIFAR-10 train and test dataloaders"""

    # Data augmentation for training
    if config.use_augmentation:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.4914, 0.4822, 0.4465],
                std=[0.2023, 0.1994, 0.2010]
            )
        ])

    # Test transform (no augmentation)
    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.4914, 0.4822, 0.4465],
            std=[0.2023, 0.1994, 0.2010]
        )
    ])

    # Datasets
    train_dataset = datasets.CIFAR10(
        root='./data',
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root='./data',
        train=False,
        download=True,
        transform=test_transform
    )

    # Dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True
    )

    return train_loader, test_loader


# ═══════════════════════════════════════════════════════════════════
# Learning Rate Schedule
# ═══════════════════════════════════════════════════════════════════

def get_lr(epoch, config):
    """Get learning rate for current epoch"""

    # Warmup
    if epoch < config.warmup_epochs:
        return config.learning_rate * (epoch + 1) / config.warmup_epochs

    # After warmup
    if config.lr_schedule == 'cosine':
        # Cosine annealing
        progress = (epoch - config.warmup_epochs) / (config.num_epochs - config.warmup_epochs)
        return config.lr_min + (config.learning_rate - config.lr_min) * \
            0.5 * (1 + torch.cos(torch.tensor(progress * 3.14159)))

    elif config.lr_schedule == 'step':
        # Step decay (divide by 10 at 50% and 75% of training)
        if epoch >= int(0.75 * config.num_epochs):
            return config.learning_rate * 0.01
        elif epoch >= int(0.5 * config.num_epochs):
            return config.learning_rate * 0.1
        else:
            return config.learning_rate

    else:  # constant
        return config.learning_rate


# ═══════════════════════════════════════════════════════════════════
# Training and Evaluation
# ═══════════════════════════════════════════════════════════════════

def train_epoch(model, train_loader, criterion, optimizer, device, epoch, config):
    """Train for one epoch"""
    model.train()

    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(train_loader, desc=f'Epoch {epoch}/{config.num_epochs}')

    for batch_idx, (images, labels) in enumerate(pbar):
        images, labels = images.to(device), labels.to(device)

        # Forward pass
        outputs = model(images)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Statistics
        total_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{100. * correct / total:.2f}%'
        })

        # Log to wandb
        if config.use_wandb and batch_idx % config.log_interval == 0:
            wandb.log({
                'train/batch_loss': loss.item(),
                'train/batch_acc': 100. * correct / total,
                'epoch': epoch
            })

    avg_loss = total_loss / len(train_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


def evaluate(model, test_loader, criterion, device):
    """Evaluate model on test set"""
    model.eval()

    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc='Evaluating'):
            images, labels = images.to(device), labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

    avg_loss = total_loss / len(test_loader)
    accuracy = 100. * correct / total

    return avg_loss, accuracy


# ═══════════════════════════════════════════════════════════════════
# Main Training Loop
# ═══════════════════════════════════════════════════════════════════

def main(config=None, experiment_name=None):
    """Main training function"""

    if config is None:
        config = BaselineConfig()

    # Set random seed
    torch.manual_seed(config.seed)

    # Initialize wandb
    if config.use_wandb:
        run_name = experiment_name if experiment_name else 'baseline'
        wandb.init(
            project=config.project_name,
            name=run_name,
            config=vars(config)
        )

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    print("=" * 70)
    print("BASELINE TINYVIT TRAINING (No LoRA)")
    print("=" * 70)
    print(f"\nConfiguration:")
    print(f"  Model: TinyViT")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"  Parameters: All trainable (full fine-tuning)")
    print(f"\nTraining:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  LR schedule: {config.lr_schedule}")
    print(f"  Device: {config.device}")
    print("=" * 70)

    # Model
    model = TinyViT(config).to(config.device)

    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\nModel Parameters:")
    print(f"  Total: {total_params:,}")
    print(f"  Trainable: {trainable_params:,}")
    print(f"  Training mode: Full fine-tuning")

    # Data
    train_loader, test_loader = get_dataloaders(config)
    print(f"\nDataset:")
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Batches per epoch: {len(train_loader):,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer
    if config.optimizer == 'adamw':
        optimizer = optim.AdamW(
            model.parameters(),
            lr=config.learning_rate,
            weight_decay=config.weight_decay
        )
    else:  # sgd
        optimizer = optim.SGD(
            model.parameters(),
            lr=config.learning_rate,
            momentum=config.momentum,
            weight_decay=config.weight_decay
        )

    # Training loop
    best_acc = 0.0

    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70 + "\n")

    for epoch in range(1, config.num_epochs + 1):
        # Update learning rate
        lr = get_lr(epoch - 1, config)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr

        print(f"\nEpoch {epoch}/{config.num_epochs} | LR: {lr:.6f}")

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer,
            config.device, epoch, config
        )

        # Evaluate
        test_loss, test_acc = evaluate(
            model, test_loader, criterion, config.device
        )

        # Print summary
        print(f"\nEpoch {epoch} Summary:")
        print(f"  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%")
        print(f"  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%")

        # Log to wandb
        if config.use_wandb:
            log_dict = {
                'global_epoch': epoch,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'test_loss': test_loss,
                'test_acc': test_acc,
                'learning_rate': lr,
            }

            wandb.log(log_dict)

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            if config.save_best_only or epoch % 10 == 0:
                save_path = os.path.join(config.save_dir, 'best_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                }, save_path)
                print(f"  ✓ Saved best model (acc={best_acc:.2f}%)")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

    if config.use_wandb:
        wandb.log({'best_test_accuracy': best_acc})
        wandb.finish()

    return model, best_acc


# ═══════════════════════════════════════════════════════════════════
# Command Line Interface
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TinyViT baseline on CIFAR-10')

    # Model
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=4)

    # Training
    parser.add_argument('--epochs', type=int, default=600)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)

    # Optimizer
    parser.add_argument('--optimizer', type=str, default='adamw', choices=['adamw', 'sgd'])
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'step', 'constant'])

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--name', type=str, default='baseline', help='Experiment name')

    args = parser.parse_args()

    # Create config from args
    config = BaselineConfig()
    config.embed_dim = args.embed_dim
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.optimizer = args.optimizer
    config.lr_schedule = args.lr_schedule
    config.device = args.device
    config.seed = args.seed
    config.use_wandb = not args.no_wandb

    # Train
    model, best_acc = main(config=config, experiment_name=args.name)

    print(f"\n🎉 Final Best Accuracy: {best_acc:.2f}%")