"""
LoRA Training Script for TinyViT on CIFAR-10 with Merge

This script:
1. Adds LoRA adapters to the model
2. Trains only the LoRA parameters (frozen base model)
3. Merges LoRA weights back into the base model
4. Saves the merged model for efficient inference
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
import math


# ═══════════════════════════════════════════════════════════════════
# Configuration
# ═══════════════════════════════════════════════════════════════════

@dataclass
class LoRAConfig:
    """Configuration for LoRA training"""

    # Model
    img_size: int = 32
    patch_size: int = 4
    embed_dim: int = 128
    num_heads: int = 4
    num_layers: int = 4
    mlp_ratio: int = 2
    num_classes: int = 10
    dropout: float = 0.1

    # LoRA parameters
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_dropout: float = 0.1
    target_modules: list = None  # Will be set in __post_init__

    # Training
    batch_size: int = 128
    num_epochs: int = 100
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    warmup_epochs: int = 5

    # Optimizer
    optimizer: str = 'adamw'

    # Learning rate schedule
    lr_schedule: str = 'cosine'
    lr_min: float = 1e-5

    # Data augmentation
    use_augmentation: bool = True

    # System
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    num_workers: int = 4
    seed: int = 42

    # Logging
    use_wandb: bool = True
    project_name: str = "tiny-vit-lora-cifar10"
    save_dir: str = 'checkpoints/lora'
    save_best_only: bool = True
    log_interval: int = 50

    def __post_init__(self):
        if self.target_modules is None:
            # Apply LoRA to all linear layers in attention
            self.target_modules = ['qkv', 'proj']


# ═══════════════════════════════════════════════════════════════════
# LoRA Layer
# ═══════════════════════════════════════════════════════════════════

class LoRALayer(nn.Module):
    """
    LoRA (Low-Rank Adaptation) layer

    Replaces W with W + (B @ A) * (alpha / rank)
    where A is (in_features, rank) and B is (rank, out_features)
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices - registered as parameters so they move with the module
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize - this happens on CPU, will be moved by .to() call
        self.reset_parameters()

    def reset_parameters(self):
        """Initialize LoRA parameters"""
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """Apply LoRA transformation"""
        # Both x and parameters should be on same device due to nn.Parameter
        result = (x @ self.lora_A) @ self.lora_B
        result = self.dropout(result)
        return result * self.scaling

    def forward(self, x):
        """Apply LoRA transformation"""
        # x @ A @ B with scaling
        result = (x @ self.lora_A) @ self.lora_B
        result = self.dropout(result)
        return result * self.scaling


class LinearWithLoRA(nn.Module):
    """Linear layer with LoRA adapter"""

    def __init__(
        self,
        linear: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0
    ):
        super().__init__()
        self.linear = linear

        # Create LoRA layer on the same device as the linear layer
        self.lora = LoRALayer(
            linear.in_features,
            linear.out_features,
            rank=rank,
            alpha=alpha,
            dropout=dropout
        )

        # Move LoRA to the same device as linear layer
        if linear.weight.is_cuda:
            self.lora = self.lora.to(linear.weight.device)

        # Freeze the original linear layer
        self.linear.weight.requires_grad = False
        if self.linear.bias is not None:
            self.linear.bias.requires_grad = False

    def forward(self, x):
        """Forward pass: original + LoRA"""
        return self.linear(x) + self.lora(x)

    def merge(self):
        """Merge LoRA weights into the base linear layer"""
        # Compute merged weight: W_new = W + (B @ A) * scaling
        with torch.no_grad():
            lora_weight = (self.lora.lora_A @ self.lora.lora_B.T) * self.lora.scaling
            self.linear.weight.data += lora_weight.T

    def unmerge(self):
        """Unmerge LoRA weights from the base linear layer"""
        with torch.no_grad():
            lora_weight = (self.lora.lora_A @ self.lora.lora_B.T) * self.lora.scaling
            self.linear.weight.data -= lora_weight.T


# ═══════════════════════════════════════════════════════════════════
# TinyViT Model with LoRA Support
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
# LoRA Utilities
# ═══════════════════════════════════════════════════════════════════

def add_lora_to_model(model, config):
    """Add LoRA adapters to specified modules in the model"""

    lora_count = 0
    model_device = next(model.parameters()).device

    for name, module in model.named_modules():
        # Check if this is a target module
        if isinstance(module, nn.Linear):
            # Get the parent module and attribute name
            parent_name = '.'.join(name.split('.')[:-1])
            attr_name = name.split('.')[-1]

            # Check if this linear layer should have LoRA
            if attr_name in config.target_modules:
                # Get parent module
                parent = model
                for part in parent_name.split('.'):
                    if part:
                        parent = getattr(parent, part)

                # Replace with LoRA version
                lora_layer = LinearWithLoRA(
                    module,
                    rank=config.lora_rank,
                    alpha=config.lora_alpha,
                    dropout=config.lora_dropout
                )

                # Ensure LoRA layer is on the correct device
                lora_layer = lora_layer.to(model_device)

                setattr(parent, attr_name, lora_layer)
                lora_count += 1

    print(f"\n✓ Added LoRA to {lora_count} linear layers")
    print(f"  All LoRA parameters on device: {model_device}")
    return model


def freeze_non_lora_params(model):
    """Freeze all parameters except LoRA parameters"""

    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()

        # Only LoRA parameters should be trainable
        if 'lora_' in name:
            param.requires_grad = True
            trainable_params += param.numel()
            lora_params += param.numel()
        else:
            param.requires_grad = False

    print(f"\nParameter Summary:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  LoRA parameters: {lora_params:,}")
    print(f"  Trainable: {trainable_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Frozen: {total_params - trainable_params:,} ({100 * (total_params - trainable_params) / total_params:.2f}%)")

    return trainable_params, total_params


def merge_lora_weights(model):
    """Merge all LoRA weights into the base model"""

    merged_count = 0

    for name, module in model.named_modules():
        if isinstance(module, LinearWithLoRA):
            module.merge()
            merged_count += 1

    print(f"\n✓ Merged LoRA weights from {merged_count} layers into base model")


def get_merged_model(model, config):
    """
    Create a clean model with LoRA weights merged in
    (no LoRA layers, just standard linear layers)
    """

    # Create a fresh model
    merged_model = TinyViT(config).to(config.device)

    # Copy state dict from LoRA model (which has merged weights)
    # We need to be careful to only copy the base weights
    model_state = model.state_dict()
    merged_state = {}

    for key, value in model_state.items():
        # Skip LoRA-specific parameters
        if 'lora_' not in key:
            merged_state[key] = value

    merged_model.load_state_dict(merged_state)

    return merged_model


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
        # Step decay
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
        config = LoRAConfig()

    # Set random seed
    torch.manual_seed(config.seed)

    # Initialize wandb
    if config.use_wandb:
        run_name = experiment_name if experiment_name else f'lora_r{config.lora_rank}'
        wandb.init(
            project=config.project_name,
            name=run_name,
            config=vars(config)
        )

    # Create save directory
    os.makedirs(config.save_dir, exist_ok=True)

    print("=" * 70)
    print("TINYVIT TRAINING WITH LoRA")
    print("=" * 70)
    print(f"\nModel Configuration:")
    print(f"  Embed dim: {config.embed_dim}")
    print(f"  Layers: {config.num_layers}")
    print(f"  Heads: {config.num_heads}")
    print(f"\nLoRA Configuration:")
    print(f"  Rank: {config.lora_rank}")
    print(f"  Alpha: {config.lora_alpha}")
    print(f"  Dropout: {config.lora_dropout}")
    print(f"  Target modules: {config.target_modules}")
    print(f"\nTraining Configuration:")
    print(f"  Epochs: {config.num_epochs}")
    print(f"  Batch size: {config.batch_size}")
    print(f"  Learning rate: {config.learning_rate}")
    print(f"  Optimizer: {config.optimizer}")
    print(f"  LR schedule: {config.lr_schedule}")
    print(f"  Device: {config.device}")
    print("=" * 70)

    # Create base model and move to device FIRST
    print(f"\nCreating model on device: {config.device}")
    model = TinyViT(config).to(config.device)

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")

    # Add LoRA adapters (model is already on correct device)
    model = add_lora_to_model(model, config)

    # Freeze non-LoRA parameters
    trainable_params, total_params = freeze_non_lora_params(model)

    # Verify all parameters are on the correct device
    print("\nVerifying device placement...")
    devices_found = set()
    for name, param in model.named_parameters():
        devices_found.add(str(param.device))

    if len(devices_found) > 1:
        print(f"  ⚠️  WARNING: Parameters found on multiple devices: {devices_found}")
        print("  Moving all parameters to target device...")
        model = model.to(config.device)
    else:
        print(f"  ✓ All parameters on {list(devices_found)[0]}")

    # Quick forward pass test
    print("\nTesting forward pass...")
    try:
        test_input = torch.randn(2, 3, 32, 32).to(config.device)
        with torch.no_grad():
            _ = model(test_input)
        print("  ✓ Forward pass successful")
    except Exception as e:
        print(f"  ✗ Forward pass failed: {e}")
        raise

    # Data
    train_loader, test_loader = get_dataloaders(config)
    print(f"\nDataset:")
    print(f"  Train samples: {len(train_loader.dataset):,}")
    print(f"  Test samples: {len(test_loader.dataset):,}")
    print(f"  Batches per epoch: {len(train_loader):,}")

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Optimizer (only for LoRA parameters)
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.learning_rate,
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
                save_path = os.path.join(config.save_dir, 'best_lora_model.pth')
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_acc': best_acc,
                    'config': config
                }, save_path)
                print(f"  ✓ Saved best LoRA model (acc={best_acc:.2f}%)")

    print("\n" + "=" * 70)
    print("Training Complete!")
    print("=" * 70)
    print(f"\nBest Test Accuracy: {best_acc:.2f}%")

    # ═══════════════════════════════════════════════════════════════
    # Merge LoRA weights and save merged model
    # ═══════════════════════════════════════════════════════════════

    print("\n" + "=" * 70)
    print("Merging LoRA Weights into Base Model")
    print("=" * 70)

    # Merge LoRA weights
    merge_lora_weights(model)

    # Create a clean merged model
    merged_model = get_merged_model(model, config)

    # Verify the merged model performs the same
    print("\nVerifying merged model performance...")
    merged_loss, merged_acc = evaluate(
        merged_model, test_loader, criterion, config.device
    )
    print(f"  Merged model accuracy: {merged_acc:.2f}%")

    # Save merged model
    merged_save_path = os.path.join(config.save_dir, 'merged_model.pth')
    torch.save({
        'epoch': config.num_epochs,
        'model_state_dict': merged_model.state_dict(),
        'best_acc': best_acc,
        'merged_acc': merged_acc,
        'config': config
    }, merged_save_path)
    print(f"\n✓ Saved merged model to {merged_save_path}")

    print("\n" + "=" * 70)
    print("Summary")
    print("=" * 70)
    print(f"  Training parameters: {trainable_params:,} / {total_params:,} ({100 * trainable_params / total_params:.2f}%)")
    print(f"  Best accuracy (LoRA): {best_acc:.2f}%")
    print(f"  Merged model accuracy: {merged_acc:.2f}%")
    print(f"  Models saved to: {config.save_dir}")

    if config.use_wandb:
        wandb.log({
            'best_test_accuracy': best_acc,
            'merged_model_accuracy': merged_acc,
            'trainable_params': trainable_params,
            'total_params': total_params,
            'param_efficiency': 100 * trainable_params / total_params
        })
        wandb.finish()

    return merged_model, best_acc


# ═══════════════════════════════════════════════════════════════════
# Command Line Interface
# ═══════════════════════════════════════════════════════════════════

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train TinyViT with LoRA on CIFAR-10')

    # Model
    parser.add_argument('--embed-dim', type=int, default=128)
    parser.add_argument('--num-layers', type=int, default=4)
    parser.add_argument('--num-heads', type=int, default=4)

    # LoRA
    parser.add_argument('--lora-rank', type=int, default=16,
                        help='Rank of LoRA matrices')
    parser.add_argument('--lora-alpha', type=float, default=32.0,
                        help='LoRA scaling factor')
    parser.add_argument('--lora-dropout', type=float, default=0.1,
                        help='Dropout for LoRA layers')

    # Training
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--warmup-epochs', type=int, default=5)

    # Optimizer
    parser.add_argument('--lr-schedule', type=str, default='cosine',
                        choices=['cosine', 'step', 'constant'])

    # System
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no-wandb', action='store_true', help='Disable wandb')
    parser.add_argument('--name', type=str, default=None, help='Experiment name')

    args = parser.parse_args()

    # Create config from args
    config = LoRAConfig()
    config.embed_dim = args.embed_dim
    config.num_layers = args.num_layers
    config.num_heads = args.num_heads
    config.lora_rank = args.lora_rank
    config.lora_alpha = args.lora_alpha
    config.lora_dropout = args.lora_dropout
    config.num_epochs = args.epochs
    config.batch_size = args.batch_size
    config.learning_rate = args.lr
    config.weight_decay = args.weight_decay
    config.warmup_epochs = args.warmup_epochs
    config.lr_schedule = args.lr_schedule
    config.device = args.device
    config.seed = args.seed
    config.use_wandb = not args.no_wandb

    # Default experiment name
    if args.name is None:
        args.name = f'lora_r{args.lora_rank}_a{int(args.lora_alpha)}'

    # Train
    model, best_acc = main(config=config, experiment_name=args.name)

    print(f"\n🎉 Final Best Accuracy: {best_acc:.2f}%")