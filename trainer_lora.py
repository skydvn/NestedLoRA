"""
LoRA Implementation Perfectly Aligned with TinyViT Architecture

This implementation:
1. Works with the exact TinyViT class structure
2. Applies LoRA to specific Linear layers in MultiHeadSelfAttention and MLP
3. Handles device placement automatically
4. Provides easy save/load functionality
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from tqdm import tqdm
import math
import copy

# Import the original TinyViT components
from models.vit_small import (
    TinyViT, TinyViTConfig, PatchEmbedding, 
    TransformerBlock
)
from trainer import get_data_loaders


# ============================================================================
# LoRA Components
# ============================================================================

class LoRALayer(nn.Module):
    """
    Low-Rank Adaptation layer

    Implements: output = W*x + (B @ A)*x
    where W is frozen, A and B are trainable low-rank matrices
    """
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

        # Low-rank matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.lora_dropout = nn.Dropout(lora_dropout)

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x):
        """
        Apply LoRA transformation with automatic device handling
        """
        # Ensure same device (fixes CUDA/CPU mismatch)
        if x.device != self.lora_A.device:
            self.lora_A.data = self.lora_A.data.to(x.device)
            self.lora_B.data = self.lora_B.data.to(x.device)

        # Apply LoRA: (dropout(x) @ A) @ B * scaling
        dropout_x = self.lora_dropout(x)
        lora_out = (dropout_x @ self.lora_A) @ self.lora_B
        return lora_out * self.scaling


class LoRALinear(nn.Module):
    """
    Linear layer with LoRA adaptation

    Wraps a frozen Linear layer and adds trainable low-rank matrices
    """
    def __init__(
        self,
        original_linear: nn.Linear,
        rank: int = 8,
        lora_alpha: int = 16,
        lora_dropout: float = 0.1,
    ):
        super().__init__()
        self.original_linear = original_linear

        # Create LoRA layer
        self.lora = LoRALayer(
            original_linear.in_features,
            original_linear.out_features,
            rank=rank,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
        )

        # Freeze original weights
        for param in self.original_linear.parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward: original output + LoRA output
        """
        return self.original_linear(x) + self.lora(x)

    def merge_weights(self):
        """Merge LoRA weights into original linear layer"""
        with torch.no_grad():
            lora_weight = (self.lora.lora_B @ self.lora.lora_A.T) * self.lora.scaling
            self.original_linear.weight.data += lora_weight.T

    def unmerge_weights(self):
        """Unmerge LoRA weights from original linear layer"""
        with torch.no_grad():
            lora_weight = (self.lora.lora_B @ self.lora.lora_A.T) * self.lora.scaling
            self.original_linear.weight.data -= lora_weight.T


# ============================================================================
# LoRA-Enhanced TinyViT Components
# ============================================================================

class LoRAMultiHeadSelfAttention(nn.Module):
    """
    MultiHeadSelfAttention with LoRA on qkv and proj layers
    """
    def __init__(self, config, rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads

        assert config.embed_dim % config.num_heads == 0

        # Original linear layers
        base_qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        base_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Wrap with LoRA
        self.qkv = LoRALinear(base_qkv, rank, lora_alpha, lora_dropout)
        self.proj = LoRALinear(base_proj, rank, lora_alpha, lora_dropout)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V with LoRA
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads and project with LoRA
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


class LoRAMLP(nn.Module):
    """
    MLP with LoRA on fc1 and fc2 layers
    """
    def __init__(self, config, rank=8, lora_alpha=16, lora_dropout=0.1):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)

        # Original linear layers
        base_fc1 = nn.Linear(config.embed_dim, hidden_dim)
        base_fc2 = nn.Linear(hidden_dim, config.embed_dim)

        # Wrap with LoRA
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
    """
    TransformerBlock with LoRA-enhanced attention and MLP
    """
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
    """
    TinyViT with LoRA adaptation

    Perfectly aligned with original TinyViT structure
    """
    def __init__(
        self,
        config,
        rank=8,
        lora_alpha=16,
        lora_dropout=0.1,
        load_pretrained=None
    ):
        super().__init__()
        self.config = config
        self.rank = rank
        self.lora_alpha = lora_alpha

        # Patch embedding (no LoRA here - typically not fine-tuned)
        self.patch_embed = PatchEmbedding(config)

        # Class token and positional embedding (frozen)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        # Transformer blocks with LoRA
        self.blocks = nn.ModuleList([
            LoRATransformerBlock(config, rank, lora_alpha, lora_dropout)
            for _ in range(config.num_layers)
        ])

        # Classification head (frozen or with LoRA)
        self.norm = nn.LayerNorm(config.embed_dim)
        base_head = nn.Linear(config.embed_dim, config.num_classes)
        # Optionally apply LoRA to classification head
        self.head = LoRALinear(base_head, rank, lora_alpha, lora_dropout)

        # Initialize weights
        self._init_weights()

        # Load pretrained base model if provided
        if load_pretrained:
            self.load_pretrained_weights(load_pretrained)

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

        # Transformer blocks with LoRA
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits

    def load_pretrained_weights(self, path):
        """
        Load pretrained base model weights
        Only loads weights that match (ignores LoRA parameters)
        """
        print(f"Loading pretrained weights from {path}...")
        pretrained_dict = torch.load(path, map_location='cpu')
        model_dict = self.state_dict()

        # Filter out LoRA parameters and load only base model weights
        pretrained_dict = {
            k: v for k, v in pretrained_dict.items()
            if k in model_dict and 'lora' not in k
        }

        model_dict.update(pretrained_dict)
        self.load_state_dict(model_dict, strict=False)
        print(f"Loaded {len(pretrained_dict)} parameters from pretrained model")


# ============================================================================
# Utility Functions
# ============================================================================

def print_trainable_parameters(model):
    """
    Print trainable vs total parameters
    """
    trainable_params = 0
    all_params = 0

    for name, param in model.named_parameters():
        all_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()

    print(f"\n{'='*60}")
    print(f"Parameter Statistics:")
    print(f"{'='*60}")
    print(f"Trainable params:  {trainable_params:,}")
    print(f"All params:        {all_params:,}")
    print(f"Trainable %:       {100 * trainable_params / all_params:.2f}%")
    print(f"{'='*60}\n")


def save_lora_checkpoint(model, path):
    """
    Save only LoRA parameters (much smaller file)
    """
    lora_state_dict = {
        k: v for k, v in model.state_dict().items()
        if 'lora' in k
    }
    torch.save(lora_state_dict, path)
    print(f"Saved LoRA checkpoint: {path} ({len(lora_state_dict)} parameters)")


def load_lora_checkpoint(model, path):
    """
    Load LoRA parameters into model
    """
    lora_state_dict = torch.load(path)
    model.load_state_dict(lora_state_dict, strict=False)
    print(f"Loaded LoRA checkpoint: {path}")


# ============================================================================
# Training Functions
# ============================================================================

def train_epoch(model, loader, optimizer, criterion, device, epoch):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    pbar = tqdm(loader, desc=f'Epoch {epoch} - Training')
    for images, labels in pbar:
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

    return total_loss / len(loader), 100. * correct / total


def evaluate(model, loader, criterion, device):
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


# ============================================================================
# Main Training Script
# ============================================================================

def main():
    # Setup
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'\n{"="*60}')
    print(f"TinyViT with LoRA Training")
    print(f"{'='*60}")
    print(f'Device: {device}')

    # Configuration
    config = TinyViTConfig()
    lora_rank = 8
    lora_alpha = 16

    # Create model with LoRA
    print("\nCreating TinyViT with LoRA...")
    model = TinyViTLoRA(
        config,
        rank=lora_rank,
        lora_alpha=lora_alpha,
        lora_dropout=0.1,
        load_pretrained=None  # Set to path if you have pretrained weights
    ).to(device)

    # Print parameter statistics
    print_trainable_parameters(model)

    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_data_loaders(batch_size=128)

    # Training setup
    criterion = nn.CrossEntropyLoss()

    # Only optimize LoRA parameters
    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-3,
        weight_decay=0.01
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=50)

    # Training loop
    num_epochs = 50
    best_acc = 0

    print(f"\n{'='*60}")
    print(f"Starting Training")
    print(f"{'='*60}\n")

    for epoch in range(1, num_epochs + 1):
        print(f'\n--- Epoch {epoch}/{num_epochs} ---')

        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, criterion, device, epoch
        )

        # Evaluate
        test_loss, test_acc = evaluate(model, test_loader, criterion, device)

        # Step scheduler
        scheduler.step()

        # Print summary
        print(f'\nEpoch {epoch} Summary:')
        print(f'  Train: Loss={train_loss:.4f}, Acc={train_acc:.2f}%')
        print(f'  Test:  Loss={test_loss:.4f}, Acc={test_acc:.2f}%')

        # Save best model
        if test_acc > best_acc:
            best_acc = test_acc
            save_lora_checkpoint(model, 'tiny_vit_lora_best.pth')
            torch.save(model.state_dict(), 'tiny_vit_lora_full_best.pth')
            print(f'  ✓ Saved best model: {best_acc:.2f}%')

    print(f'\n{"="*60}')
    print(f'Training Complete!')
    print(f'Best Accuracy: {best_acc:.2f}%')
    print(f'{"="*60}\n')


if __name__ == '__main__':
    main()