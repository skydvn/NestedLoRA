# nested_lora_multi_rank.py - Multi-Optimizer LoRA with Different Ranks
# W_final = W₀ + ΔW₁(rank=4) + ΔW₂(rank=8) + ΔW₃(rank=16)

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Import base model components
from models.vit_small import (
    TinyViTConfig,
    PatchEmbedding,
    MultiHeadSelfAttention,
    MLP,
    TransformerBlock
)


# ============================================================================
# Multi-Rank LoRA Linear Layer
# ============================================================================

class MultiRankLoRALinear(nn.Module):
    """
    Linear layer with 3 independent LoRA adaptations of DIFFERENT ranks

    Output: y = W₀x + (B₁A₁)x + (B₂A₂)x + (B₃A₃)x

    Where:
        - ΔW₁ has rank r₁ (e.g., 4)  - Low rank, coarse features
        - ΔW₂ has rank r₂ (e.g., 8)  - Medium rank, balanced features
        - ΔW₃ has rank r₃ (e.g., 16) - High rank, fine-grained features
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            ranks: list = [4, 8, 16],  # Different rank for each LoRA!
            lora_alphas: list = [4, 8, 16],  # Corresponding alphas
            lora_dropout: float = 0.1
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.num_loras = len(ranks)

        # Base linear layer (frozen)
        self.base_linear = nn.Linear(in_features, out_features)
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # Create LoRA adaptations with DIFFERENT ranks
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.lora_dropout = nn.ModuleList()
        self.scalings = []

        for i, (rank, alpha) in enumerate(zip(ranks, lora_alphas)):
            # A matrix: (in_features, rank_i)
            self.lora_A.append(
                nn.Parameter(torch.empty(in_features, rank))
            )

            # B matrix: (rank_i, out_features)
            self.lora_B.append(
                nn.Parameter(torch.zeros(rank, out_features))
            )

            # Dropout
            self.lora_dropout.append(nn.Dropout(lora_dropout))

            # Scaling for this LoRA
            self.scalings.append(alpha / rank)

        # Initialize LoRA weights
        for i in range(self.num_loras):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[i])

    def forward(self, x):
        # Base transformation (frozen): W₀x
        base_output = self.base_linear(x)

        # Add all LoRA adaptations with different ranks
        lora_output = 0
        for i in range(self.num_loras):
            # Apply dropout
            dropped_x = self.lora_dropout[i](x)
            # ΔWᵢ·x = Bᵢ(Aᵢ·x) where Aᵢ has rank rᵢ
            lora_i = (dropped_x @ self.lora_A[i]) @ self.lora_B[i]
            lora_output = lora_output + lora_i * self.scalings[i]

        # Final output: W₀x + ΔW₁x + ΔW₂x + ΔW₃x
        return base_output + lora_output

    def get_lora_parameters(self, lora_idx):
        """Get parameters for a specific LoRA (for optimizer assignment)"""
        if lora_idx >= self.num_loras:
            raise ValueError(f"lora_idx {lora_idx} out of range [0, {self.num_loras - 1}]")
        return [self.lora_A[lora_idx], self.lora_B[lora_idx]]

    def get_lora_info(self, lora_idx):
        """Get information about a specific LoRA"""
        return {
            'rank': self.ranks[lora_idx],
            'scaling': self.scalings[lora_idx],
            'A_shape': self.lora_A[lora_idx].shape,
            'B_shape': self.lora_B[lora_idx].shape,
            'num_params': self.lora_A[lora_idx].numel() + self.lora_B[lora_idx].numel()
        }


# ============================================================================
# Multi-Rank LoRA Attention
# ============================================================================

class MultiRankLoRAAttention(nn.Module):
    """
    Multi-head self-attention with 3 independent LoRA adaptations of different ranks
    """

    def __init__(self, config, ranks=[4, 8, 16], lora_alphas=[4, 8, 16], lora_dropout=0.1):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads

        # QKV projection with multi-rank LoRAs
        self.qkv = MultiRankLoRALinear(
            config.embed_dim,
            config.embed_dim * 3,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

        # Output projection with multi-rank LoRAs
        self.proj = MultiRankLoRALinear(
            config.embed_dim,
            config.embed_dim,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, C = x.shape

        # QKV with additive multi-rank LoRA
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention mechanism
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Output with additive multi-rank LoRA
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x


# ============================================================================
# Multi-Rank LoRA MLP
# ============================================================================

class MultiRankLoRAMLP(nn.Module):
    """
    MLP with 3 independent LoRA adaptations of different ranks
    """

    def __init__(self, config, ranks=[4, 8, 16], lora_alphas=[4, 8, 16], lora_dropout=0.1):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)

        # Both layers have multi-rank LoRAs
        self.fc1 = MultiRankLoRALinear(
            config.embed_dim,
            hidden_dim,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

        self.fc2 = MultiRankLoRALinear(
            hidden_dim,
            config.embed_dim,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)  # W₀ + ΔW₁(r=4) + ΔW₂(r=8) + ΔW₃(r=16)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)  # W₀ + ΔW₁(r=4) + ΔW₂(r=8) + ΔW₃(r=16)
        x = self.dropout(x)
        return x


# ============================================================================
# Multi-Rank LoRA Transformer Block
# ============================================================================

class MultiRankLoRATransformerBlock(nn.Module):
    """Transformer block with Multi-Rank LoRA attention and MLP"""

    def __init__(self, config, ranks=[4, 8, 16], lora_alphas=[4, 8, 16], lora_dropout=0.1):
        super().__init__()

        # Layer norms (frozen)
        self.norm1 = nn.LayerNorm(config.embed_dim)
        self.norm2 = nn.LayerNorm(config.embed_dim)

        # Freeze layer norms
        for param in self.norm1.parameters():
            param.requires_grad = False
        for param in self.norm2.parameters():
            param.requires_grad = False

        # Multi-Rank LoRA components
        self.attn = MultiRankLoRAAttention(config, ranks, lora_alphas, lora_dropout)
        self.mlp = MultiRankLoRAMLP(config, ranks, lora_alphas, lora_dropout)

    def forward(self, x):
        # Pre-norm architecture with residual connections
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


# ============================================================================
# TinyViT with Multi-Rank LoRA
# ============================================================================

class TinyViTMultiRankLoRA(nn.Module):
    """
    TinyViT with 3 independent LoRA adaptations of DIFFERENT ranks

    Each linear layer computes:
        W_final = W₀ + ΔW₁(rank=r₁) + ΔW₂(rank=r₂) + ΔW₃(rank=r₃)

    Typical configuration:
        - LoRA 1: rank=4  (coarse, low-level features)
        - LoRA 2: rank=8  (medium-level features)
        - LoRA 3: rank=16 (fine-grained, high-level features)
    """

    def __init__(
            self,
            config,
            ranks=[4, 8, 16],
            lora_alphas=[4, 8, 16],
            lora_dropout=0.1
    ):
        super().__init__()
        self.config = config
        self.ranks = ranks
        self.lora_alphas = lora_alphas
        self.num_loras = len(ranks)

        print(f"\n{'=' * 70}")
        print(f"Creating TinyViT with Multi-Rank LoRA")
        print(f"{'=' * 70}")
        print(f"Ranks: {ranks}")
        print(f"Alphas: {lora_alphas}")
        print(f"Configuration:")
        for i, (r, a) in enumerate(zip(ranks, lora_alphas)):
            print(f"  LoRA {i + 1}: rank={r:2d}, alpha={a:2d}, scaling={a / r:.2f}")

        # Base components (frozen)
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        # Freeze base embeddings
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

        # Transformer blocks with Multi-Rank LoRA
        self.blocks = nn.ModuleList([
            MultiRankLoRATransformerBlock(config, ranks, lora_alphas, lora_dropout)
            for _ in range(config.num_layers)
        ])

        # Classification head with Multi-Rank LoRA
        self.norm = nn.LayerNorm(config.embed_dim)
        for param in self.norm.parameters():
            param.requires_grad = False

        self.head = MultiRankLoRALinear(
            config.embed_dim,
            config.num_classes,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

    def get_lora_parameters(self, lora_idx):
        """
        Get all parameters for a specific LoRA index
        """
        if lora_idx >= self.num_loras:
            raise ValueError(f"lora_idx {lora_idx} out of range [0, {self.num_loras - 1}]")

        params = []

        # Collect from all transformer blocks
        for block in self.blocks:
            # Attention LoRA parameters
            params.extend(block.attn.qkv.get_lora_parameters(lora_idx))
            params.extend(block.attn.proj.get_lora_parameters(lora_idx))

            # MLP LoRA parameters
            params.extend(block.mlp.fc1.get_lora_parameters(lora_idx))
            params.extend(block.mlp.fc2.get_lora_parameters(lora_idx))

        # Head LoRA parameters
        params.extend(self.head.get_lora_parameters(lora_idx))

        return params

    def forward(self, x):
        B = x.shape[0]

        # Patch embedding (frozen)
        x = self.patch_embed(x)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks (each uses W₀ + ΔW₁ + ΔW₂ + ΔW₃ with different ranks)
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]
        logits = self.head(cls_token_final)

        return logits


# ============================================================================
# Helper Functions
# ============================================================================

def create_multi_rank_optimizers(
        model,
        lrs=[1e-3, 1e-3, 3e-3],
        weight_decay=0.01
):
    """
    Create 3 independent optimizers for the 3 LoRA adaptations

    Args:
        model: TinyViTMultiRankLoRA model
        lrs: List of learning rates for each LoRA
        weight_decay: Weight decay for all optimizers

    Returns:
        Tuple of 3 optimizers
    """

    optimizers = []

    for i, lr in enumerate(lrs):
        optimizer = torch.optim.AdamW(
            model.get_lora_parameters(lora_idx=i),
            lr=lr,
            betas=(0.9, 0.999),
            weight_decay=weight_decay
        )
        optimizers.append(optimizer)
        print(f"Optimizer {i + 1}: lr={lr:.6f}, rank={model.ranks[i]}")

    return tuple(optimizers)


def print_multi_rank_parameter_stats(model):
    """Print detailed statistics for each LoRA with different ranks"""

    print(f"\n{'=' * 70}")
    print(f"Multi-Rank LoRA Parameter Statistics")
    print(f"{'=' * 70}")

    # Count base (frozen) parameters
    base_params = sum(
        p.numel() for p in model.parameters()
        if not p.requires_grad
    )

    # Count parameters for each LoRA
    lora_params = []
    lora_details = []

    for i in range(model.num_loras):
        params = sum(p.numel() for p in model.get_lora_parameters(i))
        lora_params.append(params)

        # Get detailed info from first linear layer
        first_layer = model.blocks[0].attn.qkv
        info = first_layer.get_lora_info(i)
        lora_details.append(info)

    total_params = base_params + sum(lora_params)
    trainable_params = sum(lora_params)

    print(f"\nBase Model (W₀):     {base_params:,} params (frozen)")
    print(f"\nLoRA Adaptations:")

    for i, (params, details) in enumerate(zip(lora_params, lora_details)):
        print(f"  LoRA {i + 1} (rank={details['rank']:2d}):")
        print(f"    Parameters:       {params:,}")
        print(f"    Scaling factor:   {details['scaling']:.2f}")
        print(f"    A matrix shape:   {details['A_shape']}")
        print(f"    B matrix shape:   {details['B_shape']}")

    print(f"\n{'-' * 70}")
    print(f"Total:               {total_params:,} params")
    print(f"Trainable:           {trainable_params:,} params ({100 * trainable_params / total_params:.2f}%)")

    # Breakdown by rank
    print(f"\n{'-' * 70}")
    print(f"Parameter Distribution by Rank:")
    for i, (params, rank) in enumerate(zip(lora_params, model.ranks)):
        percentage = 100 * params / trainable_params
        print(f"  Rank {rank:2d} (LoRA {i + 1}): {params:,} params ({percentage:.1f}% of trainable)")

    print(f"{'=' * 70}\n")

    return trainable_params, total_params


def compute_grad_norm(parameters):
    """Compute L2 norm of gradients"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    return total_norm ** 0.5


def visualize_rank_differences(model):
    """Visualize how different ranks affect the LoRA matrices"""
    import matplotlib.pyplot as plt
    import numpy as np

    # Get first linear layer for visualization
    layer = model.blocks[0].attn.qkv

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle('Multi-Rank LoRA: Different Ranks for Different LoRAs', fontsize=16)

    for i in range(model.num_loras):
        A = layer.lora_A[i].detach().cpu().numpy()
        B = layer.lora_B[i].detach().cpu().numpy()

        rank = model.ranks[i]

        # Plot A matrix
        im = axes[0, i].imshow(A, cmap='RdBu', aspect='auto')
        axes[0, i].set_title(f'LoRA {i + 1}: A matrix (rank={rank})')
        axes[0, i].set_xlabel(f'Rank dimension ({rank})')
        axes[0, i].set_ylabel('Input features')
        plt.colorbar(im, ax=axes[0, i])

        # Plot B matrix
        im = axes[1, i].imshow(B, cmap='RdBu', aspect='auto')
        axes[1, i].set_title(f'LoRA {i + 1}: B matrix (rank={rank})')
        axes[1, i].set_xlabel('Output features')
        axes[1, i].set_ylabel(f'Rank dimension ({rank})')
        plt.colorbar(im, ax=axes[1, i])

    plt.tight_layout()
    plt.savefig('multi_rank_lora_matrices.png', dpi=150)
    print("✓ Saved visualization to 'multi_rank_lora_matrices.png'")


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == '__main__':
    print("\n" + "=" * 70)
    print("Multi-Rank LoRA Example")
    print("=" * 70)

    # Create configuration
    config = TinyViTConfig()

    # Create model with different ranks
    # LoRA 1: rank=4  (coarse features, fewer params, faster)
    # LoRA 2: rank=8  (medium features, balanced)
    # LoRA 3: rank=16 (fine features, more params, slower)
    model = TinyViTMultiRankLoRA(
        config,
        ranks=[4, 8, 16],
        lora_alphas=[4, 8, 16],
        lora_dropout=0.1
    )

    # Print parameter statistics
    trainable, total = print_multi_rank_parameter_stats(model)

    # Create 3 optimizers with different learning rates
    print(f"\n{'=' * 70}")
    print("Creating Optimizers")
    print(f"{'=' * 70}")

    # Different learning rates for different ranks
    # Lower rank = coarse features = higher LR (faster learning)
    # Higher rank = fine features = lower LR (careful learning)
    optimizer1, optimizer2, optimizer3 = create_multi_rank_optimizers(
        model,
        lrs=[3e-3, 1e-3, 5e-4],  # Decreasing LR for increasing rank
        weight_decay=0.01
    )

    print(f"{'=' * 70}")

    # Test forward pass
    print(f"\n{'=' * 70}")
    print("Testing Forward Pass")
    print(f"{'=' * 70}")

    dummy_input = torch.randn(2, 3, 32, 32)
    output = model(dummy_input)

    print(f"✓ Forward pass successful!")
    print(f"  Input shape:  {dummy_input.shape}")
    print(f"  Output shape: {output.shape}")

    # Visualize
    print(f"\n{'=' * 70}")
    print("Creating Visualization")
    print(f"{'=' * 70}")
    visualize_rank_differences(model)

    print(f"\n{'=' * 70}")
    print("✓ Multi-Rank LoRA model created successfully!")
    print(f"{'=' * 70}\n")