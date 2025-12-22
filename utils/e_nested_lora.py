import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Optional, Tuple


# ============================================================================
# LoRA Modules for Different Layer Types
# ============================================================================

class LoRALinear(nn.Module):
    """LoRA adaptation for Linear layers"""

    def __init__(
            self,
            in_features: int,
            out_features: int,
            rank: int = 4,
            alpha: int = 1,
            dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation"""
        # x @ A @ B with scaling
        return self.dropout(x @ self.lora_A @ self.lora_B) * self.scaling


class LoRAConv2d(nn.Module):
    """LoRA adaptation for Conv2d layers (for PatchEmbedding)"""

    def __init__(
            self,
            in_channels: int,
            out_channels: int,
            kernel_size: int,
            rank: int = 4,
            alpha: int = 1,
            dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scalings = alpha / rank
        self.kernel_size = kernel_size

        # LoRA decomposition for convolution
        # Conv2d can be decomposed as: in_channels*k*k -> rank -> out_channels
        self.lora_A = nn.Parameter(
            torch.zeros(rank, in_channels, kernel_size, kernel_size)
        )
        self.lora_B = nn.Parameter(
            torch.zeros(out_channels, rank, 1, 1)
        )

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply LoRA adaptation to convolution"""
        # Apply low-rank convolution: conv(x, A) then conv(., B)
        x_adapted = F.conv2d(x, self.lora_A, stride=self.kernel_size)
        x_adapted = F.conv2d(x_adapted, self.lora_B)
        return self.dropout(x_adapted) * self.scalings


class LoRALayerNorm(nn.Module):
    """LoRA adaptation for LayerNorm

    Applies LoRA to the affine transformation (scale and shift)
    """

    def __init__(
            self,
            normalized_shape: int,
            rank: int = 4,
            alpha: int = 1,
            dropout: float = 0.0
    ):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        # LoRA for scale (weight) parameter
        self.lora_scale_A = nn.Parameter(torch.zeros(normalized_shape, rank))
        self.lora_scale_B = nn.Parameter(torch.zeros(rank, normalized_shape))

        # LoRA for shift (bias) parameter
        self.lora_shift_A = nn.Parameter(torch.zeros(normalized_shape, rank))
        self.lora_shift_B = nn.Parameter(torch.zeros(rank, normalized_shape))

        # Dropout
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        # Initialize
        nn.init.kaiming_uniform_(self.lora_scale_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_scale_B)
        nn.init.kaiming_uniform_(self.lora_shift_A, a=5 ** 0.5)
        nn.init.zeros_(self.lora_shift_B)

    def get_adapted_weight(self) -> torch.Tensor:
        """Get the LoRA-adapted weight (scale) vector"""
        # lora_A @ lora_B creates a matrix, take diagonal
        lora_weight = (self.lora_scale_A @ self.lora_scale_B).diagonal()
        return self.dropout(lora_weight) * self.scaling

    def get_adapted_bias(self) -> torch.Tensor:
        """Get the LoRA-adapted bias (shift) vector"""
        lora_bias = (self.lora_shift_A @ self.lora_shift_B).diagonal()
        return self.dropout(lora_bias) * self.scaling


# ============================================================================
# Multi-Rank LoRA Wrapper for each layer type
# ============================================================================

class MultiRankLoRALinear(nn.Module):
    """Linear layer with multiple LoRA adaptations"""

    def __init__(
            self,
            base_layer: nn.Linear,
            ranks: List[int],
            alphas: List[int],
            dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_loras = len(ranks)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Create multiple LoRA adaptations
        self.loras = nn.ModuleList([
            LoRALinear(
                base_layer.in_features,
                base_layer.out_features,
                rank=ranks[i],
                alpha=alphas[i],
                dropout=dropout
            )
            for i in range(self.num_loras)
        ])

        # Track which LoRAs are active
        self.active_loras = [True] * self.num_loras

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with active LoRAs"""
        # Base layer output
        output = self.base_layer(x)

        # Add active LoRA adaptations
        for i, lora in enumerate(self.loras):
            if self.active_loras[i]:
                output = output + lora(x)

        return output

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        assert len(active) == self.num_loras
        self.active_loras = active


class MultiRankLoRAConv2d(nn.Module):
    """Conv2d layer with multiple LoRA adaptations"""

    def __init__(
            self,
            base_layer: nn.Conv2d,
            ranks: List[int],
            alphas: List[int],
            dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_loras = len(ranks)

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Create multiple LoRA adaptations
        self.loras = nn.ModuleList([
            LoRAConv2d(
                base_layer.in_channels,
                base_layer.out_channels,
                base_layer.kernel_size[0],
                rank=ranks[i],
                alpha=alphas[i],
                dropout=dropout
            )
            for i in range(self.num_loras)
        ])

        # Track which LoRAs are active
        self.active_loras = [True] * self.num_loras

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with active LoRAs"""
        # Base layer output
        output = self.base_layer(x)

        # Add active LoRA adaptations
        for i, lora in enumerate(self.loras):
            if self.active_loras[i]:
                output = output + lora(x)

        return output

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        assert len(active) == self.num_loras
        self.active_loras = active


class MultiRankLoRALayerNorm(nn.Module):
    """LayerNorm with multiple LoRA adaptations"""

    def __init__(
            self,
            base_layer: nn.LayerNorm,
            ranks: List[int],
            alphas: List[int],
            dropout: float = 0.0
    ):
        super().__init__()
        self.base_layer = base_layer
        self.num_loras = len(ranks)
        self.normalized_shape = base_layer.normalized_shape[0]

        # Freeze base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False

        # Create multiple LoRA adaptations
        self.loras = nn.ModuleList([
            LoRALayerNorm(
                self.normalized_shape,
                rank=ranks[i],
                alpha=alphas[i],
                dropout=dropout
            )
            for i in range(self.num_loras)
        ])

        # Track which LoRAs are active
        self.active_loras = [True] * self.num_loras

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with active LoRAs"""
        # Base LayerNorm
        output = self.base_layer(x)

        # Add LoRA adaptations to the affine transformation
        for i, lora in enumerate(self.loras):
            if self.active_loras[i]:
                # Apply LoRA to weight and bias
                adapted_weight = lora.get_adapted_weight()
                adapted_bias = lora.get_adapted_bias()

                # output = output * (1 + delta_weight) + delta_bias
                output = output * (1 + adapted_weight.unsqueeze(0)) + adapted_bias.unsqueeze(0)

        return output

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        assert len(active) == self.num_loras
        self.active_loras = active


# ============================================================================
# Enhanced Model Components with LoRA
# ============================================================================

class PatchEmbeddingWithLoRA(nn.Module):
    """Patch embedding with LoRA adaptation"""

    def __init__(self, config, ranks: List[int], alphas: List[int], dropout: float = 0.0):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches

        # Base convolution
        base_projection = nn.Conv2d(
            3, config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

        # Wrap with multi-rank LoRA
        self.projection = MultiRankLoRAConv2d(
            base_projection,
            ranks=ranks,
            alphas=alphas,
            dropout=dropout
        )

    def forward(self, x):
        # x: (B, 3, 32, 32) -> (B, embed_dim, 8, 8)
        x = self.projection(x)
        # Flatten patches: (B, embed_dim, 8, 8) -> (B, embed_dim, 64) -> (B, 64, embed_dim)
        x = x.flatten(2).transpose(1, 2)
        return x

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        self.projection.set_active_loras(active)


class MultiHeadSelfAttentionWithLoRA(nn.Module):
    """Multi-head attention with LoRA"""

    def __init__(self, config, ranks: List[int], alphas: List[int], dropout: float = 0.0):
        super().__init__()
        self.num_heads = config.num_heads
        self.embed_dim = config.embed_dim
        self.head_dim = config.embed_dim // config.num_heads

        assert config.embed_dim % config.num_heads == 0

        # Base layers
        base_qkv = nn.Linear(config.embed_dim, config.embed_dim * 3)
        base_proj = nn.Linear(config.embed_dim, config.embed_dim)

        # Wrap with multi-rank LoRA
        self.qkv = MultiRankLoRALinear(base_qkv, ranks, alphas, dropout)
        self.proj = MultiRankLoRALinear(base_proj, ranks, alphas, dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        B, N, C = x.shape

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Combine heads
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.dropout(x)

        return x

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        self.qkv.set_active_loras(active)
        self.proj.set_active_loras(active)


class MLPWithLoRA(nn.Module):
    """MLP with LoRA"""

    def __init__(self, config, ranks: List[int], alphas: List[int], dropout: float = 0.0):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)

        # Base layers
        base_fc1 = nn.Linear(config.embed_dim, hidden_dim)
        base_fc2 = nn.Linear(hidden_dim, config.embed_dim)

        # Wrap with multi-rank LoRA
        self.fc1 = MultiRankLoRALinear(base_fc1, ranks, alphas, dropout)
        self.fc2 = MultiRankLoRALinear(base_fc2, ranks, alphas, dropout)

        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        self.fc1.set_active_loras(active)
        self.fc2.set_active_loras(active)


class TransformerBlockWithLoRA(nn.Module):
    """Transformer block with LoRA on all components including LayerNorm"""

    def __init__(self, config, ranks: List[int], alphas: List[int], dropout: float = 0.0):
        super().__init__()

        # Base LayerNorms
        base_norm1 = nn.LayerNorm(config.embed_dim)
        base_norm2 = nn.LayerNorm(config.embed_dim)

        # Wrap with multi-rank LoRA
        self.norm1 = MultiRankLoRALayerNorm(base_norm1, ranks, alphas, dropout)
        self.norm2 = MultiRankLoRALayerNorm(base_norm2, ranks, alphas, dropout)

        # Attention and MLP with LoRA
        self.attn = MultiHeadSelfAttentionWithLoRA(config, ranks, alphas, dropout)
        self.mlp = MLPWithLoRA(config, ranks, alphas, dropout)

    def forward(self, x):
        # Pre-norm architecture
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active"""
        self.norm1.set_active_loras(active)
        self.norm2.set_active_loras(active)
        self.attn.set_active_loras(active)
        self.mlp.set_active_loras(active)


# ============================================================================
# Complete TinyViT with Enhanced Multi-Rank LoRA
# ============================================================================

class TinyViTEnhancedMultiRankLoRA(nn.Module):
    """TinyViT with multi-rank LoRA on ALL layers including PatchEmbedding and LayerNorm"""

    def __init__(
            self,
            config,
            ranks: List[int] = [4, 8, 16],
            lora_alphas: List[int] = [4, 8, 16],
            lora_dropout: float = 0.1
    ):
        super().__init__()
        self.config = config
        self.num_loras = len(ranks)

        # Patch embedding with LoRA
        self.patch_embed = PatchEmbeddingWithLoRA(config, ranks, lora_alphas, lora_dropout)

        # Class token (trainable, not LoRA)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))

        # Positional embedding (trainable, not LoRA)
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        # Transformer blocks with LoRA
        self.blocks = nn.ModuleList([
            TransformerBlockWithLoRA(config, ranks, lora_alphas, lora_dropout)
            for _ in range(config.num_layers)
        ])

        # Final LayerNorm with LoRA
        base_final_norm = nn.LayerNorm(config.embed_dim)
        self.norm = MultiRankLoRALayerNorm(base_final_norm, ranks, lora_alphas, lora_dropout)

        # Classification head with LoRA
        base_head = nn.Linear(config.embed_dim, config.num_classes)
        self.head = MultiRankLoRALinear(base_head, ranks, lora_alphas, lora_dropout)

        # Initialize weights
        self._init_weights()

        # Track active LoRAs
        self.active_loras = [True] * self.num_loras

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

    def set_active_loras(self, active: List[bool]):
        """Set which LoRAs are active across all layers"""
        assert len(active) == self.num_loras, \
            f"Expected {self.num_loras} boolean values, got {len(active)}"

        self.active_loras = active

        # Update all components
        self.patch_embed.set_active_loras(active)
        for block in self.blocks:
            block.set_active_loras(active)
        self.norm.set_active_loras(active)
        self.head.set_active_loras(active)

    def get_lora_parameters(self, lora_idx: int):
        """Get parameters for a specific LoRA"""
        params = []

        # Patch embedding
        params.extend(self.patch_embed.projection.loras[lora_idx].parameters())

        # Transformer blocks
        for block in self.blocks:
            params.extend(block.norm1.loras[lora_idx].parameters())
            params.extend(block.attn.qkv.loras[lora_idx].parameters())
            params.extend(block.attn.proj.loras[lora_idx].parameters())
            params.extend(block.norm2.loras[lora_idx].parameters())
            params.extend(block.mlp.fc1.loras[lora_idx].parameters())
            params.extend(block.mlp.fc2.loras[lora_idx].parameters())

        # Final norm
        params.extend(self.norm.loras[lora_idx].parameters())

        # Classification head
        params.extend(self.head.loras[lora_idx].parameters())

        return params


# ============================================================================
# Utility Functions
# ============================================================================

def create_multi_rank_optimizers(
        model: TinyViTEnhancedMultiRankLoRA,
        lrs: List[float],
        weight_decay: float = 0.0
):
    """Create separate optimizers for each LoRA"""
    optimizers = []

    for i, lr in enumerate(lrs):
        params = model.get_lora_parameters(i)
        optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
        optimizers.append(optimizer)

    return optimizers


def print_enhanced_parameter_stats(model: TinyViTEnhancedMultiRankLoRA):
    """Print parameter statistics for the enhanced model"""

    total_params = sum(p.numel() for p in model.parameters())
    frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
    trainable_params = total_params - frozen_params

    print(f"\n{'=' * 70}")
    print(f"Enhanced Model Parameter Statistics")
    print(f"{'=' * 70}")
    print(f"Total parameters: {total_params:,}")
    print(f"Frozen parameters: {frozen_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Trainable %: {100 * trainable_params / total_params:.2f}%")

    # Per-LoRA statistics
    print(f"\n{'=' * 70}")
    print(f"Per-LoRA Parameter Counts")
    print(f"{'=' * 70}")

    for i in range(model.num_loras):
        lora_params = sum(p.numel() for p in model.get_lora_parameters(i))
        print(f"LoRA {i + 1} (rank={model.config.ranks[i]}): {lora_params:,} parameters")

    print(f"{'=' * 70}\n")

    return trainable_params, total_params


def compute_grad_norm(parameters):
    """Compute gradient norm for a set of parameters"""
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm