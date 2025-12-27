import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm


# Tiny ViT Configuration for CIFAR-10
class TinyViTConfig:
    img_size = 224
    patch_size = 16  # 32/4 = 8x8 patches
    num_patches = (img_size // patch_size) ** 2  # 64 patches
    embed_dim = 192  # Small embedding dimension
    num_heads = 3
    num_layers = 12
    mlp_ratio = 4
    num_classes = 100
    dropout = 0.1


def load_pretrained_vit_tiny(model, pretrained_model_name='vit_tiny_patch16_224'):
    """
    Load pretrained ViT-Tiny weights into the frozen base layers of MultiRankLoRA model
    """
    print(f"\nLoading pretrained weights from {pretrained_model_name}...")

    pretrained_vit = timm.create_model(pretrained_model_name, pretrained=True)
    pretrained_state_dict = pretrained_vit.state_dict()
    model_state_dict = model.state_dict()

    # =========================
    # Counters
    # =========================
    loaded_tensors = 0
    skipped_tensors = 0
    loaded_blocks = 0
    total_blocks = model.config.num_layers

    def safe_copy(dst_key, src_key):
        nonlocal loaded_tensors, skipped_tensors
        if src_key in pretrained_state_dict and dst_key in model_state_dict:
            try:
                model_state_dict[dst_key].copy_(pretrained_state_dict[src_key])
                loaded_tensors += 1
                return True
            except Exception as e:
                print(f"✗ Shape mismatch: {dst_key} <- {src_key} ({e})")
        else:
            print(f"✗ Missing key: {dst_key} or {src_key}")
        skipped_tensors += 1
        return False

    # =========================
    # Embeddings
    # =========================
    embedding_keys = {
        'patch_embed.proj.weight': 'patch_embed.proj.weight',
        'patch_embed.proj.bias': 'patch_embed.proj.bias',
        'cls_token': 'cls_token',
        'pos_embed': 'pos_embed',
    }

    print("\nLoading embeddings:")
    for src, dst in embedding_keys.items():
        if safe_copy(dst, src):
            print(f"✓ Loaded {src}")

    # =========================
    # Transformer blocks
    # =========================
    print("\nLoading transformer blocks:")
    for layer_idx in range(total_blocks):
        success = True

        success &= safe_copy(
            f'blocks.{layer_idx}.attn.qkv.base_linear.weight',
            f'blocks.{layer_idx}.attn.qkv.weight'
        )
        success &= safe_copy(
            f'blocks.{layer_idx}.attn.qkv.base_linear.bias',
            f'blocks.{layer_idx}.attn.qkv.bias'
        )

        success &= safe_copy(
            f'blocks.{layer_idx}.attn.proj.base_linear.weight',
            f'blocks.{layer_idx}.attn.proj.weight'
        )
        success &= safe_copy(
            f'blocks.{layer_idx}.attn.proj.base_linear.bias',
            f'blocks.{layer_idx}.attn.proj.bias'
        )

        success &= safe_copy(
            f'blocks.{layer_idx}.mlp.fc1.base_linear.weight',
            f'blocks.{layer_idx}.mlp.fc1.weight'
        )
        success &= safe_copy(
            f'blocks.{layer_idx}.mlp.fc1.base_linear.bias',
            f'blocks.{layer_idx}.mlp.fc1.bias'
        )

        success &= safe_copy(
            f'blocks.{layer_idx}.mlp.fc2.base_linear.weight',
            f'blocks.{layer_idx}.mlp.fc2.weight'
        )
        success &= safe_copy(
            f'blocks.{layer_idx}.mlp.fc2.base_linear.bias',
            f'blocks.{layer_idx}.mlp.fc2.bias'
        )

        success &= safe_copy(
            f'blocks.{layer_idx}.norm1.weight',
            f'blocks.{layer_idx}.norm1.weight'
        )
        success &= safe_copy(
            f'blocks.{layer_idx}.norm1.bias',
            f'blocks.{layer_idx}.norm1.bias'
        )

        success &= safe_copy(
            f'blocks.{layer_idx}.norm2.weight',
            f'blocks.{layer_idx}.norm2.weight'
        )
        success &= safe_copy(
            f'blocks.{layer_idx}.norm2.bias',
            f'blocks.{layer_idx}.norm2.bias'
        )

        if success:
            loaded_blocks += 1
            print(f"✓ Loaded block {layer_idx}")
        else:
            print(f"⚠ Partial load for block {layer_idx}")

    # =========================
    # Final norm
    # =========================
    print("\nLoading final norm:")
    safe_copy('norm.weight', 'norm.weight')
    safe_copy('norm.bias', 'norm.bias')

    # =========================
    # Summary
    # =========================
    print("\n" + "=" * 50)
    print("✅ PRETRAINED WEIGHT LOADING SUMMARY")
    print(f"• Transformer blocks loaded: {loaded_blocks}/{total_blocks}")
    print(f"• Tensors loaded successfully: {loaded_tensors}")
    print(f"• Tensors skipped/failed: {skipped_tensors}")
    print("• Classification head: NOT loaded (expected)")
    print("• LoRA parameters: untouched and trainable")
    print("=" * 50)

    return model


class PatchEmbedding(nn.Module):
    """Split image into patches and embed them"""

    def __init__(self, config):
        super().__init__()
        self.patch_size = config.patch_size
        self.num_patches = config.num_patches

        # Convolution to extract patches and project to embedding dimension
        self.projection = nn.Conv2d(
            3, config.embed_dim,
            kernel_size=config.patch_size,
            stride=config.patch_size
        )

    def forward(self, x):
        # x: (B, 3, 32, 32) -> (B, embed_dim, 8, 8)
        x = self.projection(x)
        # Flatten patches: (B, embed_dim, 8, 8) -> (B, embed_dim, 64) -> (B, 64, embed_dim)
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

        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, num_heads, N, head_dim)
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


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        hidden_dim = int(config.embed_dim * config.mlp_ratio)
        self.fc1 = nn.Linear(config.embed_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, config.embed_dim)
        self.dropout = nn.Dropout(config.dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = F.gelu(x)
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
        # Pre-norm architecture
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
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

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
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)

        # Add class token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches+1, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        cls_token_final = x[:, 0]  # Take class token
        logits = self.head(cls_token_final)

        return logits

