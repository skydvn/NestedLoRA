import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


# utils/memory_efficient_model.py

class MemoryEfficientTinyViTMultiRankLoRA(nn.Module):
    """
    Memory-efficient TinyViT with multi-rank LoRA.
    Only one LoRA is kept in GPU memory at a time.
    """

    def __init__(
            self,
            config,
            ranks=[4, 8, 16],
            lora_alphas=[4, 8, 16],
            lora_dropout=0.1,
            offload_device='cpu'
    ):
        super().__init__()
        self.config = config
        self.ranks = ranks
        self.num_loras = len(ranks)
        self.active_lora_idx = 0

        # Base components (frozen)
        self.patch_embed = PatchEmbedding(config)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, config.embed_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, config.num_patches + 1, config.embed_dim)
        )

        # Freeze base
        for param in self.patch_embed.parameters():
            param.requires_grad = False
        self.cls_token.requires_grad = False
        self.pos_embed.requires_grad = False

        # Build model with memory-efficient LoRA layers
        self.blocks = nn.ModuleList([
            self._make_memory_efficient_block(config, ranks, lora_alphas, lora_dropout)
            for _ in range(config.num_layers)
        ])

        self.norm = nn.LayerNorm(config.embed_dim)
        for param in self.norm.parameters():
            param.requires_grad = False

        self.head = MemoryEfficientMultiRankLoRALinear(
            config.embed_dim,
            config.num_classes,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

    def _make_memory_efficient_block(self, config, ranks, lora_alphas, lora_dropout):
        """Create a transformer block with memory-efficient LoRA"""
        # Implementation similar to your MultiRankLoRATransformerBlock
        # but using MemoryEfficientMultiRankLoRALinear
        pass

    def switch_active_lora(self, new_lora_idx: int):
        """Switch active LoRA across entire model"""
        print(f"\n{'=' * 70}")
        print(f"Switching Active LoRA: {self.active_lora_idx} → {new_lora_idx}")
        print(f"{'=' * 70}")

        # Switch in all blocks
        for i, block in enumerate(self.blocks):
            # Attention layers
            block.attn.qkv.switch_active_lora(new_lora_idx)
            block.attn.proj.switch_active_lora(new_lora_idx)

            # MLP layers
            block.mlp.fc1.switch_active_lora(new_lora_idx)
            block.mlp.fc2.switch_active_lora(new_lora_idx)

        # Head
        self.head.switch_active_lora(new_lora_idx)

        self.active_lora_idx = new_lora_idx

        # Print memory stats
        self.print_memory_stats()

    def print_memory_stats(self):
        """Print current memory usage"""
        total_active = 0
        total_offloaded = 0

        for block in self.blocks:
            stats = block.attn.qkv.get_memory_stats()
            total_active += stats['active_params']
            total_offloaded += stats['offloaded_params']

        gpu_memory = torch.cuda.memory_allocated() / 1024 ** 2  # MB

        print(f"\n📊 Memory Statistics:")
        print(f"  Active LoRA {self.active_lora_idx} params:   {total_active:,}")
        print(f"  Offloaded LoRA params:  {total_offloaded:,}")
        print(f"  GPU Memory Used:        {gpu_memory:.2f} MB")
        print(
            f"  Savings:                {100 * total_offloaded / (total_active + total_offloaded):.1f}% params offloaded")

    def get_active_lora_parameters(self):
        """Get all parameters of currently active LoRA"""
        params = []

        for block in self.blocks:
            params.extend(block.attn.qkv.get_active_lora_parameters())
            params.extend(block.attn.proj.get_active_lora_parameters())
            params.extend(block.mlp.fc1.get_active_lora_parameters())
            params.extend(block.mlp.fc2.get_active_lora_parameters())

        params.extend(self.head.get_active_lora_parameters())

        return params

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
