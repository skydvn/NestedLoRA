"""
Automatic Experiment Naming System

Generates meaningful experiment names based on:
- Model architecture
- Dataset
- LoRA configuration (rank, alpha)
- Training hyperparameters
- Timestamp

Example names:
- tinyvit_cifar10_lora_r8_a16_lr1e-3_bs128_20231206_143052
- tinyvit_cifar10_lora_r16_a32_lr5e-4_bs256_20231206_143052
"""

import torch
import torch.nn as nn
from datetime import datetime
import hashlib
import os


class ExperimentNamer:
    """
    Smart experiment naming based on configuration
    """

    def __init__(
            self,
            model_name='tinyvit',
            dataset_name='cifar10',
            rank=8,
            lora_alpha=16,
            learning_rate=1e-3,
            batch_size=128,
            num_layers=4,
            embed_dim=128,
            num_heads=4,
            optimizer='adamw',
            scheduler='cosine',
            include_timestamp=True,
            include_hash=False
    ):
        self.model_name = model_name
        self.dataset_name = dataset_name
        self.rank = rank
        self.lora_alpha = lora_alpha
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.include_timestamp = include_timestamp
        self.include_hash = include_hash

    def _format_lr(self, lr):
        """Format learning rate nicely"""
        if lr >= 1e-2:
            return f"{lr:.0e}".replace('e-0', 'e-')
        elif lr >= 1e-4:
            # e.g., 0.001 -> 1e-3
            return f"{lr:.0e}".replace('e-0', 'e-')
        else:
            return f"{lr:.0e}".replace('e-0', 'e-')

    def _get_hash(self, length=6):
        """Generate short hash from configuration"""
        config_str = f"{self.model_name}_{self.dataset_name}_{self.rank}_{self.lora_alpha}"
        config_str += f"_{self.learning_rate}_{self.batch_size}_{self.num_layers}"
        config_str += f"_{self.embed_dim}_{self.num_heads}"

        hash_obj = hashlib.md5(config_str.encode())
        return hash_obj.hexdigest()[:length]

    def generate_name(self, format='detailed'):
        """
        Generate experiment name

        Formats:
        - 'minimal': model_data_r8
        - 'standard': model_data_lora_r8_a16
        - 'detailed': model_data_lora_r8_a16_lr1e-3_bs128
        - 'full': model_data_lora_r8_a16_lr1e-3_bs128_l4_d128_h4
        """

        parts = []

        # Model and dataset (always included)
        parts.append(self.model_name)
        parts.append(self.dataset_name)

        if format == 'minimal':
            parts.append(f'r{self.rank}')

        elif format == 'standard':
            parts.append('lora')
            parts.append(f'r{self.rank}')
            parts.append(f'a{self.lora_alpha}')

        elif format == 'detailed':
            parts.append('lora')
            parts.append(f'r{self.rank}')
            parts.append(f'a{self.lora_alpha}')
            parts.append(f'lr{self._format_lr(self.learning_rate)}')
            parts.append(f'bs{self.batch_size}')

        elif format == 'full':
            parts.append('lora')
            parts.append(f'r{self.rank}')
            parts.append(f'a{self.lora_alpha}')
            parts.append(f'lr{self._format_lr(self.learning_rate)}')
            parts.append(f'bs{self.batch_size}')
            parts.append(f'l{self.num_layers}')
            parts.append(f'd{self.embed_dim}')
            parts.append(f'h{self.num_heads}')

        # Add timestamp if requested
        if self.include_timestamp:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            parts.append(timestamp)

        # Add hash if requested
        if self.include_hash:
            parts.append(self._get_hash())

        return '_'.join(parts)

    def generate_group_name(self):
        """Generate group name for related experiments"""
        return f"{self.model_name}_{self.dataset_name}_lora"

    def generate_tags(self):
        """Generate tags for experiment"""
        tags = [
            self.model_name,
            self.dataset_name,
            'lora',
            f'rank_{self.rank}',
            f'batch_{self.batch_size}',
            self.optimizer,
            self.scheduler
        ]
        return tags


def create_experiment_name(
        model_name='tinyvit',
        dataset_name='cifar10',
        config=None,
        rank=8,
        lora_alpha=16,
        learning_rate=1e-3,
        batch_size=128,
        format='detailed',
        include_timestamp=True
):
    """
    Quick function to create experiment name

    Args:
        model_name: Name of model architecture
        dataset_name: Name of dataset
        config: TinyViTConfig object (optional, extracts params)
        rank: LoRA rank
        lora_alpha: LoRA alpha
        learning_rate: Learning rate
        batch_size: Batch size
        format: Name format ('minimal', 'standard', 'detailed', 'full')
        include_timestamp: Whether to include timestamp

    Returns:
        Experiment name string
    """

    # Extract from config if provided
    if config is not None:
        num_layers = config.num_layers
        embed_dim = config.embed_dim
        num_heads = config.num_heads
    else:
        num_layers = 4
        embed_dim = 128
        num_heads = 4

    namer = ExperimentNamer(
        model_name=model_name,
        dataset_name=dataset_name,
        rank=rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        include_timestamp=include_timestamp
    )

    return namer.generate_name(format=format)


def create_wandb_config(
        model_name='tinyvit',
        dataset_name='cifar10',
        config=None,
        rank=8,
        lora_alpha=16,
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=50,
        optimizer='adamw',
        scheduler='cosine',
        format='detailed'
):
    """
    Create complete W&B configuration with smart naming

    Returns:
        Dictionary with experiment_name, group_name, tags, config
    """

    # Extract from config if provided
    if config is not None:
        num_layers = config.num_layers
        embed_dim = config.embed_dim
        num_heads = config.num_heads
        patch_size = config.patch_size
        dropout = config.dropout
    else:
        num_layers = 4
        embed_dim = 128
        num_heads = 4
        patch_size = 4
        dropout = 0.1

    namer = ExperimentNamer(
        model_name=model_name,
        dataset_name=dataset_name,
        rank=rank,
        lora_alpha=lora_alpha,
        learning_rate=learning_rate,
        batch_size=batch_size,
        num_layers=num_layers,
        embed_dim=embed_dim,
        num_heads=num_heads,
        optimizer=optimizer,
        scheduler=scheduler,
        include_timestamp=True
    )

    return {
        'experiment_name': namer.generate_name(format=format),
        'group_name': namer.generate_group_name(),
        'tags': namer.generate_tags(),
        'config': {
            'model': model_name,
            'dataset': dataset_name,
            'architecture': {
                'num_layers': num_layers,
                'embed_dim': embed_dim,
                'num_heads': num_heads,
                'patch_size': patch_size,
                'dropout': dropout,
            },
            'lora': {
                'rank': rank,
                'alpha': lora_alpha,
                'dropout': 0.1,
            },
            'training': {
                'learning_rate': learning_rate,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'optimizer': optimizer,
                'scheduler': scheduler,
            }
        }
    }


# Example usage functions

def example_naming():
    """Show example experiment names"""

    print("=" * 70)
    print("EXPERIMENT NAMING EXAMPLES")
    print("=" * 70)

    configs = [
        {'rank': 4, 'lora_alpha': 8, 'learning_rate': 1e-3, 'batch_size': 128},
        {'rank': 8, 'lora_alpha': 16, 'learning_rate': 1e-3, 'batch_size': 128},
        {'rank': 16, 'lora_alpha': 32, 'learning_rate': 5e-4, 'batch_size': 256},
        {'rank': 8, 'lora_alpha': 16, 'learning_rate': 1e-3, 'batch_size': 64},
    ]

    for i, cfg in enumerate(configs, 1):
        print(f"\nConfiguration {i}:")
        print(f"  Rank: {cfg['rank']}, Alpha: {cfg['lora_alpha']}")
        print(f"  LR: {cfg['learning_rate']}, Batch: {cfg['batch_size']}")

        # Different formats
        for fmt in ['minimal', 'standard', 'detailed', 'full']:
            name = create_experiment_name(
                **cfg,
                format=fmt,
                include_timestamp=False
            )
            print(f"  {fmt:10s}: {name}")

        # With timestamp
        name = create_experiment_name(**cfg, format='detailed', include_timestamp=True)
        print(f"  {'timestamp':10s}: {name}")


def example_wandb_setup():
    """Show how to use with W&B"""

    print("\n" + "=" * 70)
    print("W&B SETUP EXAMPLE")
    print("=" * 70 + "\n")

    # Create configuration
    wandb_config = create_wandb_config(
        model_name='tinyvit',
        dataset_name='cifar10',
        rank=8,
        lora_alpha=16,
        learning_rate=1e-3,
        batch_size=128,
        num_epochs=50
    )

    print(f"Experiment Name: {wandb_config['experiment_name']}")
    print(f"Group Name:      {wandb_config['group_name']}")
    print(f"Tags:            {', '.join(wandb_config['tags'])}")

    print("\nFull Configuration:")
    import json
    print(json.dumps(wandb_config['config'], indent=2))

    print("\n" + "=" * 70)
    print("USE IN CODE:")
    print("=" * 70)
    print("""
import wandb

# Get smart configuration
wandb_cfg = create_wandb_config(
    rank=8,
    lora_alpha=16,
    learning_rate=1e-3,
    batch_size=128
)

# Initialize W&B
wandb.init(
    project='tiny-vit-lora',
    name=wandb_cfg['experiment_name'],
    group=wandb_cfg['group_name'],
    tags=wandb_cfg['tags'],
    config=wandb_cfg['config']
)
    """)


if __name__ == '__main__':
    # Show examples
    example_naming()
    example_wandb_setup()

    print("\n" + "=" * 70)
    print("COMPARISON EXPERIMENT NAMES")
    print("=" * 70 + "\n")

    # Show how names help with comparison
    ranks = [4, 8, 16, 32]

    print("Comparing different LoRA ranks:")
    for rank in ranks:
        name = create_experiment_name(
            rank=rank,
            lora_alpha=rank * 2,
            format='standard',
            include_timestamp=False
        )
        print(f"  Rank {rank:2d}: {name}")

    print("\nComparing different learning rates:")
    lrs = [1e-2, 5e-3, 1e-3, 5e-4, 1e-4]
    for lr in lrs:
        name = create_experiment_name(
            learning_rate=lr,
            format='detailed',
            include_timestamp=False
        )
        print(f"  LR {lr:.0e}: {name}")

    print("\nComparing different batch sizes:")
    batch_sizes = [64, 128, 256, 512]
    for bs in batch_sizes:
        name = create_experiment_name(
            batch_size=bs,
            format='detailed',
            include_timestamp=False
        )
        print(f"  BS {bs:3d}: {name}")