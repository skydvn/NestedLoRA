# train_nested_lora_cifar10.py
# Complete training script for Nested LoRA on CIFAR-10

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import os
from datetime import datetime
from typing import List, Tuple, Dict
import math

# Import W&B if available
try:
    import wandb

    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    print("WARNING: wandb not installed. Install with: pip install wandb")


# ============================================================================
# Newton-Schulz Orthogonalization
# ============================================================================

def newton_schulz_iteration(W: torch.Tensor, num_iters: int = 5) -> torch.Tensor:
    """
    Newton-Schulz iteration for computing orthogonal approximation

    For non-square matrices, orthogonalizes the column space
    Iteratively computes: Z_{k+1} = (1/2) * Z_k * (3I - Z_k^T Z_k)

    Args:
        W: Input matrix (m × n) where m >= n
        num_iters: Number of iterations (T in algorithm)

    Returns:
        Orthogonalized matrix with orthonormal columns
    """

    m, n = W.shape

    # For numerical stability, normalize by spectral norm approximation
    alpha = 1.0 / (W.norm() + 1e-8)
    Z = W * alpha

    # Identity matrix for column space
    I = torch.eye(n, device=W.device, dtype=W.dtype)

    # Iterative refinement: Z_{k+1} = (1/2) * Z_k * (3I - Z_k^T Z_k)
    for _ in range(num_iters):
        ZtZ = Z.T @ Z  # (n × n)
        Z = 0.5 * Z @ (3 * I - ZtZ)

    return Z


# ============================================================================
# Nested LoRA Linear Layer
# ============================================================================

class NestedLoRALinear(nn.Module):
    """
    Linear layer with Nested Low-Rank Adaptation

    Implements: W_final = W_0 + sum_{l=1}^L M_l

    Where M_l uses Newton-Schulz orthogonalization and Adam-like updates
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            ranks: List[int] = [4, 8, 16],
            lora_alphas: List[int] = [4, 8, 16],
            lora_dropout: float = 0.1,
            newton_schulz_iters: int = 5,
            epsilon: float = 1e-8,
            use_zero_init: bool = True
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.L = len(ranks)
        self.newton_schulz_iters = newton_schulz_iters
        self.epsilon = epsilon

        # Base linear layer (frozen) - W_0
        self.base_linear = nn.Linear(in_features, out_features)
        for param in self.base_linear.parameters():
            param.requires_grad = False

        # LoRA matrices A_l, B_l for each layer l
        self.lora_A = nn.ParameterList()
        self.lora_B = nn.ParameterList()
        self.lora_dropout = nn.ModuleList()
        self.alphas = []

        for rank, alpha in zip(ranks, lora_alphas):
            # A_l: (in_features, rank)
            self.lora_A.append(nn.Parameter(torch.empty(in_features, rank)))
            # B_l: (rank, out_features)
            self.lora_B.append(nn.Parameter(torch.empty(rank, out_features)))
            self.lora_dropout.append(nn.Dropout(lora_dropout))
            self.alphas.append(alpha)

        # Initialize matrices
        for i in range(self.L):
            nn.init.kaiming_uniform_(self.lora_A[i], a=math.sqrt(5))
            if use_zero_init:
                # Zero init for B: model starts as pretrained base
                nn.init.zeros_(self.lora_B[i])
            else:
                # Small random init for testing
                nn.init.normal_(self.lora_B[i], mean=0, std=0.01)

        # Adam-like moment estimates: U_l and V_l
        self.register_buffer('U', torch.zeros(self.L, out_features, in_features))
        self.register_buffer('V', torch.zeros(self.L, out_features, in_features))

        # Cached orthogonalized matrices O_l
        self.register_buffer('O', torch.zeros(self.L, out_features, in_features))

    def compute_dW(self, lora_idx: int) -> torch.Tensor:
        """
        Compute dW_l = B_l^T × A_l^T

        Args:
            lora_idx: Index of LoRA layer

        Returns:
            Weight gradient matrix (out_features, in_features)
        """
        A = self.lora_A[lora_idx]  # (in_features, rank)
        B = self.lora_B[lora_idx]  # (rank, out_features)

        # dW = B^T @ A^T
        dW = B.T @ A.T  # (out_features, in_features)

        return dW

    def update_moments(
            self,
            lora_idx: int,
            dW: torch.Tensor,
            beta1: float = 0.9,
            beta2: float = 0.999
    ):
        """
        Update U_l and V_l (Adam-like moment estimates)

        U_l = U_l + β₁ × dW_l
        V_l = V_l + β₂ × (dW_l)²

        Args:
            lora_idx: Index of LoRA layer
            dW: Gradient dW_l = B_l^T × A_l^T
            beta1: First moment decay (β_l in algorithm)
            beta2: Second moment decay
        """
        # Update first moment: U_l = U_l + β₁ × dW_l
        self.U[lora_idx] = self.U[lora_idx] + beta1 * dW

        # Update second moment: V_l = V_l + β₂ × (dW_l)²
        self.V[lora_idx] = self.V[lora_idx] + beta2 * (dW ** 2)

    def orthogonalize(self, lora_idx: int):
        """
        Apply Newton-Schulz orthogonalization

        O_l = Newton-Schulz_T(U_l)

        Args:
            lora_idx: Index of LoRA layer
        """
        U_l = self.U[lora_idx]

        # Apply Newton-Schulz iteration
        O_l = newton_schulz_iteration(U_l, num_iters=self.newton_schulz_iters)

        self.O[lora_idx] = O_l

    def compute_M(self, lora_idx: int) -> torch.Tensor:
        """
        Compute M_l with Adam-like normalization and orthogonalization

        M_l = (α_l × O_l) / sqrt(V_l + ε)

        Args:
            lora_idx: Index of LoRA layer

        Returns:
            Normalized and orthogonalized update matrix
        """
        # Get orthogonalized matrix
        O_l = self.O[lora_idx]

        # Get second moment
        V_l = self.V[lora_idx]

        # Get alpha
        alpha_l = self.alphas[lora_idx]

        # Compute M_l = (α_l × O_l) / sqrt(V_l + ε)
        M_l = (alpha_l * O_l) / torch.sqrt(V_l + self.epsilon)

        return M_l

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: y = W_0·x + Σ M_l·x

        Args:
            x: Input tensor

        Returns:
            Output tensor
        """
        # Base transformation: W_0·x
        output = self.base_linear(x)

        # Add contributions from all nested LoRA layers
        for l in range(self.L):
            # Get M_l
            M_l = self.compute_M(l)

            # Apply M_l·x
            lora_out = F.linear(x, M_l)
            output = output + lora_out

        return output

    def get_lora_parameters(self, lora_idx: int) -> List[torch.Tensor]:
        """Get parameters for specific LoRA layer"""
        return [self.lora_A[lora_idx], self.lora_B[lora_idx]]


# ============================================================================
# TinyViT with Nested LoRA
# ============================================================================

class TinyViTNestedLoRA(nn.Module):
    """Simple ViT model with Nested LoRA for CIFAR-10"""

    def __init__(
            self,
            img_size: int = 32,
            patch_size: int = 4,
            in_channels: int = 3,
            num_classes: int = 10,
            embed_dim: int = 128,
            depth: int = 6,
            num_heads: int = 4,
            mlp_ratio: float = 2.0,
            ranks: List[int] = [4, 8, 16],
            lora_alphas: List[int] = [4, 8, 16],
            lora_dropout: float = 0.1
    ):
        super().__init__()

        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2

        # Patch embedding
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Positional embedding
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches, embed_dim))

        # Transformer blocks with Nested LoRA
        self.blocks = nn.ModuleList([
            TransformerBlockNestedLoRA(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                ranks=ranks,
                lora_alphas=lora_alphas,
                lora_dropout=lora_dropout
            )
            for _ in range(depth)
        ])

        # Classification head
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # Initialize
        nn.init.trunc_normal_(self.pos_embed, std=0.02)

    def forward(self, x):
        # Patch embedding
        x = self.patch_embed(x)  # (B, embed_dim, H', W')
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, embed_dim)

        # Add positional embedding
        x = x + self.pos_embed

        # Transformer blocks
        for block in self.blocks:
            x = block(x)

        # Classification
        x = self.norm(x)
        x = x.mean(dim=1)  # Global average pooling
        x = self.head(x)

        return x


class TransformerBlockNestedLoRA(nn.Module):
    """Transformer block with Nested LoRA"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            mlp_ratio: float = 2.0,
            ranks: List[int] = [4, 8, 16],
            lora_alphas: List[int] = [4, 8, 16],
            lora_dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttentionNestedLoRA(
            dim, num_heads, ranks, lora_alphas, lora_dropout
        )

        self.norm2 = nn.LayerNorm(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Linear(mlp_hidden_dim, dim)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class MultiHeadAttentionNestedLoRA(nn.Module):
    """Multi-head attention with Nested LoRA on QKV projection"""

    def __init__(
            self,
            dim: int,
            num_heads: int,
            ranks: List[int] = [4, 8, 16],
            lora_alphas: List[int] = [4, 8, 16],
            lora_dropout: float = 0.1
    ):
        super().__init__()

        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5

        # QKV projection with Nested LoRA
        self.qkv = NestedLoRALinear(
            dim, dim * 3,
            ranks=ranks,
            lora_alphas=lora_alphas,
            lora_dropout=lora_dropout
        )

        self.proj = nn.Linear(dim, dim)

    def forward(self, x):
        B, N, C = x.shape

        # QKV projection
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)

        return x


# ============================================================================
# Data Loading
# ============================================================================

def get_cifar10_loaders(batch_size: int = 128, num_workers: int = 4):
    """Get CIFAR-10 data loaders"""

    # Transforms
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    # Datasets
    train_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train
    )

    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test
    )

    # Loaders
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size,
        shuffle=True, num_workers=num_workers, pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset, batch_size=batch_size,
        shuffle=False, num_workers=num_workers, pin_memory=True
    )

    return train_loader, test_loader


# ============================================================================
# Nested LoRA Trainer
# ============================================================================

class NestedLoRATrainer:
    """
    Trainer implementing the Nested LoRA algorithm
    """

    def __init__(
            self,
            model: nn.Module,
            update_steps: List[int] = [1, 2, 4],  # U_1, U_2, U_3
            betas: List[float] = [0.9, 0.999, 0.9999],  # β_1, β_2, β_3
            learning_rates: List[float] = [3e-3, 1e-3, 5e-4],
            weight_decay: float = 0.01,
            device: str = 'cuda'
    ):
        self.model = model.to(device)
        self.device = device
        self.update_steps = update_steps  # U_l
        self.betas = betas  # β_l
        self.L = len(update_steps)

        # Create optimizers for each LoRA layer
        self.optimizers = []
        for l, lr in enumerate(learning_rates):
            params = []
            for module in model.modules():
                if isinstance(module, NestedLoRALinear):
                    params.extend(module.get_lora_parameters(l))

            optimizer = torch.optim.AdamW(params, lr=lr, weight_decay=weight_decay)
            self.optimizers.append(optimizer)

        self.global_iteration = 0

    def train_epoch(
            self,
            train_loader: DataLoader,
            criterion: nn.Module,
            epoch: int
    ) -> Dict:
        """Train for one epoch"""

        self.model.train()

        total_loss = 0
        correct = 0
        total = 0

        # Per-LoRA tracking
        lora_losses = {f'lora{i}': 0.0 for i in range(self.L)}
        lora_updates = {f'lora{i}': 0 for i in range(self.L)}

        pbar = tqdm(train_loader, desc=f'Epoch {epoch}')

        for images, labels in pbar:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Dictionary to store dW for each layer
            dW_dict = {}

            # For each layer l
            for l in range(self.L):
                # Check if this layer should update
                if self.global_iteration % self.update_steps[l] == 0:
                    # Optimize A_l, B_l
                    self.optimizers[l].zero_grad()

                    outputs = self.model(images)
                    loss = criterion(outputs, labels)

                    loss.backward()
                    self.optimizers[l].step()

                    # Compute dW_l and store
                    for module in self.model.modules():
                        if isinstance(module, NestedLoRALinear):
                            dW = module.compute_dW(l)
                            if f'layer_{l}' not in dW_dict:
                                dW_dict[f'layer_{l}'] = []
                            dW_dict[f'layer_{l}'].append(dW)

                    lora_losses[f'lora{l}'] += loss.item()
                    lora_updates[f'lora{l}'] += 1

            # Apply dW updates to moments
            for l in range(self.L):
                if f'layer_{l}' in dW_dict:
                    for module_idx, module in enumerate(self.model.modules()):
                        if isinstance(module, NestedLoRALinear):
                            if module_idx < len(dW_dict[f'layer_{l}']):
                                dW = dW_dict[f'layer_{l}'][module_idx]
                                module.update_moments(l, dW, beta1=self.betas[l])
                                module.orthogonalize(l)

            # Track accuracy
            with torch.no_grad():
                outputs = self.model(images)
                loss = criterion(outputs, labels)
                total_loss += loss.item()

                _, predicted = outputs.max(1)
                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100. * correct / total:.2f}%',
                'iter': self.global_iteration
            })

            self.global_iteration += 1

        # Calculate metrics
        avg_loss = total_loss / len(train_loader)
        accuracy = 100. * correct / total

        # Average LoRA losses
        for l in range(self.L):
            if lora_updates[f'lora{l}'] > 0:
                lora_losses[f'lora{l}'] /= lora_updates[f'lora{l}']

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'lora_losses': lora_losses,
            'lora_updates': lora_updates
        }

    def evaluate(
            self,
            test_loader: DataLoader,
            criterion: nn.Module
    ) -> Dict:
        """Evaluate model"""

        self.model.eval()

        total_loss = 0
        correct = 0
        total = 0

        # Per-class accuracy
        class_correct = [0] * 10
        class_total = [0] * 10

        with torch.no_grad():
            for images, labels in tqdm(test_loader, desc='Evaluating'):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
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

        avg_loss = total_loss / len(test_loader)
        accuracy = 100. * correct / total

        # Per-class accuracy
        class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
                       'dog', 'frog', 'horse', 'ship', 'truck']
        per_class_acc = {}
        for i, name in enumerate(class_names):
            if class_total[i] > 0:
                per_class_acc[name] = 100. * class_correct[i] / class_total[i]

        return {
            'loss': avg_loss,
            'accuracy': accuracy,
            'per_class_acc': per_class_acc
        }


# ============================================================================
# Main Training Function
# ============================================================================

def main(
        # Model config
        embed_dim: int = 128,
        depth: int = 6,
        num_heads: int = 4,
        ranks: List[int] = [4, 8, 16],
        lora_alphas: List[int] = [4, 8, 16],

        # Training config
        batch_size: int = 128,
        num_epochs: int = 100,
        learning_rates: List[float] = [3e-3, 1e-3, 5e-4],
        weight_decay: float = 0.01,
        update_steps: List[int] = [1, 2, 4],
        betas: List[float] = [0.9, 0.999, 0.9999],

        # Logging
        use_wandb: bool = True,
        project_name: str = 'nested-lora-cifar10',
        experiment_name: str = None,
        save_dir: str = './checkpoints'
):
    """Main training function"""

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(save_dir, exist_ok=True)

    # Create experiment name
    if experiment_name is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        experiment_name = f'nested_lora_{timestamp}'

    print(f"\n{'=' * 70}")
    print(f"Nested LoRA Training on CIFAR-10")
    print(f"{'=' * 70}")
    print(f"Device: {device}")
    print(f"Experiment: {experiment_name}")
    print(f"\nModel Configuration:")
    print(f"  Embed dim: {embed_dim}")
    print(f"  Depth: {depth}")
    print(f"  Heads: {num_heads}")
    print(f"  LoRA ranks: {ranks}")
    print(f"  LoRA alphas: {lora_alphas}")
    print(f"\nTraining Configuration:")
    print(f"  Batch size: {batch_size}")
    print(f"  Epochs: {num_epochs}")
    print(f"  Learning rates: {learning_rates}")
    print(f"  Update steps: {update_steps}")
    print(f"  Betas: {betas}")
    print(f"{'=' * 70}\n")

    # Initialize W&B
    if use_wandb and WANDB_AVAILABLE:
        wandb.init(
            project=project_name,
            name=experiment_name,
            config={
                'embed_dim': embed_dim,
                'depth': depth,
                'num_heads': num_heads,
                'ranks': ranks,
                'lora_alphas': lora_alphas,
                'batch_size': batch_size,
                'num_epochs': num_epochs,
                'learning_rates': learning_rates,
                'update_steps': update_steps,
                'betas': betas
            }
        )
        print(f"✓ W&B initialized: {wandb.run.url}\n")

    # Create model
    model = TinyViTNestedLoRA(
        embed_dim=embed_dim,
        depth=depth,
        num_heads=num_heads,
        ranks=ranks,
        lora_alphas=lora_alphas
    )

    print(f"Model created with {sum(p.numel() for p in model.parameters()):,} parameters")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Trainable parameters: {trainable:,} ({100. * trainable / sum(p.numel() for p in model.parameters()):.2f}%)\n")

    # Get data loaders
    print("Loading CIFAR-10 dataset...")
    train_loader, test_loader = get_cifar10_loaders(batch_size=batch_size)
    print(f"  Train samples: {len(train_loader.dataset)}")
    print(f"  Test samples: {len(test_loader.dataset)}")
    print(f"  Batches per epoch: {len(train_loader)}\n")

    # Create trainer
    trainer = NestedLoRATrainer(
        model=model,
        update_steps=update_steps,
        betas=betas,
        learning_rates=learning_rates,
        weight_decay=weight_decay,
        device=device
    )

    # Loss function
    criterion = nn.CrossEntropyLoss()

    # Training loop
    best_acc = 0

    print(f"{'=' * 70}")
    print(f"Starting Training")
    print(f"{'=' * 70}\n")

    for epoch in range(1, num_epochs + 1):
        # Train
        train_metrics = trainer.train_epoch(train_loader, criterion, epoch)

        # Evaluate
        test_metrics = trainer.evaluate(test_loader, criterion)

        # Print summary
        print(f'\nEpoch {epoch}/{num_epochs}:')
        print(f"  Train: Loss={train_metrics['loss']:.4f}, Acc={train_metrics['accuracy']:.2f}%")
        print(f"  Test:  Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.2f}%")
        print(f"  LoRA Updates: {train_metrics['lora_updates']}")

        # Log to W&B
        if use_wandb and WANDB_AVAILABLE:
            log_dict = {
                'epoch': epoch,
                'train_loss': train_metrics['loss'],
                'train_acc': train_metrics['accuracy'],
                'test_loss': test_metrics['loss'],
                'test_acc': test_metrics['accuracy'],
                'global_iteration': trainer.global_iteration
            }

            # Add LoRA losses
            for key, val in train_metrics['lora_losses'].items():
                log_dict[f'{key}_loss'] = val

            # Add per-class accuracy
            for name, acc in test_metrics['per_class_acc'].items():
                log_dict[f'test_acc_{name}'] = acc

            wandb.log(log_dict)

        # Save best model
        if test_metrics['accuracy'] > best_acc:
            best_acc = test_metrics['accuracy']

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'best_acc': best_acc,
                'config': {
                    'embed_dim': embed_dim,
                    'depth': depth,
                    'num_heads': num_heads,
                    'ranks': ranks,
                    'lora_alphas': lora_alphas
                }
            }

            checkpoint_path = os.path.join(save_dir, f'{experiment_name}_best.pth')
            torch.save(checkpoint, checkpoint_path)
            print(f"  ✓ Saved best model: {best_acc:.2f}%")

    # Final summary
    print(f'\n{"=" * 70}')
    print(f'Training Complete!')
    print(f'{"=" * 70}')
    print(f'Best Test Accuracy: {best_acc:.2f}%')
    print(f'Total Iterations: {trainer.global_iteration}')
    print(f'Checkpoints saved to: {save_dir}')

    if use_wandb and WANDB_AVAILABLE:
        wandb.run.summary['best_accuracy'] = best_acc
        wandb.finish()

    print(f'{"=" * 70}\n')

    return model, best_acc


# ============================================================================
# Entry Point
# ============================================================================

if __name__ == '__main__':
    # Run training
    model, best_acc = main(
        # Model config
        embed_dim=128,
        depth=6,
        num_heads=4,
        ranks=[4, 8, 16],
        lora_alphas=[4, 8, 16],

        # Training config
        batch_size=128,
        num_epochs=100,
        learning_rates=[3e-3, 1e-3, 5e-4],
        update_steps=[1, 2, 4],
        betas=[0.9, 0.999, 0.9999],

        # Logging
        use_wandb=True,
        project_name = "tiny-vit-lora-cifar10" ,
        save_dir='checkpoints'
    )

    print(f'\n🎉 Training finished! Best accuracy: {best_acc:.2f}%')