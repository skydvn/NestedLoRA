"""
LoRA Merging Utility

Complete implementation for merging LoRA adapters into base model weights.
Supports:
- Single-rank LoRA merging
- Multi-rank LoRA merging
- Merge and unmerge operations
- Save merged/separate models
- Verification and testing

Usage:
    python merge_lora.py --model path/to/model.pth --output merged_model.pth
"""

import torch
import torch.nn as nn
import argparse
import os
from typing import Dict, List, Optional, Tuple
from collections import OrderedDict
import copy


class LoRAMerger:
    """
    Utility class for merging LoRA adapters into base model weights
    """

    def __init__(self, model, verbose: bool = True):
        """
        Args:
            model: PyTorch model with LoRA adapters
            verbose: Print merge progress
        """
        self.model = model
        self.verbose = verbose
        self.original_state = None
        self.is_merged = False
        self.merge_history = []

    def _print(self, msg: str):
        """Print if verbose"""
        if self.verbose:
            print(msg)

    def save_original_state(self):
        """Save original model state before any merging"""
        if self.original_state is None:
            self.original_state = copy.deepcopy(self.model.state_dict())
            self._print("✓ Saved original model state")

    def merge_lora_single_rank(self, scaling: float = 1.0) -> int:
        """
        Merge single-rank LoRA adapters into base weights

        For each linear layer with LoRA:
            W_merged = W_base + (B.T @ A.T) * scaling

        Args:
            scaling: Scale factor for LoRA (default: 1.0)

        Returns:
            Number of layers merged
        """
        self._print("\n" + "=" * 70)
        self._print("MERGING SINGLE-RANK LoRA")
        self._print("=" * 70)

        # Save original if not saved
        self.save_original_state()

        merged_count = 0

        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Check if it's single-rank (not a list)
                if not isinstance(module.lora_A, (list, nn.ModuleList)):
                    # Get base weight
                    W_base = module.weight.data

                    # Get LoRA matrices
                    A = module.lora_A.data  # [in_features, rank]
                    B = module.lora_B.data  # [rank, out_features]

                    # Get scaling (might be stored as attribute)
                    if hasattr(module, 'scaling'):
                        scale = module.scaling
                    else:
                        scale = scaling

                    # Compute delta weight: B.T @ A.T
                    # Shape: [out_features, rank] @ [rank, in_features] = [out_features, in_features]
                    delta_W = (B.T @ A.T) * scale

                    # Merge into base weight
                    module.weight.data = W_base + delta_W

                    self._print(f"  ✓ Merged LoRA in {name}")
                    self._print(f"    Base shape: {W_base.shape}, LoRA rank: {A.shape[1]}")
                    self._print(f"    Delta norm: {delta_W.norm().item():.6f}")

                    merged_count += 1
                    self.merge_history.append({
                        'layer': name,
                        'rank': A.shape[1],
                        'scaling': scale
                    })

        self._print(f"\n✓ Merged {merged_count} LoRA layers")
        self.is_merged = True

        return merged_count

    def merge_lora_multi_rank(self, scalings: Optional[List[float]] = None) -> int:
        """
        Merge multi-rank LoRA adapters into base weights

        For each linear layer with multiple LoRAs:
            W_merged = W_base + Σ(B_i.T @ A_i.T * scaling_i)

        Args:
            scalings: List of scale factors for each LoRA rank

        Returns:
            Number of layers merged
        """
        self._print("\n" + "=" * 70)
        self._print("MERGING MULTI-RANK LoRA")
        self._print("=" * 70)

        # Save original if not saved
        self.save_original_state()

        merged_count = 0

        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A') and hasattr(module, 'lora_B'):
                # Check if it's multi-rank (list or ModuleList)
                if isinstance(module.lora_A, (list, nn.ModuleList)):
                    num_loras = len(module.lora_A)

                    # Get base weight
                    W_base = module.weight.data

                    # Accumulate deltas from all LoRAs
                    delta_W_total = torch.zeros_like(W_base)

                    ranks = []
                    for i in range(num_loras):
                        # Get LoRA matrices
                        A = module.lora_A[i].data  # [in_features, rank_i]
                        B = module.lora_B[i].data  # [rank_i, out_features]

                        # Get scaling
                        if hasattr(module, 'scalings'):
                            scale = module.scalings[i]
                        elif scalings and i < len(scalings):
                            scale = scalings[i]
                        else:
                            scale = 1.0

                        # Compute delta for this LoRA
                        delta_W = (B.T @ A.T) * scale
                        delta_W_total += delta_W

                        ranks.append(A.shape[1])

                        self._print(f"  → LoRA {i + 1}: rank={A.shape[1]}, scale={scale:.3f}, "
                                    f"norm={delta_W.norm().item():.6f}")

                    # Merge total delta into base weight
                    module.weight.data = W_base + delta_W_total

                    self._print(f"  ✓ Merged {num_loras} LoRAs in {name}")
                    self._print(f"    Ranks: {ranks}")
                    self._print(f"    Total delta norm: {delta_W_total.norm().item():.6f}")
                    self._print("")

                    merged_count += 1
                    self.merge_history.append({
                        'layer': name,
                        'ranks': ranks,
                        'num_loras': num_loras
                    })

        self._print(f"✓ Merged {merged_count} multi-rank LoRA layers")
        self.is_merged = True

        return merged_count

    def merge_lora_auto(self, scalings: Optional[List[float]] = None) -> int:
        """
        Automatically detect and merge LoRA (single or multi-rank)

        Args:
            scalings: Optional scale factors for multi-rank

        Returns:
            Number of layers merged
        """
        # Check if model has multi-rank LoRA
        has_multi_rank = False
        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A'):
                if isinstance(module.lora_A, (list, nn.ModuleList)):
                    has_multi_rank = True
                    break

        if has_multi_rank:
            return self.merge_lora_multi_rank(scalings)
        else:
            return self.merge_lora_single_rank()

    def remove_lora_parameters(self):
        """
        Remove LoRA parameters from model after merging

        This reduces model size and makes it a standard PyTorch model
        """
        self._print("\n" + "=" * 70)
        self._print("REMOVING LoRA PARAMETERS")
        self._print("=" * 70)

        removed_count = 0

        for name, module in self.model.named_modules():
            if hasattr(module, 'lora_A'):
                del module.lora_A
                removed_count += 1
                self._print(f"  ✓ Removed lora_A from {name}")

            if hasattr(module, 'lora_B'):
                del module.lora_B
                self._print(f"  ✓ Removed lora_B from {name}")

            if hasattr(module, 'scaling'):
                del module.scaling
                self._print(f"  ✓ Removed scaling from {name}")

            if hasattr(module, 'scalings'):
                del module.scalings
                self._print(f"  ✓ Removed scalings from {name}")

        self._print(f"\n✓ Removed LoRA parameters from {removed_count} layers")

        return removed_count

    def unmerge_lora(self):
        """
        Restore original model state (unmerge LoRA)

        This requires that save_original_state() was called before merging
        """
        if self.original_state is None:
            raise ValueError("Cannot unmerge: original state not saved!")

        self._print("\n" + "=" * 70)
        self._print("UNMERGING LoRA (Restoring Original Weights)")
        self._print("=" * 70)

        self.model.load_state_dict(self.original_state)
        self.is_merged = False

        self._print("✓ Restored original model state")

    def verify_merge(self, test_input: torch.Tensor, tolerance: float = 1e-5) -> bool:
        """
        Verify that merged model produces same output as original

        Args:
            test_input: Sample input tensor for testing
            tolerance: Maximum allowed difference

        Returns:
            True if outputs match within tolerance
        """
        if self.original_state is None:
            self._print("⚠️  Cannot verify: original state not saved")
            return False

        self._print("\n" + "=" * 70)
        self._print("VERIFYING MERGE")
        self._print("=" * 70)

        # Get output from merged model
        self.model.eval()
        with torch.no_grad():
            merged_output = self.model(test_input)

        # Restore original and get output
        self.unmerge_lora()
        self.model.eval()
        with torch.no_grad():
            original_output = self.model(test_input)

        # Re-merge
        self.merge_lora_auto()

        # Compare outputs
        diff = (merged_output - original_output).abs().max().item()

        self._print(f"\nOutput comparison:")
        self._print(f"  Max absolute difference: {diff:.2e}")
        self._print(f"  Tolerance: {tolerance:.2e}")

        if diff < tolerance:
            self._print(f"  ✅ PASS - Outputs match within tolerance")
            return True
        else:
            self._print(f"  ❌ FAIL - Outputs differ beyond tolerance")
            return False

    def get_parameter_stats(self) -> Dict:
        """Get statistics about model parameters"""
        total_params = 0
        lora_params = 0
        base_params = 0

        for name, param in self.model.named_parameters():
            total_params += param.numel()
            if 'lora' in name.lower():
                lora_params += param.numel()
            else:
                base_params += param.numel()

        return {
            'total': total_params,
            'base': base_params,
            'lora': lora_params,
            'lora_percentage': (lora_params / total_params * 100) if total_params > 0 else 0
        }

    def save_merged_model(self, path: str, remove_lora: bool = True):
        """
        Save merged model to file

        Args:
            path: Output file path
            remove_lora: If True, remove LoRA parameters before saving
        """
        if not self.is_merged:
            self._print("⚠️  Model not merged yet, merging now...")
            self.merge_lora_auto()

        if remove_lora:
            self.remove_lora_parameters()

        # Save model
        torch.save(self.model.state_dict(), path)

        # Get file size
        size_mb = os.path.getsize(path) / (1024 * 1024)

        self._print(f"\n✓ Saved merged model to {path}")
        self._print(f"  File size: {size_mb:.2f} MB")

    def save_separate(self, base_path: str, lora_path: str):
        """
        Save base model and LoRA adapters separately

        Args:
            base_path: Path for base model weights
            lora_path: Path for LoRA adapter weights
        """
        # Get state dict
        state_dict = self.model.state_dict()

        # Split into base and LoRA
        base_state = OrderedDict()
        lora_state = OrderedDict()

        for key, value in state_dict.items():
            if 'lora' in key.lower():
                lora_state[key] = value
            else:
                base_state[key] = value

        # Save both
        torch.save(base_state, base_path)
        torch.save(lora_state, lora_path)

        # Get file sizes
        base_size = os.path.getsize(base_path) / (1024 * 1024)
        lora_size = os.path.getsize(lora_path) / (1024 * 1024)

        self._print(f"\n✓ Saved separate models:")
        self._print(f"  Base model: {base_path} ({base_size:.2f} MB)")
        self._print(f"  LoRA adapters: {lora_path} ({lora_size:.2f} MB)")
        self._print(f"  Total: {base_size + lora_size:.2f} MB")

    def print_summary(self):
        """Print summary of merge operation"""
        self._print("\n" + "=" * 70)
        self._print("MERGE SUMMARY")
        self._print("=" * 70)

        stats = self.get_parameter_stats()

        self._print(f"\nParameter counts:")
        self._print(f"  Total parameters: {stats['total']:,}")
        self._print(f"  Base parameters: {stats['base']:,}")
        self._print(f"  LoRA parameters: {stats['lora']:,}")
        self._print(f"  LoRA percentage: {stats['lora_percentage']:.2f}%")

        self._print(f"\nMerge status:")
        self._print(f"  Is merged: {self.is_merged}")
        self._print(f"  Layers merged: {len(self.merge_history)}")

        if self.merge_history:
            self._print(f"\nMerge history:")
            for item in self.merge_history[:5]:  # Show first 5
                if 'ranks' in item:
                    self._print(f"  • {item['layer']}: {item['num_loras']} LoRAs "
                                f"(ranks={item['ranks']})")
                else:
                    self._print(f"  • {item['layer']}: rank={item['rank']}, "
                                f"scaling={item['scaling']:.3f}")

            if len(self.merge_history) > 5:
                self._print(f"  ... and {len(self.merge_history) - 5} more")

        self._print("=" * 70)


def merge_lora_model(
        model_path: str,
        output_path: str,
        multi_rank: bool = False,
        remove_lora: bool = True,
        verify: bool = False,
        test_input: Optional[torch.Tensor] = None
):
    """
    Convenience function to merge LoRA model

    Args:
        model_path: Path to model with LoRA
        output_path: Path for output merged model
        multi_rank: If True, use multi-rank merging
        remove_lora: If True, remove LoRA parameters
        verify: If True, verify merge correctness
        test_input: Sample input for verification
    """
    # Load model
    print(f"Loading model from {model_path}...")
    checkpoint = torch.load(model_path, map_location='cpu')

    # Create model (you need to adjust this for your specific model)
    # For now, assume the checkpoint is a state dict
    # In practice, you'd load your actual model architecture

    print("Creating LoRA merger...")
    merger = LoRAMerger(model=None, verbose=True)  # Pass actual model here

    # Merge
    if multi_rank:
        merger.merge_lora_multi_rank()
    else:
        merger.merge_lora_auto()

    # Verify if requested
    if verify and test_input is not None:
        merger.verify_merge(test_input)

    # Save
    merger.save_merged_model(output_path, remove_lora=remove_lora)

    # Print summary
    merger.print_summary()

    return merger


# ============================================================================
# Command Line Interface
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Merge LoRA adapters into base model weights'
    )

    parser.add_argument(
        '--model', type=str, required=True,
        help='Path to model checkpoint with LoRA'
    )
    parser.add_argument(
        '--output', type=str, required=True,
        help='Path for output merged model'
    )
    parser.add_argument(
        '--multi-rank', action='store_true',
        help='Use multi-rank LoRA merging'
    )
    parser.add_argument(
        '--keep-lora', action='store_true',
        help='Keep LoRA parameters in saved model'
    )
    parser.add_argument(
        '--save-separate', action='store_true',
        help='Save base and LoRA separately'
    )
    parser.add_argument(
        '--base-path', type=str,
        help='Path for base model (when using --save-separate)'
    )
    parser.add_argument(
        '--lora-path', type=str,
        help='Path for LoRA adapters (when using --save-separate)'
    )
    parser.add_argument(
        '--verify', action='store_true',
        help='Verify merge correctness'
    )

    args = parser.parse_args()

    # Validate arguments
    if args.save_separate and (not args.base_path or not args.lora_path):
        parser.error("--save-separate requires --base-path and --lora-path")

    print("=" * 70)
    print("LoRA MERGING UTILITY")
    print("=" * 70)
    print(f"\nInput: {args.model}")
    print(f"Output: {args.output}")
    print(f"Multi-rank: {args.multi_rank}")
    print(f"Remove LoRA: {not args.keep_lora}")
    print(f"Save separate: {args.save_separate}")
    print("")

    # TODO: Load your actual model here
    # model = load_your_model(args.model)

    # For now, print instructions
    print("⚠️  To use this script, you need to:")
    print("   1. Implement model loading for your specific architecture")
    print("   2. Pass the loaded model to LoRAMerger")
    print("   3. Run the merge operations")
    print("")
    print("Example usage:")
    print("")
    print("  from train_s_nlora_gpm import YourModel")
    print("  model = YourModel.load(args.model)")
    print("  merger = LoRAMerger(model)")
    print("  merger.merge_lora_auto()")
    print("  merger.save_merged_model(args.output)")


if __name__ == '__main__':
    main()