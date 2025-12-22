# utils/memory_efficient_lora.py - Memory-Efficient Multi-Rank LoRA

import torch
import torch.nn as nn
from typing import Dict, Optional
import os

class MemoryEfficientMultiRankLoRALinear(nn.Module):
    """
    Memory-efficient multi-rank LoRA that only keeps active LoRA in memory.
    Inactive LoRAs are offloaded to CPU or disk.
    """
    
    def __init__(
        self,
        in_features: int,
        out_features: int,
        ranks: list = [4, 8, 16],
        lora_alphas: list = [4, 8, 16],
        lora_dropout: float = 0.1,
        offload_device: str = 'cpu'  # 'cpu' or 'disk'
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.ranks = ranks
        self.num_loras = len(ranks)
        self.offload_device = offload_device
        
        # Base linear layer (frozen)
        self.base_linear = nn.Linear(in_features, out_features)
        for param in self.base_linear.parameters():
            param.requires_grad = False
        
        # Active LoRA index
        self.active_lora_idx = 0
        
        # Storage for LoRA states (only active one in GPU memory)
        self.lora_states = {}  # Store offloaded states
        self.lora_A = None  # Active LoRA A matrix
        self.lora_B = None  # Active LoRA B matrix
        self.dropout = nn.Dropout(lora_dropout)
        
        # Scaling factors (always in memory, minimal overhead)
        self.scalings = [alpha / rank for rank, alpha in zip(ranks, lora_alphas)]
        
        # Initialize all LoRAs and offload
        self._initialize_all_loras()
    
    def _initialize_all_loras(self):
        """Initialize all LoRA matrices and offload to storage"""
        import math
        
        for i, rank in enumerate(self.ranks):
            # Create temporary matrices
            A = torch.empty(self.in_features, rank)
            B = torch.zeros(rank, self.out_features)
            
            # Initialize
            nn.init.kaiming_uniform_(A, a=math.sqrt(5))
            
            # Store state
            self.lora_states[i] = {
                'A': A.clone(),
                'B': B.clone(),
                'rank': rank
            }
        
        # Load first LoRA into active memory
        self._load_lora(0)
    
    def _load_lora(self, lora_idx: int):
        """Load a specific LoRA into GPU memory"""
        if lora_idx not in self.lora_states:
            raise ValueError(f"LoRA {lora_idx} not initialized")
        
        state = self.lora_states[lora_idx]
        
        # Move to GPU and make parameters
        self.lora_A = nn.Parameter(state['A'].to('cuda'))
        self.lora_B = nn.Parameter(state['B'].to('cuda'))
        self.active_lora_idx = lora_idx
    
    def _offload_lora(self, lora_idx: int):
        """Offload a specific LoRA from GPU memory"""
        if lora_idx == self.active_lora_idx:
            # Save current state before offloading
            self.lora_states[lora_idx] = {
                'A': self.lora_A.data.cpu().clone(),
                'B': self.lora_B.data.cpu().clone(),
                'rank': self.ranks[lora_idx]
            }
            
            # Free GPU memory
            del self.lora_A
            del self.lora_B
            self.lora_A = None
            self.lora_B = None
            torch.cuda.empty_cache()
    
    def switch_active_lora(self, new_lora_idx: int):
        """Switch to a different LoRA, offloading current one"""
        if new_lora_idx == self.active_lora_idx:
            return  # Already active
        
        print(f"  💾 Offloading LoRA {self.active_lora_idx} (rank={self.ranks[self.active_lora_idx]})")
        self._offload_lora(self.active_lora_idx)
        
        print(f"  📥 Loading LoRA {new_lora_idx} (rank={self.ranks[new_lora_idx]})")
        self._load_lora(new_lora_idx)
        
        # Force garbage collection
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    def forward(self, x):
        # Base transformation
        base_output = self.base_linear(x)
        
        # Only compute active LoRA
        if self.lora_A is not None and self.lora_B is not None:
            dropped_x = self.dropout(x)
            lora_output = (dropped_x @ self.lora_A) @ self.lora_B
            lora_output = lora_output * self.scalings[self.active_lora_idx]
            return base_output + lora_output
        
        return base_output
    
    def get_active_lora_parameters(self):
        """Get parameters of the currently active LoRA"""
        if self.lora_A is not None and self.lora_B is not None:
            return [self.lora_A, self.lora_B]
        return []
    
    def get_memory_stats(self):
        """Get memory usage statistics"""
        active_params = 0
        if self.lora_A is not None:
            active_params += self.lora_A.numel() + self.lora_B.numel()
        
        offloaded_params = sum(
            state['A'].numel() + state['B'].numel() 
            for i, state in self.lora_states.items() 
            if i != self.active_lora_idx
        )
        
        return {
            'active_params': active_params,
            'offloaded_params': offloaded_params,
            'active_lora': self.active_lora_idx,
            'active_rank': self.ranks[self.active_lora_idx]
        }