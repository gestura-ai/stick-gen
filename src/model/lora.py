"""
LoRA (Low-Rank Adaptation) module for efficient fine-tuning.

This module provides:
- LoRALinear: A wrapper that adds low-rank adapters to nn.Linear layers
- inject_lora_adapters: Injects LoRA into target modules of a model
- freeze_base_model: Freezes all non-LoRA parameters
- get_lora_parameters: Returns only LoRA parameters for optimizer
- merge_lora_weights: Merges LoRA weights into base model for inference

Reference: https://arxiv.org/abs/2106.09685
"""

import math
import re
from typing import List, Optional, Iterator

import torch
import torch.nn as nn
import torch.nn.functional as F


class LoRALinear(nn.Module):
    """Linear layer with Low-Rank Adaptation (LoRA).
    
    Wraps an existing nn.Linear and adds trainable low-rank matrices A and B.
    Output: y = Wx + (alpha/rank) * B @ A @ x
    
    Args:
        base_layer: The original nn.Linear layer to wrap
        rank: Rank of the low-rank decomposition (default: 8)
        alpha: Scaling factor for LoRA output (default: 16)
        dropout: Dropout probability for LoRA path (default: 0.0)
    """
    
    def __init__(
        self,
        base_layer: nn.Linear,
        rank: int = 8,
        alpha: float = 16.0,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        
        self.base_layer = base_layer
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        in_features = base_layer.in_features
        out_features = base_layer.out_features
        
        # LoRA matrices: A projects down, B projects up
        # Initialize A with Kaiming uniform, B with zeros
        # This ensures LoRA starts as identity (no change to base model)
        self.lora_A = nn.Parameter(torch.zeros(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))
        
        # Initialize A with scaled random values
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        # B stays zero-initialized
        
        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()
        
        # Freeze the base layer
        for param in self.base_layer.parameters():
            param.requires_grad = False
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass with LoRA adaptation."""
        # Base layer output (frozen)
        base_output = self.base_layer(x)
        
        # LoRA path: dropout -> A -> B -> scale
        lora_input = self.dropout(x)
        lora_output = F.linear(lora_input, self.lora_A)  # [*, rank]
        lora_output = F.linear(lora_output, self.lora_B)  # [*, out_features]
        lora_output = lora_output * self.scaling
        
        return base_output + lora_output
    
    def merge_weights(self) -> None:
        """Merge LoRA weights into base layer for inference."""
        with torch.no_grad():
            # W' = W + (alpha/rank) * B @ A
            delta_w = (self.lora_B @ self.lora_A) * self.scaling
            self.base_layer.weight.add_(delta_w)
    
    @property
    def weight(self) -> torch.Tensor:
        """Return effective weight (base + LoRA)."""
        return self.base_layer.weight + (self.lora_B @ self.lora_A) * self.scaling
    
    @property
    def bias(self) -> Optional[torch.Tensor]:
        """Return bias from base layer."""
        return self.base_layer.bias


def _get_submodule(model: nn.Module, target: str) -> nn.Module:
    """Get a submodule by dot-separated path."""
    atoms = target.split(".")
    mod = model
    for atom in atoms:
        if hasattr(mod, atom):
            mod = getattr(mod, atom)
        elif atom.isdigit():
            mod = mod[int(atom)]
        else:
            raise AttributeError(f"Module has no attribute '{atom}'")
    return mod


def _set_submodule(model: nn.Module, target: str, new_module: nn.Module) -> None:
    """Set a submodule by dot-separated path."""
    atoms = target.split(".")
    parent = model
    for atom in atoms[:-1]:
        if hasattr(parent, atom):
            parent = getattr(parent, atom)
        elif atom.isdigit():
            parent = parent[int(atom)]
        else:
            raise AttributeError(f"Module has no attribute '{atom}'")
    
    final_attr = atoms[-1]
    if final_attr.isdigit():
        parent[int(final_attr)] = new_module
    else:
        setattr(parent, final_attr, new_module)


def inject_lora_adapters(
    model: nn.Module,
    target_modules: List[str],
    rank: int = 8,
    alpha: float = 16.0,
    dropout: float = 0.0,
) -> int:
    """Inject LoRA adapters into target modules of a model.
    
    Args:
        model: The model to modify
        target_modules: List of module name patterns to target (supports regex)
        rank: LoRA rank
        alpha: LoRA alpha scaling factor
        dropout: LoRA dropout probability
    
    Returns:
        Number of layers modified
    """
    modified_count = 0
    
    # Compile patterns
    patterns = [re.compile(p) for p in target_modules]
    
    # Find all Linear layers matching patterns
    targets_to_replace = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            for pattern in patterns:
                if pattern.search(name):
                    targets_to_replace.append(name)
                    break
    
    # Replace with LoRA layers
    for target in targets_to_replace:
        base_layer = _get_submodule(model, target)
        lora_layer = LoRALinear(base_layer, rank=rank, alpha=alpha, dropout=dropout)
        _set_submodule(model, target, lora_layer)
        modified_count += 1

    return modified_count


def freeze_base_model(model: nn.Module) -> int:
    """Freeze all non-LoRA parameters in the model.

    Args:
        model: The model to freeze

    Returns:
        Number of parameters frozen
    """
    frozen_count = 0
    for name, param in model.named_parameters():
        # Keep LoRA parameters trainable
        if "lora_A" in name or "lora_B" in name:
            param.requires_grad = True
        else:
            param.requires_grad = False
            frozen_count += 1
    return frozen_count


def get_lora_parameters(model: nn.Module) -> Iterator[nn.Parameter]:
    """Get only LoRA parameters for optimizer.

    Args:
        model: The model with LoRA adapters

    Yields:
        LoRA parameters (lora_A and lora_B)
    """
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            yield param


def count_lora_parameters(model: nn.Module) -> tuple:
    """Count trainable and total parameters.

    Args:
        model: The model to count

    Returns:
        Tuple of (trainable_params, total_params, lora_params)
    """
    total_params = 0
    trainable_params = 0
    lora_params = 0

    for name, param in model.named_parameters():
        total_params += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
        if "lora_A" in name or "lora_B" in name:
            lora_params += param.numel()

    return trainable_params, total_params, lora_params


def merge_lora_weights(model: nn.Module) -> int:
    """Merge all LoRA weights into base layers for inference.

    After merging, the model can be used without LoRA overhead.

    Args:
        model: The model with LoRA adapters

    Returns:
        Number of layers merged
    """
    merged_count = 0
    for module in model.modules():
        if isinstance(module, LoRALinear):
            module.merge_weights()
            merged_count += 1
    return merged_count


def get_lora_state_dict(model: nn.Module) -> dict:
    """Extract only LoRA parameters from model state dict.

    Useful for saving/loading just the LoRA adapters.

    Args:
        model: The model with LoRA adapters

    Returns:
        State dict containing only LoRA parameters
    """
    lora_state = {}
    for name, param in model.named_parameters():
        if "lora_A" in name or "lora_B" in name:
            lora_state[name] = param.data.clone()
    return lora_state


def load_lora_state_dict(model: nn.Module, lora_state: dict) -> int:
    """Load LoRA parameters into model.

    Args:
        model: The model with LoRA adapters
        lora_state: State dict containing LoRA parameters

    Returns:
        Number of parameters loaded
    """
    loaded_count = 0
    model_state = model.state_dict()
    for name, param in lora_state.items():
        if name in model_state:
            model_state[name].copy_(param)
            loaded_count += 1
    return loaded_count

