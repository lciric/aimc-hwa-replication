"""
Analog neural network layers with Straight-Through Estimator (STE).

Implements quantization-aware training with hardware noise injection.
The STE trick lets gradients flow through non-differentiable quantization.

Key insight from the paper: the backward pass uses clean gradients,
but the forward pass sees quantized + noisy weights. This trains the
network to be robust to hardware imperfections.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Function
from typing import Optional, Tuple
import math

from .physics import PCMPhysicsEngine


class StraightThroughEstimator(Function):
    """
    STE for weight quantization with optional noise and drift.
    
    Forward: quantize → noise → drift (inference only)
    Backward: gradient passes through unchanged (∂L/∂w_clean)
    
    This is THE critical component for HWA training. The network
    learns with full precision gradients but sees what hardware sees.
    """
    
    @staticmethod
    def forward(ctx, weight: torch.Tensor, gamma: torch.Tensor, 
                alpha: torch.Tensor, physics: Optional[PCMPhysicsEngine],
                t_inference: float, training: bool,
                drop_connect_prob: float = 0.0) -> torch.Tensor:
        """
        Args:
            weight: raw trainable weights
            gamma: quantization levels (typically 256 for 8-bit)
            alpha: per-layer scale factor (auto-computed or learned)
            physics: PCM physics engine (None for digital baseline)
            t_inference: inference time in seconds (0 during training)
            training: whether we're in training mode
            drop_connect_prob: probability of dropping weights to 0
            
        Returns:
            Effective weights after quantization/noise/drift
        """
        # Scale to [-1, 1] range (analog weight range)
        # Adding eps to avoid division by zero if alpha collapses
        w_scaled = weight / (alpha + 1e-9)
        
        # Uniform quantization to gamma levels
        # Example: gamma=256 → 255 levels → resolution of 1/255 ≈ 0.004
        levels = gamma - 1
        w_quant = torch.clamp(
            torch.round(w_scaled * levels) / levels,
            -1.0, 1.0
        )
        
        # Drop-connect: randomly zero out weights (1% in paper)
        # This regularization helps with PCM stuck-at-faults
        if training and drop_connect_prob > 0:
            mask = torch.bernoulli(
                torch.full_like(w_quant, 1.0 - drop_connect_prob)
            )
            w_quant = w_quant * mask
        
        # Apply PCM physics
        if physics is not None:
            w_noisy = physics.apply_programming_noise(w_quant)
            if not training and t_inference > 0:
                w_noisy = physics.apply_drift(w_noisy, t_inference)
        else:
            w_noisy = w_quant
        
        # Scale back to original magnitude
        return w_noisy * alpha
    
    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> Tuple:
        """
        STE: gradient flows through as if quantization wasn't there.
        
        This is mathematically questionable but empirically works great.
        The intuition: we want to update the underlying weights even
        though forward used quantized versions.
        """
        return grad_output, None, None, None, None, None, None


class AnalogLinear(nn.Linear):
    """
    Linear layer with analog weight representation.
    
    Wraps standard Linear with:
      - Per-layer scale factor (alpha)
      - Quantization level (gamma)
      - PCM physics injection
      - Drop-connect option
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 bias: bool = True, physics: Optional[PCMPhysicsEngine] = None,
                 gamma: int = 256, drop_connect_prob: float = 0.0):
        super().__init__(in_features, out_features, bias)
        
        self.physics = physics
        self.drop_connect_prob = drop_connect_prob
        
        # Alpha: per-layer scale factor
        # Initialized to 1.0, can be learned or set via caws
        self.alpha = nn.Parameter(torch.tensor(1.0))
        
        # Gamma: quantization levels (not trainable)
        # 256 = 8-bit, matches typical ADC resolution
        self.register_buffer('gamma', torch.tensor(float(gamma)))
        
        # Inference time (set before evaluation)
        self.t_inference = 0.0
    
    def set_alpha(self, value: float) -> None:
        """Set scale factor (used by caws initialization)."""
        self.alpha.data.fill_(value)
    
    def set_inference_time(self, t: float) -> None:
        """Set drift time for inference."""
        self.t_inference = t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Get effective weights through STE
        w_eff = StraightThroughEstimator.apply(
            self.weight, self.gamma, self.alpha, self.physics,
            self.t_inference, self.training, self.drop_connect_prob
        )
        return F.linear(x, w_eff, self.bias)


class AnalogConv2d(nn.Conv2d):
    """
    Conv2d layer with analog weight representation.
    
    Same philosophy as AnalogLinear but for convolutions.
    Most analog accelerators tile conv weights onto crossbars.
    """
    
    def __init__(self, in_channels: int, out_channels: int,
                 kernel_size: int, stride: int = 1, padding: int = 0,
                 bias: bool = True, physics: Optional[PCMPhysicsEngine] = None,
                 gamma: int = 256, drop_connect_prob: float = 0.0):
        super().__init__(in_channels, out_channels, kernel_size, 
                         stride=stride, padding=padding, bias=bias)
        
        self.physics = physics
        self.drop_connect_prob = drop_connect_prob
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.register_buffer('gamma', torch.tensor(float(gamma)))
        self.t_inference = 0.0
    
    def set_alpha(self, value: float) -> None:
        self.alpha.data.fill_(value)
    
    def set_inference_time(self, t: float) -> None:
        self.t_inference = t
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w_eff = StraightThroughEstimator.apply(
            self.weight, self.gamma, self.alpha, self.physics,
            self.t_inference, self.training, self.drop_connect_prob
        )
        return F.conv2d(x, w_eff, self.bias, self.stride, self.padding)


# ============================================================================
# HWA Training Techniques
# ============================================================================

def compute_caws_alpha(layer: nn.Module) -> float:
    """
    Compute auto-weight-scaling factor using Xavier/He heuristics.
    
    α = √(3 / fan_in) for Linear layers
    α = √(3 / (k² × c_in)) for Conv layers
    
    This ensures quantized weights span a reasonable dynamic range
    after initialization. Without this, some layers might be
    underutilizing the [-1, 1] conductance range.
    """
    if isinstance(layer, nn.Linear):
        fan_in = layer.in_features
    elif isinstance(layer, nn.Conv2d):
        fan_in = layer.in_channels * layer.kernel_size[0] * layer.kernel_size[1]
    else:
        raise ValueError(f"Unsupported layer type: {type(layer)}")
    
    return math.sqrt(3.0 / fan_in)


def apply_caws(model: nn.Module) -> None:
    """
    Apply Crossbar-Aware Weight Scaling to all analog layers.
    
    This is called once after model initialization.
    """
    for name, module in model.named_modules():
        if isinstance(module, (AnalogLinear, AnalogConv2d)):
            alpha = compute_caws_alpha(module)
            module.set_alpha(alpha)


def remap_weights(layer: nn.Module) -> None:
    """
    Remap weights to fully utilize [-1, 1] range.
    
    If max(|w|) < 1, we're wasting dynamic range.
    Rescale so max(|w|) = 1 and adjust alpha accordingly.
    
    NOTE: This is DISABLED in final SOTA version (remap_interval=0).
    The paper found that periodic remapping during training hurts
    convergence when combined with knowledge distillation.
    """
    if not isinstance(layer, (AnalogLinear, AnalogConv2d)):
        return
    
    with torch.no_grad():
        w = layer.weight.data
        w_max = w.abs().max()
        
        if w_max > 1e-6:  # avoid div by zero
            # Scale weights to [-1, 1]
            layer.weight.data = w / w_max
            # Adjust alpha to preserve effective weights
            layer.alpha.data = layer.alpha.data * w_max


def remap_all_weights(model: nn.Module) -> None:
    """Apply weight remapping to all analog layers."""
    for module in model.modules():
        if isinstance(module, (AnalogLinear, AnalogConv2d)):
            remap_weights(module)


def set_drop_connect_prob(model: nn.Module, prob: float) -> None:
    """Set drop-connect probability for all analog layers."""
    for module in model.modules():
        if isinstance(module, (AnalogLinear, AnalogConv2d)):
            module.drop_connect_prob = prob


def set_inference_time(model: nn.Module, t: float) -> None:
    """Set inference time for all analog layers (for drift simulation)."""
    for module in model.modules():
        if isinstance(module, (AnalogLinear, AnalogConv2d)):
            module.set_inference_time(t)


# Debug helper
def check_weight_stats(model: nn.Module, prefix: str = "") -> None:
    """Print weight statistics for debugging. Remove in production."""
    for name, module in model.named_modules():
        if isinstance(module, (AnalogLinear, AnalogConv2d)):
            w = module.weight.data
            print(f"{prefix}{name}: shape={tuple(w.shape)}, "
                  f"range=[{w.min():.3f}, {w.max():.3f}], "
                  f"alpha={module.alpha.item():.4f}")
