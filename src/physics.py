"""
PCM Physics Engine for Analog In-Memory Computing.

Implements the Phase-Change Memory (PCM) noise model from:
    Rasch et al. "Hardware-aware training for large-scale and diverse 
    deep learning inference on analog in-memory computing"
    Nature Electronics, 2023. arXiv:2302.08469

The physics here are calibrated to real IBM hardware measurements.
Messing with these numbers will break drift compensation.
"""

import torch
import torch.nn as nn
from typing import Optional


class PCMPhysicsEngine(nn.Module):
    """
    Phase-Change Memory physics with programming noise and conductance drift.
    
    Hardware parameters from IBM's measurements on real PCM devices.
    Don't touch g_max or the polynomial coefficients unless you have
    new hardware characterization data.
    
    Key equations:
        Programming noise std: σ(w) = c₀ + c₁|w| + c₂w²
        Drift: w(t) = w₀ × (t/t₀)^(-ν)
    """
    
    # IBM hardware characterization (Table S4 in supplementary)
    G_MAX = 25.0  # µS, max conductance
    T0 = 20.0     # seconds, reference time for drift
    DRIFT_NU = 0.05  # drift exponent, empirically measured
    
    # Programming noise polynomial coefficients (raw, before g_max normalization)
    # These come from fitting σ(g) on real hardware
    RAW_PROG_COEFFS = torch.tensor([0.26348, 1.9650, -1.1731])
    
    def __init__(self, device: str = 'cuda', noise_scale: float = 1.0):
        """
        Args:
            device: torch device
            noise_scale: multiplier for noise injection during training
                         Use 0.0 for clean inference, >1.0 for HWA training
        """
        super().__init__()
        self.device = device
        self.noise_scale = noise_scale
        
        # Normalize coefficients by g_max for weight-space operations
        # w = g/g_max, so σ(w) needs rescaling
        self.register_buffer(
            'prog_coeffs', 
            self.RAW_PROG_COEFFS.clone().to(device) / self.G_MAX
        )
    
    def set_noise_scale(self, scale: float) -> None:
        """Update noise scale (for noise ramping during training)."""
        self.noise_scale = scale
    
    def programming_noise_std(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Compute per-weight programming noise standard deviation.
        
        σ(w) = c₀ + c₁|w| + c₂w²
        
        The quadratic term models the fact that high-conductance states
        have different noise characteristics than low-conductance states.
        """
        w_abs = torch.abs(weight)
        std = (self.prog_coeffs[0] + 
               self.prog_coeffs[1] * w_abs + 
               self.prog_coeffs[2] * weight.pow(2))
        # Clamp to avoid numerical issues with very small std
        return torch.clamp(std, min=1e-6)
    
    def apply_programming_noise(self, weight: torch.Tensor) -> torch.Tensor:
        """
        Add weight-dependent programming noise.
        
        This simulates the write variability when programming PCM cells.
        Each cell's programmed conductance deviates from target.
        """
        if self.noise_scale == 0.0:
            return weight
        
        std = self.programming_noise_std(weight)
        noise = torch.randn_like(weight) * std * self.noise_scale
        return weight + noise
    
    def apply_drift(self, weight: torch.Tensor, t_inference: float) -> torch.Tensor:
        """
        Apply conductance drift for a given inference time.
        
        PCM cells drift: g(t) = g₀ × (t/t₀)^(-ν)
        
        This is why analog compute degrades over time without compensation.
        At t=1 year, conductances drop by ~50% (2^(-0.05*log2(year/t0))).
        
        Args:
            weight: quantized/noisy weights
            t_inference: time since programming in seconds
        """
        if t_inference <= self.T0:
            return weight
        
        drift_factor = (t_inference / self.T0) ** (-self.DRIFT_NU)
        return weight * drift_factor
    
    def forward(self, weight: torch.Tensor, t_inference: float = 0.0, 
                training: bool = True) -> torch.Tensor:
        """
        Full physics pipeline: noise → drift (inference only).
        
        During training: apply noise only (drift happens at inference)
        During inference: apply noise once, then drift
        """
        w_noisy = self.apply_programming_noise(weight)
        
        if not training and t_inference > 0:
            w_noisy = self.apply_drift(w_noisy, t_inference)
        
        return w_noisy


class NoiseScheduler:
    """
    Linear noise ramping for HWA training.
    
    The paper shows that gradually increasing noise during training
    helps the network adapt. Starting with full noise from epoch 1
    tends to destabilize training.
    
    Typical schedule: 0 → 3.0× over 10 epochs
    """
    
    def __init__(self, final_scale: float = 3.0, ramp_epochs: int = 10):
        """
        Args:
            final_scale: target noise multiplier after ramping
            ramp_epochs: epochs to reach final scale
        """
        self.final_scale = final_scale
        self.ramp_epochs = ramp_epochs
    
    def get_scale(self, epoch: int) -> float:
        """Get noise scale for current epoch (1-indexed)."""
        if epoch >= self.ramp_epochs:
            return self.final_scale
        # Linear ramp from 0 to final_scale
        return self.final_scale * (epoch / self.ramp_epochs)


def compute_gdc_factor(t_inference: float, t0: float = 20.0, 
                       nu: float = 0.05) -> float:
    """
    Compute Global Drift Compensation (GDC) scaling factor.
    
    GDC compensates for conductance drift by scaling outputs.
    If all weights drift by factor (t/t₀)^(-ν), we can recover
    by multiplying outputs by (t/t₀)^(+ν).
    
    This is the "oracle" version - in real hardware you'd estimate
    drift from reference cells.
    
    Args:
        t_inference: time since programming in seconds
        t0: reference time (20s for IBM PCM)
        nu: drift exponent (0.05 typical)
        
    Returns:
        Scaling factor to apply to layer outputs
    """
    if t_inference <= t0:
        return 1.0
    return (t_inference / t0) ** nu


# Quick sanity check when module is imported
if __name__ == '__main__':
    engine = PCMPhysicsEngine(device='cpu', noise_scale=1.0)
    w = torch.randn(10, 10) * 0.1
    w_noisy = engine.apply_programming_noise(w)
    print(f"[physics.py] Noise std check: {(w_noisy - w).std():.4f}")
    
    # Check drift at 1 year
    w_drifted = engine.apply_drift(w, t_inference=365*24*3600)
    drift_ratio = w_drifted.mean() / w.mean()
    print(f"[physics.py] 1-year drift ratio: {drift_ratio:.3f} (expected ~0.76)")
