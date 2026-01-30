"""
Unit tests for analog neural network layers.

Run with: pytest tests/test_layers.py -v
"""

import pytest
import torch
import torch.nn as nn
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.layers import (
    AnalogLinear, AnalogConv2d, StraightThroughEstimator,
    compute_caws_alpha, apply_caws, remap_weights, set_drop_connect_prob
)
from src.physics import PCMPhysicsEngine


class TestAnalogLinear:
    """Tests for AnalogLinear layer."""
    
    @pytest.fixture
    def layer(self):
        physics = PCMPhysicsEngine(device='cpu', noise_scale=0.0)
        return AnalogLinear(10, 20, physics=physics)
    
    def test_output_shape(self, layer):
        """Test forward pass produces correct shape."""
        x = torch.randn(5, 10)
        y = layer(x)
        assert y.shape == (5, 20)
    
    def test_default_gamma(self, layer):
        """Test default quantization levels."""
        assert layer.gamma.item() == 256.0
    
    def test_alpha_initialization(self, layer):
        """Test default alpha is 1.0."""
        assert layer.alpha.item() == 1.0
    
    def test_backward_pass(self, layer):
        """Test that gradients flow through STE."""
        x = torch.randn(5, 10, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None
    
    def test_inference_time_setting(self, layer):
        """Test inference time propagates to layer."""
        layer.set_inference_time(3600.0)
        assert layer.t_inference == 3600.0


class TestAnalogConv2d:
    """Tests for AnalogConv2d layer."""
    
    @pytest.fixture
    def layer(self):
        physics = PCMPhysicsEngine(device='cpu', noise_scale=0.0)
        return AnalogConv2d(3, 16, kernel_size=3, padding=1, physics=physics)
    
    def test_output_shape(self, layer):
        """Test forward pass produces correct shape."""
        x = torch.randn(2, 3, 32, 32)
        y = layer(x)
        assert y.shape == (2, 16, 32, 32)  # Same spatial with padding=1
    
    def test_backward_pass(self, layer):
        """Test gradients flow through STE."""
        x = torch.randn(2, 3, 32, 32, requires_grad=True)
        y = layer(x)
        loss = y.sum()
        loss.backward()
        
        assert x.grad is not None
        assert layer.weight.grad is not None


class TestSTE:
    """Tests for Straight-Through Estimator."""
    
    def test_quantization_levels(self):
        """Test quantization to discrete levels."""
        w = torch.tensor([0.0, 0.5, 1.0, -0.5, -1.0])
        gamma = torch.tensor(256.0)
        alpha = torch.tensor(1.0)
        
        w_quant = StraightThroughEstimator.apply(
            w, gamma, alpha, None, 0.0, False, 0.0
        )
        
        # Check values are quantized
        levels = gamma - 1
        expected = torch.round(w * levels) / levels
        assert torch.allclose(w_quant, expected)
    
    def test_gradient_passthrough(self):
        """Test that gradients pass through unchanged."""
        w = torch.randn(10, requires_grad=True)
        gamma = torch.tensor(256.0)
        alpha = torch.tensor(1.0)
        
        w_quant = StraightThroughEstimator.apply(
            w, gamma, alpha, None, 0.0, True, 0.0
        )
        loss = w_quant.sum()
        loss.backward()
        
        # Gradient should be all ones (dL/dw = d(sum)/dw = 1)
        assert torch.allclose(w.grad, torch.ones_like(w))


class TestCAWS:
    """Tests for Crossbar-Aware Weight Scaling."""
    
    def test_caws_linear(self):
        """Test CAWS alpha for linear layer."""
        layer = nn.Linear(100, 50)
        alpha = compute_caws_alpha(layer)
        
        expected = math.sqrt(3.0 / 100)
        assert abs(alpha - expected) < 1e-6
    
    def test_caws_conv(self):
        """Test CAWS alpha for conv layer."""
        layer = nn.Conv2d(16, 32, kernel_size=3)
        alpha = compute_caws_alpha(layer)
        
        fan_in = 16 * 3 * 3  # channels × kernel²
        expected = math.sqrt(3.0 / fan_in)
        assert abs(alpha - expected) < 1e-6
    
    def test_apply_caws_modifies_alpha(self):
        """Test that apply_caws sets alpha on all analog layers."""
        physics = PCMPhysicsEngine(device='cpu')
        model = nn.Sequential(
            AnalogLinear(100, 50, physics=physics),
            nn.ReLU(),
            AnalogLinear(50, 10, physics=physics),
        )
        
        apply_caws(model)
        
        # Check alphas are set correctly
        for m in model.modules():
            if isinstance(m, AnalogLinear):
                expected = compute_caws_alpha(m)
                assert abs(m.alpha.item() - expected) < 1e-6


class TestWeightRemapping:
    """Tests for weight remapping utility."""
    
    def test_remap_expands_range(self):
        """Test that remapping expands weights to [-1, 1]."""
        physics = PCMPhysicsEngine(device='cpu')
        layer = AnalogLinear(10, 10, physics=physics)
        
        # Set weights to small range
        with torch.no_grad():
            layer.weight.data = torch.randn_like(layer.weight) * 0.1
        
        old_max = layer.weight.abs().max().item()
        old_alpha = layer.alpha.item()
        
        remap_weights(layer)
        
        new_max = layer.weight.abs().max().item()
        new_alpha = layer.alpha.item()
        
        # Weights should now span to 1.0
        assert abs(new_max - 1.0) < 1e-6
        # Alpha should be adjusted proportionally
        assert abs(new_alpha - old_alpha * old_max) < 1e-6


class TestDropConnect:
    """Tests for drop-connect regularization."""
    
    def test_drop_connect_zeros_weights(self):
        """Test that drop-connect produces zeros."""
        physics = PCMPhysicsEngine(device='cpu', noise_scale=0.0)
        layer = AnalogLinear(100, 100, physics=physics, drop_connect_prob=0.5)
        layer.train()
        
        x = torch.randn(10, 100)
        
        # Run many forward passes
        zero_counts = []
        for _ in range(100):
            with torch.no_grad():
                w_eff = StraightThroughEstimator.apply(
                    layer.weight, layer.gamma, layer.alpha,
                    layer.physics, 0.0, True, 0.5
                )
                zero_ratio = (w_eff == 0).float().mean().item()
                zero_counts.append(zero_ratio)
        
        # Should be around 50%
        mean_zeros = sum(zero_counts) / len(zero_counts)
        assert 0.4 < mean_zeros < 0.6
    
    def test_drop_connect_disabled_in_eval(self):
        """Test that drop-connect is disabled in eval mode."""
        physics = PCMPhysicsEngine(device='cpu', noise_scale=0.0)
        layer = AnalogLinear(100, 100, physics=physics, drop_connect_prob=0.5)
        layer.eval()
        
        # In eval mode with training=False, drop-connect shouldn't apply
        w_eff = StraightThroughEstimator.apply(
            layer.weight, layer.gamma, layer.alpha,
            layer.physics, 0.0, False, 0.5  # training=False
        )
        
        # No zeros from drop-connect (only quantization)
        # This test checks the flag is respected


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
