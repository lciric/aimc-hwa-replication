"""
Unit tests for PCM physics engine.

Run with: pytest tests/test_physics.py -v
"""

import pytest
import torch
import math

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.physics import PCMPhysicsEngine, NoiseScheduler, compute_gdc_factor


class TestPCMPhysicsEngine:
    """Tests for PCM physics simulation."""
    
    @pytest.fixture
    def engine(self):
        return PCMPhysicsEngine(device='cpu', noise_scale=1.0)
    
    def test_init(self, engine):
        """Test engine initialization with correct parameters."""
        assert engine.G_MAX == 25.0
        assert engine.T0 == 20.0
        assert engine.DRIFT_NU == 0.05
        assert engine.prog_coeffs.shape == (3,)
    
    def test_programming_noise_shape(self, engine):
        """Test that noise preserves tensor shape."""
        w = torch.randn(10, 20)
        w_noisy = engine.apply_programming_noise(w)
        assert w_noisy.shape == w.shape
    
    def test_programming_noise_distribution(self, engine):
        """Test that noise has reasonable magnitude."""
        # Small weights should have small noise
        w = torch.zeros(1000) + 0.1
        w_noisy = engine.apply_programming_noise(w)
        noise = (w_noisy - w).std().item()
        assert 0.001 < noise < 0.1, f"Noise std {noise} out of expected range"
    
    def test_zero_noise_scale(self, engine):
        """Test that noise_scale=0 produces no noise."""
        engine.set_noise_scale(0.0)
        w = torch.randn(100)
        w_noisy = engine.apply_programming_noise(w)
        assert torch.allclose(w, w_noisy)
    
    def test_drift_identity_at_t0(self, engine):
        """Test no drift at reference time."""
        w = torch.randn(100)
        w_drifted = engine.apply_drift(w, t_inference=20.0)  # t0 = 20s
        assert torch.allclose(w, w_drifted)
    
    def test_drift_magnitude_at_1year(self, engine):
        """Test drift magnitude at 1 year matches expected."""
        w = torch.ones(100)
        t_year = 365 * 24 * 3600  # 1 year in seconds
        w_drifted = engine.apply_drift(w, t_inference=t_year)
        
        # Expected: (t_year / 20)^(-0.05) ≈ 0.76
        expected_ratio = (t_year / 20.0) ** (-0.05)
        actual_ratio = w_drifted.mean().item()
        
        assert abs(actual_ratio - expected_ratio) < 0.01, \
            f"Drift ratio {actual_ratio} != expected {expected_ratio}"
    
    def test_drift_monotonic(self, engine):
        """Test that drift decreases weights over time."""
        w = torch.ones(100)
        times = [1, 100, 10000, 1e6]
        prev_mean = float('inf')
        
        for t in times:
            w_drifted = engine.apply_drift(w.clone(), t)
            assert w_drifted.mean().item() < prev_mean or t <= 20
            prev_mean = w_drifted.mean().item()


class TestNoiseScheduler:
    """Tests for noise ramping schedule."""
    
    def test_zero_at_epoch_zero(self):
        scheduler = NoiseScheduler(final_scale=3.0, ramp_epochs=10)
        assert scheduler.get_scale(0) == 0.0
    
    def test_linear_ramp(self):
        scheduler = NoiseScheduler(final_scale=3.0, ramp_epochs=10)
        
        # Halfway through ramp
        assert abs(scheduler.get_scale(5) - 1.5) < 0.01
    
    def test_final_scale_reached(self):
        scheduler = NoiseScheduler(final_scale=3.0, ramp_epochs=10)
        
        assert scheduler.get_scale(10) == 3.0
        assert scheduler.get_scale(100) == 3.0  # Stays at final


class TestGDCFactor:
    """Tests for Global Drift Compensation factor."""
    
    def test_gdc_at_t0(self):
        """GDC should be 1.0 at reference time."""
        assert compute_gdc_factor(20.0) == 1.0
    
    def test_gdc_compensates_drift(self):
        """GDC × drifted_weight should recover original."""
        t = 1e6  # Some drift time
        t0, nu = 20.0, 0.05
        
        drift_factor = (t / t0) ** (-nu)
        gdc_factor = compute_gdc_factor(t, t0, nu)
        
        # drift × gdc should be close to 1
        assert abs(drift_factor * gdc_factor - 1.0) < 0.01
    
    def test_gdc_at_1year(self):
        """Test GDC factor at 1 year."""
        t_year = 365 * 24 * 3600
        gdc = compute_gdc_factor(t_year)
        
        # Expected: (t_year / 20)^(0.05) ≈ 2.04
        expected = (t_year / 20.0) ** 0.05
        assert abs(gdc - expected) < 0.01


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
