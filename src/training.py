"""
Hardware-Aware Training with Knowledge Distillation.

Two-phase training:
    1. Teacher: Standard FP32 training (200 epochs)
    2. Student: HWA training with noise, initialized from teacher (80 epochs)

The knowledge distillation from teacher helps student learn despite
the noise injection - soft targets are more informative than hard labels.
"""

import os
import time
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple, Dict, Any, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from .physics import PCMPhysicsEngine, NoiseScheduler, compute_gdc_factor
from .layers import (AnalogLinear, AnalogConv2d, apply_caws, 
                     remap_all_weights, set_drop_connect_prob, set_inference_time)


logger = logging.getLogger(__name__)


@dataclass
class TrainingConfig:
    """Configuration for HWA training."""
    
    # Teacher training
    epochs_teacher: int = 200
    lr_teacher: float = 0.1
    
    # Student HWA training
    epochs_student: int = 80
    lr_student: float = 0.01
    
    # Knowledge distillation
    distill_temp: float = 4.0    # Temperature for soft targets
    distill_alpha: float = 0.9   # Weight: 0.9 distill + 0.1 hard labels
    
    # HWA techniques
    noise_scale: float = 3.0     # Final noise multiplier
    noise_ramp_epochs: int = 10  # Epochs to reach final noise
    drop_connect_prob: float = 0.01  # 1% weight dropout
    remap_interval: int = 0      # 0 = disabled (SOTA setting)
    
    # General
    batch_size: int = 128
    weight_decay: float = 5e-4
    momentum: float = 0.9
    seed: int = 42
    checkpoint_dir: str = "checkpoints"
    device: str = "cuda"


class HWATrainer:
    """
    Trainer for Hardware-Aware models with teacher-student distillation.
    """
    
    def __init__(self, config: TrainingConfig):
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() 
                                   else 'cpu')
        self.checkpoint_dir = Path(config.checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Noise scheduler for ramping
        self.noise_scheduler = NoiseScheduler(
            final_scale=config.noise_scale,
            ramp_epochs=config.noise_ramp_epochs
        )
    
    def train_teacher(self, model: nn.Module, 
                      train_loader: DataLoader,
                      val_loader: DataLoader) -> nn.Module:
        """
        Phase 1: Train teacher model (standard FP32).
        
        Uses SGD with momentum, cosine LR schedule, standard CE loss.
        """
        logger.info("=" * 60)
        logger.info("PHASE 1: Teacher Training (FP32)")
        logger.info("=" * 60)
        
        model = model.to(self.device)
        
        optimizer = optim.SGD(
            model.parameters(),
            lr=self.config.lr_teacher,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        # Cosine annealing LR
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs_teacher
        )
        
        criterion = nn.CrossEntropyLoss()
        best_acc = 0.0
        
        for epoch in range(1, self.config.epochs_teacher + 1):
            # Train
            model.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                _, pred = output.max(1)
                train_correct += pred.eq(target).sum().item()
                train_total += data.size(0)
            
            scheduler.step()
            
            # Validate
            val_loss, val_acc = self._validate(model, val_loader, criterion)
            train_acc = 100.0 * train_correct / train_total
            
            # Logging every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:3d}/{self.config.epochs_teacher} | "
                           f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                           f"LR: {scheduler.get_last_lr()[0]:.4f}")
            
            # Save best
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(model.state_dict(), 
                          self.checkpoint_dir / 'teacher_best.pth')
        
        logger.info(f"Teacher best accuracy: {best_acc:.2f}%")
        
        # Load best for student initialization
        model.load_state_dict(
            torch.load(self.checkpoint_dir / 'teacher_best.pth')
        )
        return model
    
    def train_student(self, teacher: nn.Module, 
                      student: nn.Module,
                      train_loader: DataLoader,
                      val_loader: DataLoader,
                      physics: PCMPhysicsEngine) -> nn.Module:
        """
        Phase 2: Train student with HWA + knowledge distillation.
        
        Key differences from teacher:
            - Warm start from teacher weights
            - PCM noise injection (ramped)
            - Drop-connect regularization
            - Soft targets from teacher (distillation)
        """
        logger.info("=" * 60)
        logger.info("PHASE 2: Student HWA Training")
        logger.info("=" * 60)
        
        teacher = teacher.to(self.device)
        student = student.to(self.device)
        teacher.eval()  # Teacher frozen
        
        # CRITICAL: Initialize student from teacher (warm start)
        # This is essential for convergence with noise injection
        logger.info("Warm starting student from teacher weights...")
        student.load_state_dict(teacher.state_dict(), strict=False)
        
        # Apply caws (crossbar-aware weight scaling)
        apply_caws(student)
        
        # Initial weight remapping to use full [-1, 1] range
        remap_all_weights(student)
        
        optimizer = optim.SGD(
            student.parameters(),
            lr=self.config.lr_student,
            momentum=self.config.momentum,
            weight_decay=self.config.weight_decay
        )
        
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.epochs_student
        )
        
        best_acc = 0.0
        T = self.config.distill_temp
        alpha = self.config.distill_alpha
        
        for epoch in range(1, self.config.epochs_student + 1):
            # Update noise scale (ramping)
            current_noise = self.noise_scheduler.get_scale(epoch)
            physics.set_noise_scale(current_noise)
            
            # Set drop-connect
            set_drop_connect_prob(student, self.config.drop_connect_prob)
            
            # Train epoch
            student.train()
            train_loss, train_correct, train_total = 0, 0, 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(self.device), target.to(self.device)
                
                optimizer.zero_grad()
                
                # Teacher forward (no grad)
                with torch.no_grad():
                    teacher_logits = teacher(data)
                
                # Student forward (with noise)
                student_logits = student(data)
                
                # Distillation loss: KL(student || teacher)
                distill_loss = F.kl_div(
                    F.log_softmax(student_logits / T, dim=1),
                    F.softmax(teacher_logits / T, dim=1),
                    reduction='batchmean'
                ) * (T * T)  # Scale by T² per Hinton et al.
                
                # Hard label loss
                hard_loss = F.cross_entropy(student_logits, target)
                
                # Combined loss
                loss = alpha * distill_loss + (1 - alpha) * hard_loss
                
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item() * data.size(0)
                _, pred = student_logits.max(1)
                train_correct += pred.eq(target).sum().item()
                train_total += data.size(0)
            
            scheduler.step()
            
            # Periodic remapping (DISABLED in SOTA - interval=0)
            if self.config.remap_interval > 0 and epoch % self.config.remap_interval == 0:
                remap_all_weights(student)
            
            # Validate (clean, no noise for fair comparison)
            physics.set_noise_scale(0.0)  # Temporarily disable
            set_drop_connect_prob(student, 0.0)
            val_loss, val_acc = self._validate(student, val_loader, 
                                               nn.CrossEntropyLoss())
            
            train_acc = 100.0 * train_correct / train_total
            
            if epoch % 5 == 0 or epoch == 1:
                logger.info(f"Epoch {epoch:3d}/{self.config.epochs_student} | "
                           f"Train: {train_acc:.2f}% | Val: {val_acc:.2f}% | "
                           f"Noise: {current_noise:.2f}x")
            
            if val_acc > best_acc:
                best_acc = val_acc
                torch.save(student.state_dict(),
                          self.checkpoint_dir / 'student_best.pth')
        
        logger.info(f"Student best accuracy: {best_acc:.2f}%")
        
        student.load_state_dict(
            torch.load(self.checkpoint_dir / 'student_best.pth')
        )
        return student
    
    def _validate(self, model: nn.Module, loader: DataLoader,
                  criterion: nn.Module) -> Tuple[float, float]:
        """Validation loop."""
        model.eval()
        val_loss, correct, total = 0, 0, 0
        
        with torch.no_grad():
            for data, target in loader:
                data, target = data.to(self.device), target.to(self.device)
                output = model(data)
                val_loss += criterion(output, target).item() * data.size(0)
                _, pred = output.max(1)
                correct += pred.eq(target).sum().item()
                total += data.size(0)
        
        return val_loss / total, 100.0 * correct / total


def evaluate_drift_with_gdc(model: nn.Module, loader: DataLoader,
                            physics: PCMPhysicsEngine, device: torch.device,
                            drift_times: Optional[list] = None,
                            n_samples: int = 3) -> Dict[str, Dict]:
    """
    Evaluate model accuracy under drift with GDC compensation.
    
    For each drift time:
        1. Program weights (apply noise once)
        2. Simulate drift
        3. Apply GDC output scaling
        4. Measure accuracy
    
    Args:
        model: trained HWA model
        loader: test data loader
        physics: PCM physics engine
        device: torch device
        drift_times: list of (seconds, label) tuples
        n_samples: number of noise samples for mean/std
        
    Returns:
        Dictionary with results per time point
    """
    if drift_times is None:
        drift_times = [
            (1, "1 sec"),
            (3600, "1 hour"),
            (86400, "1 day"),
            (365 * 24 * 3600, "1 year")
        ]
    
    model.eval()
    results = {}
    
    for t_sec, label in drift_times:
        accuracies = []
        
        for sample in range(n_samples):
            # Compute GDC factor for this time
            gdc = compute_gdc_factor(t_sec, t0=physics.T0, nu=physics.DRIFT_NU)
            
            # Set up hooks for GDC compensation
            # Hook applies: output = (output - bias) * gdc + bias
            handles = []
            
            def get_gdc_hook(gdc_val):
                def hook(module, inp, out):
                    if hasattr(module, 'bias') and module.bias is not None:
                        return (out - module.bias) * gdc_val + module.bias
                    return out * gdc_val
                return hook
            
            # Register hooks on analog layers
            for m in model.modules():
                if isinstance(m, (AnalogLinear, AnalogConv2d)):
                    m.set_inference_time(t_sec)
                    handle = m.register_forward_hook(get_gdc_hook(gdc))
                    handles.append(handle)
            
            # Re-sample programming noise
            physics.set_noise_scale(1.0)  # Enable noise
            
            # Evaluate
            correct, total = 0, 0
            with torch.no_grad():
                for data, target in loader:
                    data, target = data.to(device), target.to(device)
                    output = model(data)
                    _, pred = output.max(1)
                    correct += pred.eq(target).sum().item()
                    total += data.size(0)
            
            acc = 100.0 * correct / total
            accuracies.append(acc)
            
            # Remove hooks
            for h in handles:
                h.remove()
            
            # Reset inference time
            set_inference_time(model, 0.0)
        
        results[label] = {
            'time_sec': t_sec,
            'gdc_factor': compute_gdc_factor(t_sec),
            'accuracy_mean': sum(accuracies) / len(accuracies),
            'accuracy_std': (sum((a - sum(accuracies)/len(accuracies))**2 
                            for a in accuracies) / len(accuracies)) ** 0.5
        }
        
        logger.info(f"{label:>12} | GDC: {results[label]['gdc_factor']:.4f} | "
                   f"Acc: {results[label]['accuracy_mean']:.2f}% "
                   f"± {results[label]['accuracy_std']:.2f}%")
    
    return results


# Quick test
if __name__ == '__main__':
    config = TrainingConfig()
    print(f"[training.py] Config: {config}")
    print(f"[training.py] Checkpoint dir: {config.checkpoint_dir}")
