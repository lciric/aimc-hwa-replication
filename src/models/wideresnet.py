"""
Wide Residual Networks with Analog Layers.

WideResNet-16-4 for CIFAR-100: 16 layers deep, width multiplier 4.
This is the standard benchmark for HWA training on vision tasks.

Reference:
    Zagoruyko & Komodakis, "Wide Residual Networks", BMVC 2016
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Type

from ..layers import AnalogConv2d
from ..physics import PCMPhysicsEngine


class BasicBlock(nn.Module):
    """
    WideResNet basic block with two conv layers.
    
    Structure:
        BN → ReLU → Conv3x3 → BN → ReLU → Conv3x3 + shortcut
    
    Unlike standard ResNet, we use pre-activation (BN-ReLU-Conv).
    """
    
    def __init__(self, in_planes: int, out_planes: int, stride: int = 1,
                 physics: Optional[PCMPhysicsEngine] = None,
                 drop_connect_prob: float = 0.0):
        super().__init__()
        
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = AnalogConv2d(in_planes, out_planes, kernel_size=3,
                                   stride=stride, padding=1, bias=False,
                                   physics=physics, 
                                   drop_connect_prob=drop_connect_prob)
        
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = AnalogConv2d(out_planes, out_planes, kernel_size=3,
                                   stride=1, padding=1, bias=False,
                                   physics=physics,
                                   drop_connect_prob=drop_connect_prob)
        
        # Shortcut for dimension mismatch
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Sequential(
                AnalogConv2d(in_planes, out_planes, kernel_size=1,
                             stride=stride, bias=False, physics=physics,
                             drop_connect_prob=drop_connect_prob)
            )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(x))
        
        # First conv
        out1 = self.conv1(out)
        
        # Second conv
        out2 = F.relu(self.bn2(out1))
        out2 = self.conv2(out2)
        
        # Residual connection
        out2 += self.shortcut(out if isinstance(self.shortcut, nn.Sequential) 
                              and len(self.shortcut) > 0 else x)
        
        return out2


class WideResNet(nn.Module):
    """
    Wide Residual Network for CIFAR.
    
    Architecture for WideResNet-d-k:
        - d: depth (total layers = 6n+4 where d=6n+4)
        - k: width multiplier
        
    WideResNet-16-4: n=2, widths=[64, 128, 256] × 4 = [256, 512, 1024]
    """
    
    def __init__(self, depth: int, widen_factor: int, num_classes: int = 100,
                 physics: Optional[PCMPhysicsEngine] = None,
                 drop_connect_prob: float = 0.0):
        super().__init__()
        
        self.physics = physics
        
        # Compute number of blocks per group
        assert (depth - 4) % 6 == 0, f"Depth must be 6n+4, got {depth}"
        n = (depth - 4) // 6
        
        # Channel widths
        nChannels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]
        
        # Initial conv (analog)
        self.conv1 = AnalogConv2d(3, nChannels[0], kernel_size=3,
                                   stride=1, padding=1, bias=False,
                                   physics=physics,
                                   drop_connect_prob=drop_connect_prob)
        
        # Three groups of residual blocks
        self.layer1 = self._make_layer(n, nChannels[0], nChannels[1], stride=1,
                                        physics=physics, 
                                        drop_connect_prob=drop_connect_prob)
        self.layer2 = self._make_layer(n, nChannels[1], nChannels[2], stride=2,
                                        physics=physics,
                                        drop_connect_prob=drop_connect_prob)
        self.layer3 = self._make_layer(n, nChannels[2], nChannels[3], stride=2,
                                        physics=physics,
                                        drop_connect_prob=drop_connect_prob)
        
        # Final BN + classifier
        self.bn = nn.BatchNorm2d(nChannels[3])
        self.fc = nn.Linear(nChannels[3], num_classes)
        
        # He initialization
        self._init_weights()
    
    def _make_layer(self, num_blocks: int, in_planes: int, out_planes: int,
                    stride: int, physics: Optional[PCMPhysicsEngine],
                    drop_connect_prob: float) -> nn.Sequential:
        """Create a group of residual blocks."""
        layers = []
        
        # First block may have stride and changes channels
        layers.append(BasicBlock(in_planes, out_planes, stride,
                                 physics=physics, 
                                 drop_connect_prob=drop_connect_prob))
        
        # Remaining blocks
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_planes, out_planes, stride=1,
                                     physics=physics,
                                     drop_connect_prob=drop_connect_prob))
        
        return nn.Sequential(*layers)
    
    def _init_weights(self) -> None:
        """He initialization for conv layers, standard for linear."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, AnalogConv2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', 
                                         nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Initial conv
        out = self.conv1(x)
        
        # Residual blocks
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        
        # Final pooling + classifier
        out = F.relu(self.bn(out))
        out = F.avg_pool2d(out, 8)  # CIFAR: 32→16→8→1 after pools
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        
        return out
    
    def set_inference_time(self, t: float) -> None:
        """Set drift time for all analog conv layers."""
        for m in self.modules():
            if isinstance(m, AnalogConv2d):
                m.set_inference_time(t)


def wideresnet16_4(num_classes: int = 100,
                   physics: Optional[PCMPhysicsEngine] = None,
                   drop_connect_prob: float = 0.0) -> WideResNet:
    """
    WideResNet-16-4 for CIFAR-100.
    
    This is the configuration used in the HWA paper for vision.
    16 layers, width factor 4 → ~11M parameters.
    """
    return WideResNet(depth=16, widen_factor=4, num_classes=num_classes,
                      physics=physics, drop_connect_prob=drop_connect_prob)


def wideresnet28_10(num_classes: int = 100,
                    physics: Optional[PCMPhysicsEngine] = None,
                    drop_connect_prob: float = 0.0) -> WideResNet:
    """
    WideResNet-28-10 for CIFAR-100 (larger variant).
    
    28 layers, width factor 10 → ~36M parameters.
    Higher accuracy but more compute.
    """
    return WideResNet(depth=28, widen_factor=10, num_classes=num_classes,
                      physics=physics, drop_connect_prob=drop_connect_prob)


# Quick parameter count check
if __name__ == '__main__':
    model = wideresnet16_4(num_classes=100)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[wideresnet.py] WRN-16-4 params: {n_params/1e6:.2f}M")
    
    # Test forward pass
    x = torch.randn(2, 3, 32, 32)
    y = model(x)
    print(f"[wideresnet.py] Output shape: {y.shape}")  # [2, 100]
