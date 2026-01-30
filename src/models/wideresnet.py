import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers import AnalogConv2d, AnalogLinear
class AnalogWideResNet(nn.Module):
    def __init__(self, depth=16, widen_factor=4, num_classes=100, physics_engine=None):
        super().__init__()
        self.in_planes = 16
        self.physics = physics_engine
        n = (depth - 4) // 6
        k = widen_factor
        self.conv1 = AnalogConv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False, physics_engine=self.physics)
        self.layer1 = self._wide_layer(16*k, n, stride=1)
        self.layer2 = self._wide_layer(32*k, n, stride=2)
        self.layer3 = self._wide_layer(64*k, n, stride=2)
        self.bn1 = nn.BatchNorm2d(64*k, momentum=0.9)
        self.linear = AnalogLinear(64*k, num_classes, physics_engine=self.physics)
    def _wide_layer(self, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(self._AnalogBlock(self.in_planes, planes, stride, self.physics))
            self.in_planes = planes
        return nn.Sequential(*layers)
    class _AnalogBlock(nn.Module):
        def __init__(self, in_planes, planes, stride, physics):
            super().__init__()
            self.bn1 = nn.BatchNorm2d(in_planes); self.conv1 = AnalogConv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False, physics_engine=physics)
            self.bn2 = nn.BatchNorm2d(planes); self.conv2 = AnalogConv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False, physics_engine=physics)
            self.shortcut = nn.Sequential()
            if stride != 1 or in_planes != planes:
                self.shortcut = nn.Sequential(AnalogConv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False, physics_engine=physics))
        def forward(self, x):
            out = F.relu(self.bn1(x)); out = self.conv1(out)
            out = F.relu(self.bn2(out)); out = self.conv2(out)
            out += self.shortcut(x)
            return out
    def forward(self, x):
        out = self.conv1(x); out = self.layer1(out); out = self.layer2(out); out = self.layer3(out); out = F.relu(self.bn1(out)); out = F.avg_pool2d(out, 8); out = out.view(out.size(0), -1); out = self.linear(out)
        return out
