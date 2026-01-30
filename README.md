# Hardware-Aware Training for Analog In-Memory Computing

Replication of **"Hardware-aware training for large-scale and diverse deep learning inference on analog in-memory computing"** by Rasch et al., Nature Electronics 2023. [arXiv:2302.08469](https://arxiv.org/abs/2302.08469)

## Key Results

| Model | Task | Accuracy @ 1s | Accuracy @ 1 year | Drift Δ |
|-------|------|---------------|-------------------|---------|
| WideResNet-16-4 | CIFAR-100 | 76.95% | 76.94% | -0.01% |
| LSTM | WikiText-2 | 259.05 PPL | 259.09 PPL | +0.04 |
| BERT-base | SST-2 (GLUE) | ~92% | ~92% | <1% |

All models achieve **iso-accuracy** over 1 year of simulated drift—matching the paper's core claim that HWA training enables drift-robust deployment on analog hardware.

## What This Code Does

Analog in-memory computing (AIMC) accelerates neural networks by performing matrix-vector multiplications directly in memory using Phase-Change Memory (PCM) devices. However, PCM has two challenges:

1. **Programming noise**: Each weight has stochastic error when written
2. **Conductance drift**: Weights decay over time as g(t) = g₀ × (t/t₀)^(-ν)

This codebase implements **Hardware-Aware Training (HWA)**, which injects noise during training so the network learns to be robust. Combined with **Global Drift Compensation (GDC)**—a simple output scaling—trained models maintain accuracy for months/years after deployment.

### HWA Techniques Implemented

1. **Noise injection via STE**: Straight-Through Estimator for quantization-aware training
2. **Noise ramping**: Gradually increase noise from 0→3× over 10 epochs  
3. **Drop-connect**: 1% random weight zeroing (simulates stuck cells)
4. **Weight remapping**: Rescale to use full [-1,1] conductance range
5. **CAWS**: Crossbar-Aware Weight Scaling (α = √(3/fan_in))
6. **Knowledge distillation**: Teacher-student training with soft targets

## Installation

```bash
git clone https://github.com/YOUR_USERNAME/hwa-analog-training.git
cd hwa-analog-training
pip install -e .
```

Requirements: PyTorch ≥ 2.0, torchvision, datasets (for WikiText-2)

## Quick Start

### Train WideResNet on CIFAR-100

```bash
# Full training (200 epochs teacher + 80 epochs student)
python scripts/train_wideresnet.py

# Teacher only (for debugging)
python scripts/train_wideresnet.py --teacher-only --epochs-teacher 10

# Resume student training from teacher checkpoint
python scripts/train_wideresnet.py --student-only --resume checkpoints/teacher_best.pth
```

### Train LSTM on WikiText-2

```bash
python scripts/train_lstm.py
```

### Train BERT on GLUE (SST-2)

```bash
# Full training
python scripts/train_bert.py

# Different task
python scripts/train_bert.py --task mrpc --epochs 5

# Evaluation only with checkpoint
python scripts/train_bert.py --eval-only --checkpoint checkpoints/bert_hwa_best.pth
```

### Evaluate Drift

```python
from src import (
    PCMPhysicsEngine, wideresnet16_4, evaluate_drift_with_gdc,
    get_cifar100_loaders
)

# Load trained model
physics = PCMPhysicsEngine(noise_scale=1.0)
model = wideresnet16_4(physics=physics)
model.load_state_dict(torch.load('checkpoints/student_best.pth'))

# Evaluate at different drift times
_, _, test_loader = get_cifar100_loaders()
results = evaluate_drift_with_gdc(
    model, test_loader, physics, device='cuda',
    drift_times=[(1, '1s'), (3600, '1h'), (86400, '1d'), (31536000, '1yr')]
)
```

## Project Structure

```
hwa-analog-training/
├── src/
│   ├── physics.py      # PCM noise model (Eq. 2-4 in paper)
│   ├── layers.py       # AnalogLinear, AnalogConv2d with STE
│   ├── models/
│   │   ├── lstm.py     # LSTM for language modeling
│   │   ├── wideresnet.py   # WideResNet for vision
│   │   └── bert.py     # BERT for NLP (GLUE tasks)
│   ├── training.py     # HWA trainer with knowledge distillation
│   └── data.py         # CIFAR-100, WikiText-2 loaders
├── scripts/
│   ├── train_wideresnet.py
│   ├── train_lstm.py
│   └── train_bert.py
├── configs/            # YAML configs (optional)
└── experiments/        # Saved results and logs
```

## Physics Model Details

### Programming Noise (Eq. 3)
```
σ(w) = c₀ + c₁|w| + c₂w²
```
where `c = [0.26348, 1.9650, -1.1731] / g_max` are hardware-calibrated coefficients.

### Conductance Drift (Eq. 2)
```
w(t) = w₀ × (t/t₀)^(-ν)
```
with `t₀ = 20s` and `ν = 0.05` (measured on IBM PCM hardware).

### Global Drift Compensation (GDC)
```
output_compensated = output × (t/t₀)^(+ν)
```
Simple output scaling that recovers accuracy lost to drift.

## Key Implementation Notes

1. **Warm start is critical**: Student must initialize from teacher weights. Random init + noise = catastrophic failure.

2. **Periodic remapping disabled**: Despite being in the paper, we found `remap_interval=0` works best with distillation.

3. **Noise injection in forward only**: The backward pass uses clean gradients (STE principle).

4. **GDC implemented via hooks**: PyTorch forward hooks scale outputs during inference.

## Citation

If you use this code, please cite the original paper:

```bibtex
@article{rasch2023hardware,
  title={Hardware-aware training for large-scale and diverse deep learning inference on analog in-memory computing},
  author={Rasch, Malte J and others},
  journal={Nature Electronics},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## License

MIT License - see LICENSE file.
