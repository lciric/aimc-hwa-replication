# PCM Analog In-Memory Computing Simulation

Simulation of Phase-Change Memory (PCM) based neural network inference, replicating results from:
- **Joshi et al. 2020**: "Accurate deep neural network inference using computational phase-change memory"
- **Le Gallo et al. 2022**: "Precision of bit slicing with in-memory computing based on analog phase-change memory crossbars"

## Project Structure

```
pcm_simulation/
├── physics.py          # PCM device physics (noise, drift, read noise)
├── quantization.py     # Bit slicing algorithms (equal/varying significance)
├── layers.py           # Analog Conv2d/Linear layers
├── replicate_paper.py  # Benchmark script
├── test_components.py  # Unit tests
└── README.md           # This file
```

## Key Concepts

### 1. Bit Slicing Modes

The code implements two bit slicing modes as described in Le Gallo 2022:

| Mode | Base (b_W) | Slice Range | Reconstruction | Best For |
|------|------------|-------------|----------------|----------|
| **Equal** | 1 | r_W / n | Σ S_i | Drift robustness |
| **Varying** | 2 | r_W*(b-1)/(b^n-1) | Σ S_i * b^i | Digital compatibility |

**Key finding**: Equal significance (b=1) is optimal for PCM inference because:
- Error averages out: η = η_s / √n (Equation 8)
- Drift acts as multiplicative noise → averaging helps

### 2. PCM Physics Model

The simulation includes three noise sources:
1. **Programming noise**: σ(G) = c₀G² + c₁G + c₂ (polynomial fit from IBM measurements)
2. **Conductance drift**: G(t) = G(t₀) × (t/t₀)^(-ν) where ν ~ N(μ_ν, σ_ν)
3. **Read noise**: 1/f noise proportional to conductance

### 3. Global Drift Compensation (GDC)

Compensates for drift by scaling outputs: α(t) = I_cal(t₀) / I_cal(t)

## Installation

```bash
pip install torch torchvision
```

## Usage

### Quick Test
```python
from layers import AnalogConv2d, AnalogLinear, convert_to_analog

# Create an analog layer
layer = AnalogLinear(784, 10, num_slices=8, slicing_mode="equal")

# Or convert an existing model
import torchvision.models as models
model = models.resnet18(pretrained=True)
analog_model = convert_to_analog(model, num_slices=8, slicing_mode="equal")
```

### Run Benchmarks
```bash
python replicate_paper.py
```

### Run Tests
```bash
python test_components.py
```

## Expected Results (from Le Gallo 2022, Figure 5)

CIFAR-10 with ResNet-32:

| Config | t=0 | t=1 month |
|--------|-----|-----------|
| Baseline (digital) | ~93% | - |
| 1 slice | ~91% | ~90% |
| 8 slices (b=1) | ~93% | ~92% |
| 8 slices (b=2) | ~92% | ~90% |

## Differences from Original ANU Code

1. **Clearer separation of concerns**: physics.py, quantization.py, layers.py
2. **Explicit slicing modes**: SlicingMode.EQUAL vs SlicingMode.VARYING (not implicit)
3. **Physical units**: Conductances in µS throughout physics model
4. **Documented reconstruction**: Equal mode uses sum, Varying uses weighted sum

## API Reference

### `AnalogConv2d` / `AnalogLinear`

```python
AnalogConv2d(
    in_channels, out_channels, kernel_size,
    stride=1, padding=0, bias=True,
    # Analog parameters:
    num_slices=1,           # Number of weight slices
    slicing_mode="equal",   # "equal" (b=1) or "varying" (b=2)
    slicing_algo="equal_fill",  # "equal_fill" or "max_fill"
    use_gdc=True            # Global Drift Compensation
)
```

### `convert_to_analog()`

```python
convert_to_analog(
    model,                  # PyTorch model
    num_slices=1,
    slicing_mode="equal",
    slicing_algo="equal_fill",
    use_gdc=True
) -> nn.Module
```

### `set_drift_time()`

```python
set_drift_time(model, t_drift)  # Set drift time in seconds
```

## Mathematical Details

### Equal Significance (b=1)

For n slices:
- Slice range: r_s = r_W / n
- Each slice: S_j = W / n
- Reconstruction: W_recon = Σ S_j
- Error: η_total = η_slice / √n (noise averages)

### Varying Significance (b=2)

For n slices with base b:
- Slice range: r_s = r_W × (b-1) / (b^n - 1)
- Reconstruction: W_recon = Σ S_i × b^i
- Error: Dominated by MSB slice (no averaging benefit)

## References

```bibtex
@article{joshi2020accurate,
  title={Accurate deep neural network inference using computational phase-change memory},
  author={Joshi, Vinay and others},
  journal={Nature Communications},
  year={2020}
}

@article{legallo2022precision,
  title={Precision of bit slicing with in-memory computing based on analog phase-change memory crossbars},
  author={Le Gallo, Manuel and others},
  journal={Neuromorphic Computing and Engineering},
  year={2022}
}
```

## Author

[Your Name]

## License

MIT
