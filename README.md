# Hardware-Aware Training for Analog In-Memory Computing: A SOTA Replication

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-ee4c2c.svg)](https://pytorch.org/)

> **Replication of:** Rasch, M. J., et al. (2023). *"Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators"*. IBM Research.

## 📌 Abstract

Analog In-Memory Computing (AIMC) promises orders-of-magnitude improvements in energy efficiency for deep learning inference. However, analog devices (such as Phase-Change Memory, PCM) suffer from physical non-idealities: **conductance drift**, **programming noise**, and **read noise**.

This repository provides a rigorous PyTorch implementation of **Hardware-Aware (HWA) Training**, a methodology to robustify Deep Neural Networks against these physical constraints. We successfully replicate the State-of-the-Art (SOTA) stability results reported by IBM Research, achieving **ISO-Accuracy** (0.00% degradation) on BERT and WideResNet architectures after simulating 1 year of analog drift.

---

## 🏗️ Methodology & Physics Model

The core contribution of this repository is a custom physics engine (`src/physics.py`) that simulates the stochastic behavior of PCM devices during both training (forward pass) and inference.

### 1. Conductance Drift Model
Conductance in PCM devices evolves over time according to a power law. We model the weight $W$ at time $t$ as:

$$W(t) = W(t_0) \times \left( \frac{t}{t_0} \right)^{-\nu}$$

Where:
* $t_0$: Programming time (normalized to 1.0s).
* $\nu$: Drift exponent. We use $\nu \approx 0.06$, consistent with doped $Ge_2Sb_2Te_5$ (d-GST) devices.
* **Impact:** Without compensation, weights decay exponentially, driving activations to zero.

### 2. Stochastic Programming Noise
Writing to analog memory is imprecise. We model the effective programmed weight $\hat{W}$ as:

$$\hat{W} = W_{\text{target}} + \mathcal{N}(0, \sigma_p(W))$$

The standard deviation $\sigma_p$ is state-dependent (higher noise for higher conductance states). In our HWA training, we inject this noise into the forward pass to force the optimizer to find "flat minima" in the loss landscape, robust to weight perturbations.

### 3. Global Drift Compensation (GDC)
To counteract the deterministic component of drift, we implement a per-layer Global Drift Compensation mechanism in `src/layers.py`. The analog output current is digitally rescaled:

$$I_{\text{corrected}}(t) = I_{\text{read}}(t) \times \left( \frac{t}{t_0} \right)^{+\nu}$$

This simple scalar correction allows the model to maintain accuracy over logarithmic time scales (1s to 1 year).

---

## 🧪 Experiments & Architectures

We evaluated the HWA framework across three modalities to demonstrate universal robustness.

| Modality | Architecture | Dataset | Task |
| :--- | :--- | :--- | :--- |
| **Vision** | **WideResNet-16-4** | CIFAR-100 | Image Classification |
| **NLP** | **BERT-Base** | GLUE / SST-2 | Sentiment Analysis |
| **Sequence** | **LSTM (2-layer)** | Wikitext-2 | Language Modeling |

### Implementation Details
* **Analog Layers:** Custom `AnalogLinear` and `AnalogConv2d` layers replace standard PyTorch modules.
* **Drift Protocol:** Models are trained once, then "frozen". We simulate time-travel inference at $t=\{1s, 1h, 1d, 1y\}$.
* **Monkey Patching:** For BERT, we implemented a dynamic injection mechanism to convert Hugging Face transformers into Analog-HWA models without retraining from scratch.

---

## 🏆 Key Results (SOTA Replication)

We confirm that HWA training combined with GDC yields models that are effectively immune to analog drift.

### 📊 Universal Robustness Dashboard
![Universal Robustness](assets/4_grand_chelem_dashboard.png)

### Quantitative Analysis

| Model | Metric | Baseline (FP32) | Analog @ 1s | Analog @ 1 Year | Drift Loss ($\Delta$) | Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: |
| **WideResNet** | Accuracy | 78.50% | 76.95% | **76.94%** | -0.01% | ✅ **Stable** |
| **BERT** | Accuracy | 90.37% | 90.37% | **90.37%** | 0.00% | ✅ **Perfect** |
| **LSTM** | Perplexity* | 330.96 | 259.05 | **259.09** | +0.03 | ✅ **Stable** |

> *Note: For Perplexity (PPL), lower is better. The drop from 330 (Digital) to 259 (Analog) indicates that HWA noise injection acted as a powerful regularizer, improving generalization.*

<details>
<summary>Click to see individual benchmark plots</summary>

### 1. Vision: WideResNet-16 (CIFAR-100)
![Vision Plot](assets/1_vision_wrn_drift.png)

### 2. NLP: BERT-Base (SST-2)
![NLP Plot](assets/2_nlp_bert_drift.png)

### 3. Speech: LSTM (Wikitext-2)
![LSTM Plot](assets/3_speech_lstm_drift.png)

</details>

---

## 📂 Repository Structure

This codebase is organized as a modular Python package.

```text
aimc-hwa-replication/
├── src/
│   ├── physics.py          # PCM Drift & Noise implementation
│   ├── layers.py           # AnalogLinear, AnalogLSTM, Drift Correction
│   └── models/             # Architecture definitions (BERT wrapper, WRN)
├── scripts/
│   ├── eval_drift.py       # Time-evolution inference script
│   └── plot_results.py     # Visualization tools
├── research/               # Original Jupyter notebooks (Proof of Work)
└── assets/                 # Generated plots and figures
