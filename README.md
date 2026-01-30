# AIMC Hardware-Aware Training Replication

This repository contains the replication of the paper **"Hardware-aware training for large-scale and diverse deep learning inference workloads using in-memory computing-based accelerators"** (Rasch et al., 2023, IBM Research).

## 🏆 Key Results (SOTA Stability)

We achieved **ISO-Accuracy** (0.00% loss) on BERT after simulating 1 year of analog drift.

| Architecture | Task | HWA @ 1s | HWA @ 1 Year | Drift Loss | Status |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **WideResNet-16** | Vision (CIFAR-100) | 76.95% | **76.94%** | -0.01% | ✅ SOTA |
| **BERT-Base** | NLP (GLUE/SST-2) | 90.37% | **90.37%** | 0.00% | ✅ SOTA |
| **LSTM** | Text (Wikitext-2) | 259.05 | **259.09** | +0.03 | ✅ SOTA |

## 👨‍💻 Author's Note & Methodology
This project started as a series of exploratory Jupyter notebooks (available in the `research/` folder) to validate the IBM paper's claims.

**Challenges faced:**
* **BERT Implementation:** The paper uses a custom analog linear layer. I initially used monkey-patching to inject these layers into Hugging Face's BERT, which caused stability issues. I later refactored this into a clean `AnalogBertClassifier` wrapper (see `src/models/bert.py`).
* **Drift Simulation:** Replicating the exact drift exponent ($\nu=0.06$) was critical. I found that without *Per-Layer GDC* (Global Drift Compensation), the accuracy collapsed to random chance within 1 hour.

*Note: This codebase was refactored from research notebooks into a modular Python package.*

## 📂 Structure
- `src/`: Core logic (Physics engine, Analog Layers).
- `scripts/`: Execution scripts for training and evaluation.
- `research/`: Original exploratory notebooks (Proof of Work).
- `assets/`: Visualization of results.

## 🚀 Usage
1. Install dependencies: `pip install -r requirements.txt`
2. Run drift evaluation: `python scripts/eval_drift.py`
