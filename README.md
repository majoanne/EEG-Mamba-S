# EEG-Mamba-S
Official PyTorch implementation of **EEG-Mamba-S: A Lightweight State-Space Model for Few-Channel Wearable Seizure Prediction**

[![GitHub](https://img.shields.io/badge/GitHub-majoanne-blue)](https://github.com/majoanne/EEG-Mamba-S)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## Abstract
The unpredictable nature of epileptic seizures imposes a relentless psychological and physical burden on patients, driving the need for continuous, real‑time EEG monitoring. Automated seizure prediction, however, remains limited by three fundamental obstacles: high inter‑subject variability, extreme preictal‑to‑interictal class imbalance, and the quadratic complexity of self‑attention mechanisms.

To address these challenges, we propose **EEG‑Mamba‑S**, a lightweight State-Space Model (SSM) designed explicitly for two‑channel wearable EEG. We introduce a unified preprocessing pipeline, construct a compact Mamba backbone with linear-time complexity, and integrate random undersampling with Focal Loss (γ=3.0, α=0.5) to combat severe class imbalance.

Evaluated on the SeizeIT2 wearable database and CHB-MIT scalp EEG dataset, EEG-Mamba-S achieves strong cross-domain performance with only **1.1M parameters** and **1.4ms inference latency**, making it ideal for real-time edge/wearable deployment.

**Source Code**: [https://github.com/majoanne/EEG-Mamba-S](https://github.com/majoanne/EEG-Mamba-S)

## Keywords
Seizure prediction; Mamba state-space model; Wearable EEG; Few-channel EEG; Edge computing

---

## Core Contributions
1. **First validation** of linear-time SSMs for seizure prediction under a clinical 2-channel wearable setup.
2. **Optimal training strategy**: Random undersampling + Focal Loss (γ=3.0, α=0.5) for few-channel Mamba networks.
3. **Ultra-efficient model**: 1.1M parameters, 1.4ms inference latency, robust cross-dataset generalization.

---

## Model Architecture
- **Backbone**: 4 stacked Mamba (S6) blocks (`d_model=200`, `d_state=16`, expand=2)
- **Input Reshape**: `(B, 2, 30, 200)` → `(B, 60, 200)`
- **Classifier**: 3-layer lightweight MLP with ReLU and Dropout
- **Complexity**: O(L) linear time complexity (no Transformer bottleneck)
- **Total Parameters**: ~1.1 Million

---

## Unified Preprocessing Pipeline
1. **Filtering**: 0.5–40 Hz (SeizeIT2) / 0.5–75 Hz (CHB-MIT) + 50 Hz notch filter
2. **Channel Mapping**: Standardized 2-channel bipolar configuration
3. **Normalization**: Per-window Z-score (SeizeIT2) / Per-recording Z-score (CHB-MIT)
4. **Resampling**: 200 Hz, segmented into 30s sliding windows
5. **Input Shape**: Fixed tensor `(2, 30, 200)`

---

## Datasets
We evaluate on two public EEG datasets:
1. **SeizeIT2**: Wearable behind-the-ear EEG (20-subject subset, focal epilepsy)
2. **CHB-MIT**: Clinical scalp EEG (20-case subset, pediatric refractory seizures)

---

## Experimental Results
All models trained from scratch (3 random seeds: 3407, 42, 1234)

| Dataset       | ROC-AUC          |
|---------------|------------------|
| SeizeIT2      | 0.9020 ± 0.0067  |
| CHB-MIT       | 0.9048 ± 0.0036  |

### Model Efficiency
- Inference Latency: **1.4 ms** (NVIDIA RTX 5060 Laptop GPU, FP32)
- Memory Footprint: **4.3 MB** (FP32)
- Total FLOPs: ~0.12 G

---

## Ablation Study (SeizeIT2, Seed 3407)
| Configuration                  | ROC-AUC  |
|--------------------------------|----------|
| Baseline (Full model)          | 0.9127   |
| 1 Mamba Layer                  | 0.8007   |
| BCE Loss (w/ balance)          | 0.8987   |
| Focal Loss γ=4.0               | 0.9043   |

---

## Environment Setup
```bash
# Install core dependencies
pip install torch mne scikit-learn tqdm numpy pandas
# Install Mamba-SSM backbone
pip install mamba-ssm
