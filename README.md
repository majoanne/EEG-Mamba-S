# EEG-Mamba-S
Official PyTorch implementation of **EEG-Mamba-S: A Lightweight State-Space Model for Few-Channel Wearable Seizure Prediction**

## Overview
Epileptic seizure prediction imposes heavy psychological and physical burdens on patients, requiring real-time, unobtrusive EEG monitoring. However, it faces three core challenges: high inter-subject variability, extreme preictal-to-interictal class imbalance, and quadratic complexity of Transformer self-attention.

**EEG-Mamba-S** is a lightweight State-Space Model (SSM) designed explicitly for **2-channel wearable EEG seizure prediction**, with linear-time complexity, ultra-low parameters, and strong cross-dataset generalization.

## Key Contributions
1. The first rigorous validation of linear-time SSMs for seizure prediction under a clinical 2-channel wearable setup.
2. Empirically optimal pairing: random undersampling + Focal Loss (γ=3.0, α=0.5) for few-channel Mamba networks.
3. Ultra-efficient model: 1.1M parameters, 1.4ms inference latency, suitable for edge/wearable deployment.

## Model Architecture
- Backbone: 4 stacked Mamba (S6) blocks (d_model=200, d_state=16, expand=2)
- Input shape: (B, 2, 30, 200) → reshaped to (B, 60, 200) for sequence modeling
- Classifier: Lightweight MLP with ReLU and Dropout
- Complexity: O(L) linear time complexity (no Transformer bottleneck)

## Datasets
We evaluate on two public EEG datasets:
1. **SeizeIT2**: Wearable behind-the-ear EEG (focal epilepsy, 20-subject subset)
2. **CHB-MIT**: Clinical scalp EEG (pediatric refractory seizures, 20-case subset)

### Unified Preprocessing Pipeline
- Band-pass filter: 0.5–40 Hz (SeizeIT2) / 0.5–75 Hz (CHB-MIT)
- 50 Hz notch filter for powerline interference
- Resample to 200 Hz, segment into 30s sliding windows
- Z-score normalization: per-window (SeizeIT2) / per-recording (CHB-MIT)
- Unified 2-channel bipolar mapping

## Environment Setup
```bash
# Install core dependencies
pip install torch mne scikit-learn tqdm numpy pandas
# Install mamba-ssm backbone
pip install mamba-ssm