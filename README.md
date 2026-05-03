# EEG-Mamba-S
Official PyTorch implementation of **EEG-Mamba-S: A Lightweight State-Space Model for Few-Channel Wearable Seizure Prediction**

## Abstract
Epileptic seizure prediction suffers from high inter-subject variability, severe class imbalance, and the quadratic complexity of Transformer self-attention. We propose EEG-Mamba-S, a lightweight linear-time State-Space Model (SSM) tailored for 2-channel wearable EEG. A unified preprocessing pipeline, Mamba backbone, and Focal Loss with random undersampling are adopted to improve prediction performance and cross-dataset generalization.

Evaluated on SeizeIT2 wearable EEG and CHB-MIT clinical scalp EEG, our model achieves consistent high AUC with only 1.1M parameters and 1.4 ms inference latency, supporting real-time wearable edge deployment.

## Key Contributions
- Validated linear-time SSMs for seizure prediction under standard 2-channel wearable configuration.
- Empirically optimal strategy: random undersampling combined with Focal Loss ($\gamma=3.0, \alpha=0.5$).
- Lightweight architecture with robust cross-dataset reproducibility and ultra-low inference latency.

## Model Architecture
- Backbone: 4 stacked Mamba (S6) blocks ($d_{\text{model}}=200$, $d_{\text{state}}=16$, expand=2)
- Input shape: $(B, 2, 30, 200)$ reshaped to $(B, 60, 200)$
- Complexity: $O(L)$ linear time
- Total parameters: ~1.1M

## Preprocessing
0.5–40 Hz / 0.5–75 Hz band-pass filter + 50 Hz notch filter, resampled to 200 Hz, 30s sliding window, per-window / per-recording Z-score normalization, unified 2-channel bipolar mapping.

## Datasets
- SeizeIT2: 20-subset wearable behind-the-ear EEG
- CHB-MIT: 20-case clinical scalp EEG

## Results
| Dataset | ROC-AUC |
|--------|---------|
| SeizeIT2 | $0.9020 \pm 0.0067$ |
| CHB-MIT | $0.9048 \pm 0.0036$ |

Inference latency: 1.4 ms on RTX 5060 Laptop GPU

## Environment
```bash
pip install torch mne scikit-learn tqdm numpy pandas
pip install mamba-ssm
