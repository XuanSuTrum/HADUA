# HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition

This repository provides the official PyTorch implementation of **HADUA**, a hierarchical attention and dynamic uniform alignment framework for robust cross-subject multimodal emotion recognition using EEG and eye-movement (EM) signals.

HADUA is designed for cross-subject physiological emotion recognition, where the model is trained with labeled source-subject data and adapted to an unlabeled target subject. The framework jointly addresses modality heterogeneity, pseudo-label noise, and class-wise pseudo-label imbalance.

---

## Overview

Cross-subject emotion recognition from physiological signals is challenging due to large inter-subject variability and heterogeneous information between EEG and eye-movement modalities. HADUA addresses these challenges through three main components:

1. **Hierarchical Attention-based Multimodal Fusion**
   - EEG-specific self-attention;
   - eye-movement-specific self-attention;
   - EEG-guided cross-modal attention.

2. **Multi-level Distribution Alignment**
   - MMD for marginal distribution alignment;
   - CMMD for conditional distribution alignment.

3. **Confidence-driven Pseudo-label Optimization**
   - Soft Gaussian Weighting for confidence-aware pseudo-label reweighting;
   - Uniform Alignment for class-wise pseudo-label balancing.

---

## Repository Structure

```text
HADUA/
├── README.md
├── requirements.txt             # Python dependencies
├── main_zhibiao.py              # Main training and evaluation script
├── SDA_DDA_3.py                 # HADUA model
├── guessmatch.py                # Soft Gaussian Weighting / pseudo-label refinement
├── mmd.py                       # MMD loss
├── cmmd_5.py                      # CMMD loss
├── load_data2_multi_eye.py      # EEG + eye-movement data loader
└── utils.py                     # Utility functions
```
---

## Requirements

The code was developed with Python and PyTorch.

Recommended environment:

```text
Python >= 3.8
PyTorch >= 2.1.0
NumPy
scikit-learn
matplotlib
```

### Installation

Create a conda environment:

```bash
conda create -n hadua python=3.8
conda activate hadua
```

Install PyTorch and other dependencies:

```bash
pip install torch torchvision torchaudio
pip install numpy scikit-learn matplotlib
```

Alternatively, create a `requirements.txt` file with the following content:

```text
torch>=2.1.0
torchvision
torchaudio
numpy
scikit-learn
matplotlib
```

Then install dependencies by:

```bash
pip install -r requirements.txt
```

---

## Dataset Preparation

This repository supports multimodal EEG and eye-movement emotion recognition experiments on datasets such as **SEED** and **SEED-IV**.

Due to dataset license restrictions, the original datasets are not included in this repository. Please request the datasets from the official dataset provider and preprocess the EEG and eye-movement features before running the code.

A recommended directory structure is:

```text
data/
└── SEED/
    └── multi-mode/
        └── contact/
            ├── EEG/
            │   ├── 1.npy
            │   ├── 2.npy
            │   └── ...
            ├── EYE/
            │   ├── 1.npy
            │   ├── 2.npy
            │   └── ...
            └── Label/
                ├── 1.npy
                ├── 2.npy
                └── ...
```

Each subject should have three corresponding files:

```text
EEG/{subject_id}.npy
EYE/{subject_id}.npy
Label/{subject_id}.npy
```

Please modify the dataset paths in:

```text
load_data2_multi_eye.py
```

For example:

```python
EEG_FOLDER = "path/to/SEED/multi-mode/contact/EEG"
EYE_FOLDER = "path/to/SEED/multi-mode/contact/EYE"
LABEL_FOLDER = "path/to/SEED/multi-mode/contact/Label"
```

For SEED-IV, please use the corresponding SEED-IV feature directory and adjust the number of classes if necessary.

---

## Evaluation Protocol

The experiments follow a subject-level cross-subject domain adaptation protocol.

For each run:

```text
Source domain: labeled samples from source subjects
Target domain: unlabeled samples from one held-out target subject
```

Each model is trained for a fixed duration of 200 epochs, and the model obtained at the final epoch is used for evaluation. This follows the standard transductive unsupervised domain adaptation setting, where unlabeled target-domain samples are available during adaptation, while their ground-truth labels are accessed only after training for final performance evaluation.

## Hyperparameter Configuration for Reported Results

The following configuration is used for the main results reported in the paper.

### Training

| Hyperparameter | Value |
|---|---:|
| Optimizer | Adam |
| Batch size | 128 |
| Training epochs | 200 |
| Backbone learning rate | 5e-3 |
| Attention learning rate | 5e-2 |
| Weight decay | 1e-3 |

### Network Architecture

| Hyperparameter | Value |
|---|---:|
| EEG embedding dimension | 64 |
| Eye-movement embedding dimension | 64 |
| Attention hidden dimension | 128 |
| Number of attention heads | 16 |
| Attention dropout | 0.5 |
| Classifier hidden dimension | 32 |
| Classifier dropout | 0.5 |

### Soft Gaussian Weighting and Uniform Alignment

| Hyperparameter | Value |
|---|---:|
| EMA momentum $m$ | 0.999 |
| Initial standard deviation $\sigma_0$ | 1.0 |
| Maximum weight $\lambda_{\max}$ | 1.0 |
| Temperature $\tau$ | 1.0 |
| Initial alignment strength $\alpha_0$ | 0.3 |

### CMMD

| Hyperparameter | Value |
|---|---:|
| Kernel multiplier | 2.0 |
| Number of kernels | 5 |
| Internal CMMD scaling coefficient | 0.1 |

## Running HADUA

To train and evaluate HADUA, run:

```bash
python main_zhibiao.py
```

The script performs cross-subject training and evaluation. It reports the performance for each target subject and the average performance across subjects.

The main reported metrics include:

- Accuracy;
- Precision;
- Macro-F1;
- AUC;
- Confusion matrix.

A typical output format is:

```text
Processing test_id: 1
Transfer result: Acc: XX.XXXX, Precision: XX.XXXX, F1: XX.XXXX, AUC: XX.XXXX
Confusion Matrix:
...

Final Results:
Average Accuracy: XX.XXXX ± XX.XXXX
Average Precision: XX.XXXX ± XX.XXXX
Average F1-Score: XX.XXXX ± XX.XXXX
Average AUC: XX.XXXX ± XX.XXXX
Average Confusion Matrix:
...
```

---

## Main Results

The following results are reported under the cross-subject multimodal emotion recognition setting.

### Comparison with Existing Methods

| Method | SEED Acc | SEED Macro-F1 | SEED AUC | SEED-IV Acc | SEED-IV Macro-F1 | SEED-IV AUC |
|---|---:|---:|---:|---:|---:|---:|
| DGCNN | 79.95±9.02 | - | - | - | - | - |
| MHESA | - | - | - | 83.25±9.98 | - | - |
| CFDA-CSF | 90.04±5.46 | - | - | 89.60±6.65 | - | - |
| MMDA | 94.82±2.41 | 94.75±2.47 | - | 85.54±8.11 | 85.28±8.05 | - |
| CMSLNet | - | - | - | 83.15±9.84 | - | - |
| MACDB | 90.49±4.04 | - | - | 83.02±4.67 | - | - |
| CSMM | 94.96±5.27 | 95.21±7.96 | 96.20±3.98 | 89.82±6.22 | 90.03±6.19 | 93.01±4.25 |
| **HADUA** | **94.68±3.91** | **94.69±3.74** | **97.68±2.50** | **92.00±5.29** | **92.88±4.64** | **92.02±5.05** |

## Model Components

### Hierarchical Attention-based Multimodal Fusion

Implemented in:

```text
SDA_DDA_3.py
```

This module contains:

- EEG self-attention;
- eye-movement self-attention;
- EEG-guided cross-attention;
- multimodal feature fusion;
- classification head.

The fused representation is constructed from EEG self-attended features, eye-movement self-attended features, and cross-attended features.

---

### MMD-based Marginal Distribution Alignment

Implemented in:

```text
mmd.py
```

MMD is used to reduce global distribution discrepancy between source-domain and target-domain multimodal features.

---

### CMMD-based Conditional Distribution Alignment

Implemented in:

```text
cmmd_5.py
```

CMMD aligns class-conditional distributions between source and target domains. Since target-domain labels are unavailable during training, target-side class information is estimated using pseudo-label probabilities.

---

### Soft Gaussian Weighting and Uniform Alignment

Implemented in:

```text
guessmatch.py
```

Soft Gaussian Weighting assigns confidence-aware weights to target-domain pseudo-labels. Uniform Alignment adjusts target pseudo-label distributions to reduce class-wise imbalance during conditional alignment.

---

## Reproducibility Checklist

Before running the code, please check the following items:

- [ ] Dataset paths in `load_data2_multi_eye.py` have been correctly modified.
- [ ] EEG and eye-movement `.npy` files have been prepared.
- [ ] The number of emotion classes is correctly configured.
  - SEED: 3 classes;
  - SEED-IV: 4 classes.
- [ ] All required Python files are included in the repository.
- [ ] The source and target subjects are separated at the subject level.
- [ ] The target domain is evaluated only once after fixed-epoch training.
- [ ] Random seeds are fixed if deterministic behavior is required.

## Contact

For questions about the paper or code, please contact:

```text
Jiahao Tang
Xi'an Jiaotong University
Email: tangjiahao@stu.xjtu.edu.cn
```

## License

This repository is released for academic research purposes only.

Please check the licenses and usage agreements of the original datasets before using them. The datasets are not redistributed in this repository.

If you use this code, please cite the corresponding paper and follow the dataset license requirements.

## Acknowledgement

We thank the providers of the SEED and SEED-IV datasets and the open-source community for supporting reproducible research in affective computing and brain-computer interfaces.
