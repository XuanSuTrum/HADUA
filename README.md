# HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition

This repository provides the official PyTorch implementation of:

**HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition**

HADUA is a cross-subject multimodal emotion recognition framework for EEG and eye-movement (EM) signals. The framework aims to improve the robustness of multimodal physiological emotion recognition under subject distribution shifts by jointly addressing modality heterogeneity, pseudo-label noise, and class-wise pseudo-label imbalance.

> Paper status: under review  
> Code status: public release for reproducibility and review

---

## Overview

Cross-subject emotion recognition from physiological signals remains challenging because EEG and eye-movement signals exhibit strong inter-subject variability and multimodal heterogeneity. HADUA integrates multimodal representation learning with unsupervised domain adaptation.

The framework mainly consists of three components:

1. **Hierarchical attention-based multimodal fusion**
   - Modality-specific self-attention is used to model intra-modal dependencies in EEG and eye-movement features.
   - Cross-modal attention is used to model semantic interactions between EEG and eye-movement modalities.

2. **Multi-level distribution alignment**
   - Marginal distribution alignment is performed using Maximum Mean Discrepancy (MMD).
   - Conditional distribution alignment is performed using Conditional MMD (CMMD) with target-domain pseudo-labels.

3. **Confidence-driven pseudo-label optimization**
   - Soft Gaussian Weighting is used to down-weight uncertain target-domain pseudo-labels.
   - Uniform Alignment is used to reduce class-wise pseudo-label imbalance during adaptation.

---

## Method Pipeline

For each cross-subject experiment, one subject is selected as the target domain, and the remaining subjects are used as source-domain subjects.

The training pipeline is as follows:

1. Load labeled source-domain EEG and EM features.
2. Load unlabeled target-domain EEG and EM features.
3. Extract modality-specific representations using EEG and EM backbone networks.
4. Apply self-attention to EEG and EM features separately.
5. Apply cross-attention for EEG-guided multimodal fusion.
6. Compute the source-domain classification loss.
7. Compute MMD for marginal distribution alignment.
8. Generate target-domain pseudo-label probabilities.
9. Apply Soft Gaussian Weighting and Uniform Alignment to refine pseudo-label supervision.
10. Compute CMMD for conditional distribution alignment.
11. Evaluate the trained model on the held-out target subject.

---

## Data Leakage Prevention

To ensure a strict cross-subject evaluation protocol, this implementation follows subject-level splitting.

- The source domain and target domain are separated at the **subject level**.
- The target subject is never included in the source-domain training set.
- Target-domain ground-truth labels are not used during training or adaptation.
- Pseudo-labels are generated only from model predictions on unlabeled target-domain samples.
- The statistics used in Soft Gaussian Weighting are estimated from prediction confidence scores, not from target-domain labels.
- Target-domain labels are used only for final evaluation.

This protocol is designed to avoid subject leakage and target-label leakage.

---

## Repository Structure

```text
HADUA/
├── README.md
├── main_zhibiao.py              # Main training and evaluation script
├── SDA_DDA_3.py                 # HADUA model with hierarchical attention and adaptation modules
├── guessmatch.py                # Soft Gaussian Weighting and Uniform Alignment
├── mmd.py                       # MMD loss
├── cmmd.py                      # CMMD loss
├── load_data2_multi_eye.py      # EEG and eye-movement data loader
└── utils.py                     # Utility functions
