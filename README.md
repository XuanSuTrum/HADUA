# HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition

This repository provides the PyTorch implementation of **HADUA**, a hierarchical attention and dynamic uniform alignment framework for robust cross-subject multimodal emotion recognition using EEG and eye-movement (EM) signals.

HADUA is designed to address three major challenges in cross-subject physiological emotion recognition:

- modality heterogeneity between EEG and eye-movement signals;
- noisy pseudo-labels during unsupervised target-domain adaptation;
- class-wise pseudo-label imbalance in conditional distribution alignment.

The source code is released to support reproducibility and independent verification of the experimental protocol.

---

## Overview

HADUA consists of three main components:

1. **Hierarchical Attention-based Multimodal Fusion**
   - EEG-specific self-attention;
   - eye-movement-specific self-attention;
   - EEG-guided cross-modal attention for multimodal interaction.

2. **Multi-level Distribution Alignment**
   - MMD for marginal distribution alignment;
   - CMMD for conditional distribution alignment using target-domain pseudo-labels.

3. **Confidence-driven Pseudo-label Optimization**
   - Soft Gaussian Weighting for confidence-aware pseudo-label reweighting;
   - Uniform Alignment for reducing class-wise pseudo-label imbalance.

---

## Repository Structure

```text
HADUA/
├── README.md
├── main_zhibiao.py              # Main training and evaluation script
├── SDA_DDA_3.py                 # HADUA model: hierarchical attention + adaptation modules
├── guessmatch.py                # Soft Gaussian Weighting / pseudo-label weighting
├── mmd.py                       # MMD loss
├── cmmd.py                      # CMMD loss
├── load_data2_multi_eye.py      # EEG + eye-movement data loader
└── utils.py                     # Utility functions
