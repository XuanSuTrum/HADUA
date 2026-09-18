# HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition

This repository provides the official PyTorch implementation of **HADUA**, a hierarchical attention and dynamic uniform alignment framework for cross-subject multimodal emotion recognition using electroencephalography (EEG) and eye-movement (EM) signals.

HADUA is developed for transductive unsupervised domain adaptation (UDA): the model learns from labeled source-subject data and adapts to an unlabeled target subject. It jointly addresses multimodal heterogeneity, inter-subject distribution shift, pseudo-label noise, and class-wise pseudo-label imbalance.

* * *

## Overview

HADUA contains three collaborative components:

1. **Hierarchical Attention-based Multimodal Representation Learning**
  
  * EEG-specific self-attention;
  * eye-movement-specific self-attention;
  * EEG-guided cross-modal attention;
  * concatenation of the three 128-dimensional attention outputs into a 384-dimensional fused representation.
2. **Multi-level Distribution Alignment**
  
  * Maximum Mean Discrepancy (MMD) for marginal distribution alignment;
  * Conditional Maximum Mean Discrepancy (CMMD) for class-conditional distribution alignment;
  * multi-kernel Gaussian formulation with five kernels and a kernel multiplier of 2.0.
3. **Confidence-driven Pseudo-label Optimization**
  
  * Soft Gaussian Weighting assigns continuous reliability weights to target predictions without hard confidence filtering;
  * Uniform Alignment (UA) dynamically regularizes the target prediction distribution toward a uniform prior;
  * confidence-weighted soft class assignments are used to stabilize CMMD.

The overall training objective is

$$
\mathcal{L}=\mathcal{L}_{\mathrm{cls}}+\gamma_{\mathrm{mmd}}\mathcal{L}_{\mathrm{MMD}}+\gamma_{\mathrm{cmmd}}\mathcal{L}_{\mathrm{CMMD}}.
$$

The reference configuration uses $\gamma_{\mathrm{mmd}}=1.0$ and $\gamma_{\mathrm{cmmd}}=0.1$.

* * *

## Repository Structure

    HADUA/
    ├── README.md
    ├── requirements.txt              # Python dependencies
    ├── main_.py                      # LOSO training and evaluation entry point
    ├── SDA_DDA_3.py                  # HADUA network and adaptation pipeline
    ├── backbone.py                   # Modality-specific MLP backbones
    ├── softmatch2.py                 # Soft Gaussian Weighting and Uniform Alignment
    ├── mmd.py                        # MMD loss
    ├── cmmd_5.py                     # Confidence-weighted CMMD loss
    ├── load_data2_multi_eye.py       # EEG + eye-movement data loader
    └── utils.py                      # Utility functions

* * *

## Requirements

The reported experiments were implemented with **PyTorch 2.1.0**.

Recommended environment:

    Python >= 3.8
    PyTorch == 2.1.0
    NumPy
    scikit-learn
    matplotlib

### Installation

Create and activate a conda environment:

    conda create -n hadua python=3.8
    conda activate hadua

Install PyTorch and the remaining dependencies:

    pip install torch==2.1.0 torchvision torchaudio
    pip install numpy scikit-learn matplotlib

Alternatively, install all dependencies from `requirements.txt`:

    pip install -r requirements.txt

* * *

## Dataset Preparation

The experiments use the synchronized EEG and eye-movement data from **SEED**, **SEED-IV**, and **SEED-V**. The original datasets are not redistributed in this repository. Please request access from the official dataset provider and comply with the corresponding license and usage agreements.

| Dataset | Multimodal subjects used | Emotion classes | Trials per session |
| --- | --- | --- | --- |
| SEED | 12  | 3   | 15  |
| SEED-IV | 15  | 4   | 24  |
| SEED-V | 16  | 5   | 15  |

For SEED, EEG recordings are available from 15 participants, but synchronized EEG–eye-movement recordings are available for 12 participants; therefore, the multimodal experiments use $n=12$.

EEG signals are represented by differential entropy (DE) features from five frequency bands: delta, theta, alpha, beta, and gamma. With 62 EEG channels, each EEG sample contains 310 DE features. The eye-movement features describe pupil dynamics, dispersion, fixation, saccade, blink, and related oculomotor statistics.

A recommended directory structure is:

    data/
    ├── SEED/
    │   ├── EEG/
    │   ├── EYE/
    │   └── Label/
    ├── SEED-IV/
    │   ├── EEG/
    │   ├── EYE/
    │   └── Label/
    └── SEED-V/
        ├── EEG/
        ├── EYE/
        └── Label/

Each subject should have corresponding EEG, eye-movement, and label files, for example:

    EEG/{subject_id}.npy
    EYE/{subject_id}.npy
    Label/{subject_id}.npy

Update the dataset paths in `load_data2_multi_eye.py` before running the experiments. Also set the dataset-specific number of classes and target-subject range in the training script.

* * *

## Evaluation Protocol

### Primary Transductive UDA Protocol

The main results follow subject-level leave-one-subject-out (LOSO) transductive UDA:

    Source domain: all labeled samples from the remaining source subjects
    Target domain: all samples from one held-out subject, used without labels during adaptation
    Evaluation: target labels are accessed only after training

The source and target domains are completely disjoint at the subject level. During training and adaptation, target-domain labels are not used for feature preprocessing, model optimization, pseudo-label generation, Gaussian confidence estimation, Uniform Alignment, MMD, or CMMD.

Every model is trained for a fixed duration of **200 epochs**, and the **final-epoch model** is used for evaluation. Target labels are not used for early stopping or epoch selection.

The reference hyperparameter configuration was selected through offline sensitivity analysis: candidate configurations were trained without target labels, and target labels were accessed only after each completed run to compute evaluation metrics. The configuration with the highest overall average recognition accuracy was then used as one common setting for the main reported experiments.

### Restricted Target-domain Access Protocols

The supplementary experiments additionally evaluate:

* **Source-only:** no target samples are available during training or adaptation; the trained source model is evaluated on the complete target subject.
* **20%–80% held-out:** the first 20% of target samples, in their original order, are used without labels for adaptation; the remaining 80% are excluded from all training and adaptation operations and used only for final testing.

These restricted-access results are reported separately and are **not directly comparable** with the primary transductive results or with transductive baselines in the main comparison table.

For multi-class AUC, one-vs-rest (OvR) AUC is calculated for each class, macro-averaged within each target subject, and then averaged across LOSO folds.

* * *

## Hyperparameter Configuration for Reported Results

Unless otherwise stated in a sensitivity experiment, the following configuration is fixed across the main experiments.

### Training

| Hyperparameter | Value |
| --- | --- |
| Optimizer | Adam |
| Batch size | 128 |
| Training epochs | 200 |
| Backbone learning rate | $5\times10^{-3}$ |
| Attention learning rate | $5\times10^{-2}$ |
| Weight decay | $1\times10^{-3}$ |
| Model selection | Final epoch |

### Network Architecture

| Hyperparameter | Value |
| --- | --- |
| EEG embedding dimension | 64  |
| Eye-movement embedding dimension | 64  |
| Attention hidden dimension | 128 |
| Number of attention heads | 16  |
| Attention dropout | 0.5 |
| Fused feature dimension | 384 |
| Classifier hidden dimension | 32  |
| Classifier dropout | 0.5 |

### Soft Gaussian Weighting and Uniform Alignment

| Hyperparameter | Value |
| --- | --- |
| EMA momentum $m$ | 0.999 |
| Initial confidence mean $\mu_0$ | $1/C$ |
| Initial Gaussian variance $\sigma_0^2$ | 1.0 |
| Maximum sample weight $\lambda_{\max}$ | 1.0 |
| Temperature $\tau$ | 1.0 |
| Initial alignment strength $\alpha_0$ | 0.3 |
| Inflection point $T_0$ | 20  |
| Decay parameter $k$ | 6   |

The dynamic UA coefficient is

$$
\alpha_t=\frac{\alpha_0}{1+\exp\left(\frac{t-T_0}{k}\right)}.
$$

### Loss and CMMD

| Hyperparameter | Value |
| --- | --- |
| MMD coefficient $\gamma_{\mathrm{mmd}}$ | 1.0 |
| CMMD coefficient $\gamma_{\mathrm{cmmd}}$ | 0.1 |
| Kernel multiplier | 2.0 |
| Number of Gaussian kernels | 5   |
| Confidence-weight exponent $\beta$ | 0.5 |

* * *

## Running HADUA

After preparing the data and setting the dataset-specific paths and class number, run:

    python main_.py

The script performs LOSO cross-subject training and evaluation, reporting target-subject results and their mean and standard deviation across folds.

The reported metrics include:

* Accuracy;
* Macro-F1;
* macro-averaged OvR AUC;
* confusion matrix.

A typical output format is:

    Processing test_id: 1
    Transfer result: Acc: XX.XXXX, Macro-F1: XX.XXXX, AUC: XX.XXXX
    Confusion Matrix:
    ...
    
    Final Results:
    Average Accuracy: XX.XXXX ± XX.XXXX
    Average Macro-F1: XX.XXXX ± XX.XXXX
    Average AUC: XX.XXXX ± XX.XXXX
    Average Confusion Matrix:
    ...

* * *

## Main Results

### Comparison with Existing Methods

The following results use the primary LOSO transductive UDA protocol. Values are reported as mean $\pm$ standard deviation across target subjects.

| Method | SEED Acc. | SEED Macro-F1 | SEED AUC | SEED-IV Acc. | SEED-IV Macro-F1 | SEED-IV AUC | SEED-V Acc. | SEED-V Macro-F1 | SEED-V AUC |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| DGCNN | 79.95±9.02 | --  | --  | --  | --  | --  | --  | --  | --  |
| MHESA | --  | --  | --  | 83.25±9.98 | --  | --  | --  | --  | --  |
| CFDA-CSF | 93.05±6.22 | --  | --  | 85.72±11.02 | --  | --  | 88.49±12.32 | 88.41±11.84 | --  |
| MMDA | 94.82±2.41 | 94.75±2.47 | --  | 85.54±8.11 | 85.28±8.05 | --  | 91.03±12.94 | 90.92±12.88 | --  |
| CMSLNet | --  | --  | --  | 83.15±9.84 | --  | --  | 87.32±14.81 | --  | --  |
| MACDB | 90.49±4.04 | --  | --  | 83.02±4.67 | --  | --  | 85.98±12.06 | 84.57±13.27 | 90.88±7.87 |
| MSDA-Net | 83.08 | --  | --  | 65.10 | --  | --  | --  | --  | --  |
| R2GFANet | --  | --  | --  | 81.30±7.06 | 81.28 | --  | 78.89±10.45 | 78.90 | --  |
| CSMM | 94.96±5.27 | 95.21±7.96 | 96.20±3.98 | 89.82±6.22 | 90.03±6.19 | 93.01±4.25 | 89.22±9.59 | 89.95±8.85 | 93.11±5.97 |
| **HADUA** | **94.68±3.91** | **94.69±3.74** | **97.68±2.50** | **92.00±5.29** | **92.88±4.64** | **92.02±5.05** | **88.82±10.76** | **90.68±8.77** | **88.29±10.97** |

`--` indicates that the corresponding metric was not reported or that the experimental protocol was not directly comparable.

HADUA achieves competitive performance on SEED and SEED-V. On SEED-IV, it obtains the highest accuracy and Macro-F1 among the compared methods, while CSMM reports a higher AUC. On SEED, HADUA reports the highest AUC among the methods listed above.

### Restricted Target-domain Access Results

Macro-F1 and AUC are reported on the $[0,1]$ scale in this table.

| Dataset | Protocol | Accuracy (%) | Macro-F1 | AUC |
| --- | --- | --- | --- | --- |
| SEED | Source-only | 87.34±6.40 | 0.8730±0.0598 | 0.9485±0.0351 |
| SEED | 20%–80% held-out | 92.72±5.22 | 0.9261±0.0515 | 0.9758±0.0322 |
| SEED-IV | Source-only | 81.05±10.16 | 0.8050±0.0966 | 0.9110±0.0509 |
| SEED-IV | 20%–80% held-out | 86.75±7.25 | 0.8684±0.0667 | 0.9522±0.0410 |
| SEED-V | Source-only | 83.51±10.75 | 0.8205±0.1189 | 0.9326±0.0601 |
| SEED-V | 20%–80% held-out | 83.39±9.95 | 0.8083±0.1181 | 0.9070±0.0854 |

The 20% SEED-V adaptation subset follows the original sample order and contains no disgust samples. This class-composition mismatch should be considered when interpreting its restricted-access result.

### Ablation Study

| Configuration | SEED Acc. (%) | SEED-IV Acc. (%) |
| --- | --- | --- |
| Deep Feedforward Network | 85.18±5.60 | 81.00±11.11 |
| + Hierarchical Attention-based Multimodal Fusion | 90.94±4.62 | 88.94±6.89 |
| + Marginal Distribution Alignment | 93.50±3.99 | 90.89±7.07 |
| + Dynamic Gaussian Confidence-weighted Domain Adaptation | 94.42±4.22 | 91.46±7.11 |
| + Uniform Alignment (**HADUA**) | **94.68±3.91** | **92.00±5.29** |

### Computational Efficiency

Measurements were obtained on an NVIDIA GeForce RTX 3070 Ti with a batch size of 128, 50 warm-up iterations, and 200 repeated measurements with CUDA synchronization.

| Model | Network parameters (M) | Training time (ms/128 samples) | Inference time (ms/128 samples) |
| --- | --- | --- | --- |
| ConcatFusion | 0.177 | 3.711±0.690 | 0.927±0.204 |
| DualSelfAttn. | 0.263 | 5.758±0.799 | 1.378±0.168 |
| Hier.Attn. | 0.308 | 6.600±0.684 | 1.409±0.115 |
| **HADUA** | **0.308** | **27.048±2.815** | **1.490±0.130** |

The additional adaptation operations increase training cost but introduce no learnable parameters beyond the hierarchical-attention backbone. They are not required at inference, so HADUA retains inference latency comparable to Hier.Attn.

* * *

## Model Components

### Hierarchical Attention-based Multimodal Fusion

Implemented in `SDA_DDA_3.py`.

The module contains modality-specific MLP encoders, EEG self-attention, eye-movement self-attention, EEG-guided cross-attention, multimodal feature fusion, and the classification head.

### MMD-based Marginal Distribution Alignment

Implemented in `mmd.py`.

MMD reduces the global distribution discrepancy between source-domain and target-domain fused features.

### Confidence-weighted CMMD

Implemented in `cmmd_5.py`.

CMMD aligns source and target class-conditional distributions. Because target labels are unavailable during adaptation, the target-side class statistics are estimated from soft prediction probabilities and modulated by Gaussian confidence weights.

### Soft Gaussian Weighting and Uniform Alignment

Implemented in `softmatch2.py`.

Soft Gaussian Weighting continuously down-weights uncertain target predictions rather than discarding them with a hard threshold. UA estimates the target prediction distribution using an exponential moving average and applies a decreasing alignment schedule before confidence estimation. The adjusted confidence controls sample reliability, while the original soft class probabilities and confidence weights are used in CMMD.

* * *

## Reproducibility Checklist

* [ ] The official EEG and eye-movement datasets have been obtained under their respective licenses.
* [ ] Dataset paths in `load_data2_multi_eye.py` have been updated.
* [ ] EEG, eye-movement, and label files are aligned sample by sample.
* [ ] The number of classes is set correctly: SEED = 3, SEED-IV = 4, SEED-V = 5.
* [ ] The LOSO target-subject range is set correctly: SEED = 12, SEED-IV = 15, SEED-V = 16.
* [ ] Source and target subjects are completely separated at the subject level.
* [ ] Target labels are excluded from training, adaptation, early stopping, and epoch selection.
* [ ] The reference run uses batch size 128, 200 epochs, and final-epoch evaluation.
* [ ] Random seeds and software versions are recorded for deterministic reproduction.
* [ ] Multi-class AUC is computed using subject-wise macro-averaged OvR AUC.

* * *

## Citation

If you use this repository in your research, please cite the corresponding paper:

> Jiahao Tang, Youjun Li, Yangxuan Zheng, Xiangting Fan, Siyuan Lu, Nuo Zhang, Nan Yao, Xueping Li, and Zi-Gang Huang, “HADUA: Hierarchical Attention and Dynamic Uniform Alignment for Robust Cross-Subject Emotion Recognition.”

The complete BibTeX entry will be added after publication.

* * *

## Contact

For questions about the paper or code, please contact:

    Jiahao Tang
    Xi'an Jiaotong University
    Email: tangjiahao@stu.xjtu.edu.cn

Corresponding author: Zi-Gang Huang (`huangzg@xjtu.edu.cn`).

* * *

## License

This repository is released for academic research purposes only. Please check the licenses and usage agreements of the original datasets before use. The datasets are not redistributed in this repository.

* * *

## Acknowledgement

We thank the providers of the SEED, SEED-IV, and SEED-V datasets and the open-source community for supporting reproducible research in affective computing and brain-computer interfaces.
