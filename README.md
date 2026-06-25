# HADUA

Official PyTorch implementation of **HADUA: Hierarchical Attention and Dynamic
Uniform Alignment for Robust Cross-Subject Emotion Recognition**.

HADUA combines modality-specific EEG/eye feature extractors, hierarchical
attention fusion, marginal MMD alignment, and confidence-weighted conditional
MMD. Soft Gaussian Weighting continuously down-weights uncertain target
predictions, while Uniform Alignment reduces class collapse in target
pseudo-labels.

## Reproducibility status

This branch restores the missing backbone and the CMMD implementation found in
the available experiment artifact, then fixes defects that made the public
snapshot incomplete or methodologically unsafe. The implementation has
synthetic unit tests, but the paper's numerical results have **not** been
re-verified in CI because SEED, SEED-IV, and SEED-V are licensed datasets and
are not distributed here. See [REPRODUCIBILITY_AUDIT.md](REPRODUCIBILITY_AUDIT.md)
for provenance, resolved defects, and remaining limitations.

Paper-reported results (not hard-coded and not claimed as reproduced here):

| Dataset | Accuracy | Macro-F1 | Macro-AUC |
|---|---:|---:|---:|
| SEED | 94.68 +/- 3.91 | 94.69 +/- 3.74 | 97.68 +/- 2.50 |
| SEED-IV | 92.00 +/- 5.29 | 92.88 +/- 4.64 | 92.02 +/- 5.05 |
| SEED-V | 88.82 +/- 10.76 | 90.68 +/- 8.77 | 88.29 +/- 10.97 |

## Installation

Python 3.10 or newer is recommended.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

For tests:

```bash
python -m pip install -r requirements-dev.txt
pytest -q
```

## Data preparation

Raw datasets are intentionally excluded. Request them from their official
provider and export one NumPy file per subject:

```text
<data-root>/
├── EEG/
│   ├── 1.npy       # [samples, 310]: 62 channels x 5 DE bands
│   └── ...
├── EYE/
│   ├── 1.npy       # [samples, 33]
│   └── ...
└── Label/
    ├── 1.npy       # [samples] indices or [samples, classes] one-hot
    └── ...
```

The three arrays for a subject must contain the same number of samples. Paths
are supplied at runtime; no machine-specific absolute path is stored in code.

## Training

The SEED multimodal subset used by the supplied artifact contains 12 subject
files:

```bash
python main_zhibiao.py \
  --data-root /path/to/preprocessed/SEED \
  --subjects 1,2,3,4,5,8,9,10,11,12,13,14
```

Run a single LOSO target for a smoke test:

```bash
python main_zhibiao.py \
  --data-root /path/to/preprocessed/SEED \
  --subjects 1,2,3,4,5,8,9,10,11,12,13,14 \
  --target-subject 1 \
  --epochs 1
```

Defaults follow the main paper implementation paragraph:

- Adam, learning rate `1e-4`, weight decay `5e-5`;
- batch size `64`, fixed `100` epochs;
- `gamma_mmd = 0.5`, `gamma_cmmd = 0.5`;
- Soft Gaussian EMA momentum `0.999`, initial variance `1.0`;
- UA temperature `1.0`, initial strength `0.3`.

All values are command-line options. Use dataset-appropriate subject IDs and
`--num-classes` for SEED-IV or SEED-V rather than editing source files.

## Leakage-safe protocol

Each run holds out one complete subject as the target domain. The adaptation
loader yields only target features; its dataset has no label field. Source
classification, MMD, Soft Gaussian Weighting, UA, and CMMD cannot receive
target labels.

Training runs for a fixed epoch count. Target accuracy is not computed during
training and cannot select a checkpoint or tune hyperparameters. The target
label file is not opened until the separate evaluation loader is constructed
after all parameter updates. Labels are then read for one final evaluation. Using
the same unlabeled target features for transductive adaptation and final
evaluation is standard transductive UDA; using their labels before final
evaluation is prohibited.

Outputs are written under `outputs/`, which is ignored by Git.

## Repository layout

```text
backbone.py                    modality-specific 310->64 and 33->64 MLPs
SDA_DDA_3.py                   hierarchical attention and HADUA model
guessmatch.py                  stateful Soft Gaussian Weighting and UA
mmd.py                         marginal MMD
cmmd.py                        confidence-weighted soft-label CMMD
load_data2_multi_eye.py        LOSO data loading and label isolation
main_zhibiao.py                CLI training and one-shot evaluation
tests/test_hadua.py            synthetic regression tests
REPRODUCIBILITY_AUDIT.md       provenance and method/code audit
```
