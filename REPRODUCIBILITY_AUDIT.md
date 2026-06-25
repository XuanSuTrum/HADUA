# HADUA code provenance and reproducibility audit

Audit date: 2026-06-24

## Scope and evidence

The audit examined every file in the public repository, every reachable commit
and remote ref, deleted paths, and unreachable Git objects. It also compared an
externally supplied experiment artifact with the paper proof and supplementary
material. No dataset, checkpoint, CSV result log, or training output was present.

The public repository has one branch (`main`), no tags, and a linear history.
Its initial code upload is commit `7fbef52`; the audited baseline is `e4e2041`.
No reachable or unreachable revision contains `backbone.py`, `cmmd_2.py` through
`cmmd_5.py`, `get_dataset.py`, `load_data2.py`, `SDA_DDA.py`, or `SDA_DDA_2.py`.
Consequently, no public commit before this repair is executable as uploaded.

The external artifact is the only available candidate for the experiment code:

| File | SHA-256 | Evidence |
|---|---|---|
| `backbone.py` | `2acea9880ce1f00e45d442800b504dbb22a8848197d8fc3d46828e3f781f25d7` | supplies the missing 310->64 EEG and 33->64 eye MLPs described in the paper |
| `cmmd_5.py` | `1c3ccc97fe71041f229d8fd96dcd474eff904e040544c530419afc0a632decde` | weighted soft-label CMMD called by the model |
| `main_.py` | `4855e820181c012ecdbd409498fa0a50750b9d8f8669f00ccd298e38c2e8d11f` | unlike public `main_zhibiao.py`, adds CMMD to total loss |

This makes the artifact the strongest available candidate for the code used to
produce the paper tables. It is not cryptographic proof of numerical provenance:
the original environment, processed arrays, checkpoints, logs, and exact run
commands are absent. The repaired repository therefore does not claim that
94.68% or any other paper value has been reproduced.

## Defects found and resolution

| Baseline defect | Consequence | Resolution |
|---|---|---|
| Missing backbone and imported CMMD variants | import fails immediately | restored the evidenced MLP backbone and one canonical CMMD implementation |
| Public training loss omitted the computed CMMD | published conditional alignment was inactive | total loss is exactly `Lcls + gamma_mmd*LMMD + gamma_cmmd*LCMMD` |
| Classifier parameters omitted from Adam | classifier never learns | optimizer receives `model.parameters()`; regression test checks classifier gradients |
| `MatchWeighting` constructed inside every forward call | EMA mean, variance, and class distribution reset each batch | made it a persistent `nn.Module` with registered checkpointable buffers |
| unconditional `.cuda()` in pseudo-label state | CPU and non-default devices fail | all buffers follow the model device |
| confidence threshold was zero | apparent filtering had no effect | removed hard selection, consistent with the paper's continuous weighting claim |
| UA output affected weights but raw probabilities entered CMMD | class balancing did not refine conditional labels | Equation (16) refinement now supplies CMMD target probabilities |
| CMMD multiplied an internal `0.1` and was weighted again outside | undocumented double scaling | loss module is unscaled; only `gamma_cmmd` controls it |
| default-weight CMMD path referenced an undefined tensor | `weights=None` crashes | covered by a gradient regression test |
| target labels were returned by the adaptation loader | training code could accidentally inspect them | adaptation dataset contains features only |
| target test accuracy evaluated every epoch and best epoch reported | target-test checkpoint selection leakage | fixed-epoch optimization followed by one final target evaluation |
| absolute Windows data/log paths | non-portable and leaked local layout | runtime `--data-root` and ignored relative `outputs/` |
| CLI used `parse_args(args=[])` and ignored user flags | advertised arguments did not work | normal command-line parsing with explicit options |
| weighted precision/F1 labeled as Macro-F1 | paper metric mismatch | macro precision, Macro-F1, and macro OVR AUC |
| `.data` used in kernel/pseudo-label paths | unsafe autograd behavior | detached only running statistics and kernel bandwidth estimates |

## Paper/code ambiguities that cannot be silently resolved

The proof contains mutually inconsistent experiment descriptions:

- the main implementation paragraph specifies learning rate `1e-4`, weight
  decay `5e-5`, batch size 64, 100 epochs, and both loss coefficients 0.5;
- the candidate artifact uses learning rate `0.05`, weight decay `1e-3`, batch
  size 128, and 200 epochs;
- the supplement calls batch size 256 and 250 epochs the optimum, but reports
  93.60% for that grid point rather than the main 94.68% result;
- Equation (17) and the candidate implement a sigmoid-decay UA coefficient,
  while the implementation paragraph says it increases linearly from 0 to 1;
- the paper mentions validation-based early stopping, but specifies no
  leakage-safe validation split and the candidate selects the best target-test
  epoch.

This repair chooses the main paper's optimizer/default loss configuration,
Equation (17)'s schedule, and fixed-epoch evaluation. Every choice is exposed as
a CLI option. Recovering an exact historical table requires the authors to add
the original run manifests and non-sensitive result logs; it must not be
reverse-engineered by target-test tuning.

## Remaining limitations

1. Licensed data are absent, so only synthetic unit tests can run in CI.
2. Preprocessing scripts and trial/session manifests were not found. The loader
   validates array dimensions but cannot independently reconstruct DE features.
3. The supplied attention artifact applies attention to one modality embedding
   token. That preserves the only evidenced architecture, but cannot by itself
   verify the paper's broader temporal-dependency interpretation.
4. A definitive numerical reproduction needs immutable dataset hashes,
   preprocessing configuration, per-dataset run manifests, seeds, and final
   subject-level metrics from a leakage-safe rerun.
