"""Leakage-safe loading for preprocessed EEG and eye-movement features."""

from __future__ import annotations

from pathlib import Path
from typing import Sequence

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def _class_indices(labels: np.ndarray) -> np.ndarray:
    if labels.ndim == 2 and labels.shape[1] > 1:
        labels = labels.argmax(axis=1)
    else:
        labels = labels.reshape(-1)
    if not np.issubdtype(labels.dtype, np.integer):
        if not np.allclose(labels, np.round(labels)):
            raise ValueError("labels must be integer indices or one-hot vectors")
        labels = np.round(labels)
    return labels.astype(np.int64, copy=False)


def _paths(data_root: str | Path, subject_id: str | int) -> dict[str, Path]:
    root = Path(data_root).expanduser()
    name = f"{subject_id}.npy"
    return {
        "EEG": root / "EEG" / name,
        "EYE": root / "EYE" / name,
        "Label": root / "Label" / name,
    }


def load_features(
    data_root: str | Path, subject_id: str | int
) -> np.ndarray:
    """Load and concatenate one subject's EEG and eye features."""
    paths = _paths(data_root, subject_id)
    missing = [str(paths[key]) for key in ("EEG", "EYE") if not paths[key].is_file()]
    if missing:
        raise FileNotFoundError("missing preprocessed subject files: " + ", ".join(missing))
    eeg = np.load(paths["EEG"], allow_pickle=False)
    eye = np.load(paths["EYE"], allow_pickle=False)
    if eeg.ndim != 2 or eeg.shape[1] != 310:
        raise ValueError(f"{paths['EEG']} must have shape [samples, 310]")
    if eye.ndim != 2 or eye.shape[1] != 33:
        raise ValueError(f"{paths['EYE']} must have shape [samples, 33]")
    if eeg.shape[0] != eye.shape[0]:
        raise ValueError(f"sample counts do not match for subject {subject_id}")
    return np.concatenate((eeg.astype(np.float32), eye.astype(np.float32)), axis=1)


def load_labels(data_root: str | Path, subject_id: str | int) -> np.ndarray:
    path = _paths(data_root, subject_id)["Label"]
    if not path.is_file():
        raise FileNotFoundError(f"missing preprocessed label file: {path}")
    return _class_indices(np.load(path, allow_pickle=False))


def create_domain_loaders(
    data_root: str | Path,
    target_subject: str | int,
    subject_ids: Sequence[str | int],
    batch_size: int,
    *,
    seed: int = 0,
    num_workers: int = 0,
) -> tuple[DataLoader, DataLoader]:
    """Create LOSO source and unlabeled-target adaptation loaders.

    Target adaptation batches contain a one-element tuple ``(features,)``. They
    never expose or load target labels.
    """
    if batch_size < 2:
        raise ValueError("batch_size must be at least 2 because the backbone uses batch norm")
    normalized_ids = [str(subject_id) for subject_id in subject_ids]
    target_id = str(target_subject)
    if target_id not in normalized_ids:
        raise ValueError("target_subject must be included in subject_ids")
    if len(set(normalized_ids)) != len(normalized_ids):
        raise ValueError("subject_ids must be unique")
    if len(normalized_ids) < 2:
        raise ValueError("LOSO adaptation requires at least two subjects")

    source_features: list[np.ndarray] = []
    source_labels: list[np.ndarray] = []
    target_features: np.ndarray | None = None
    for subject_id in normalized_ids:
        combined = load_features(data_root, subject_id)
        if subject_id == target_id:
            target_features = combined
        else:
            source_features.append(combined)
            labels = load_labels(data_root, subject_id)
            if labels.shape[0] != combined.shape[0]:
                raise ValueError(f"sample counts do not match for subject {subject_id}")
            source_labels.append(labels)

    assert target_features is not None
    source_tensor = torch.from_numpy(np.concatenate(source_features, axis=0))
    source_label_tensor = torch.from_numpy(np.concatenate(source_labels, axis=0))
    target_tensor = torch.from_numpy(target_features)
    if source_tensor.shape[0] < batch_size or target_tensor.shape[0] < batch_size:
        raise ValueError("each domain must contain at least one full training batch")

    source_generator = torch.Generator().manual_seed(seed)
    target_generator = torch.Generator().manual_seed(seed + 1)
    source_loader = DataLoader(
        TensorDataset(source_tensor, source_label_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        generator=source_generator,
    )
    target_adaptation_loader = DataLoader(
        TensorDataset(target_tensor),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        generator=target_generator,
    )
    return source_loader, target_adaptation_loader


def create_evaluation_loader(
    data_root: str | Path,
    target_subject: str | int,
    batch_size: int,
    *,
    num_workers: int = 0,
) -> DataLoader:
    """Load target labels only after training has irreversibly finished."""
    target_features = load_features(data_root, target_subject)
    target_labels = load_labels(data_root, target_subject)
    if target_features.shape[0] != target_labels.shape[0]:
        raise ValueError(f"sample counts do not match for subject {target_subject}")
    return DataLoader(
        TensorDataset(torch.from_numpy(target_features), torch.from_numpy(target_labels)),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=num_workers,
    )
