"""Confidence-weighted conditional maximum mean discrepancy."""

from __future__ import annotations

import torch

from mmd import gaussian_kernel


def cmmd(
    source: torch.Tensor,
    target: torch.Tensor,
    source_labels: torch.Tensor,
    target_probabilities: torch.Tensor,
    weights: torch.Tensor | None = None,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Average class-wise RKHS centroid distance.

    ``source_labels`` and ``target_probabilities`` are soft class assignments.
    Target confidence weights affect only target class centroids. The caller is
    responsible for applying the global CMMD coefficient exactly once.
    """
    if target.shape[0] == 0:
        return source.sum() * 0.0
    if source_labels.ndim != 2 or target_probabilities.ndim != 2:
        raise ValueError("source labels and target probabilities must be matrices")
    if source_labels.shape[0] != source.shape[0]:
        raise ValueError("source label count does not match source features")
    if target_probabilities.shape[0] != target.shape[0]:
        raise ValueError("target probability count does not match target features")
    if source_labels.shape[1] != target_probabilities.shape[1]:
        raise ValueError("source and target class counts do not match")

    source_labels = source_labels.to(dtype=source.dtype)
    target_probabilities = target_probabilities.to(dtype=target.dtype)
    if weights is None:
        weights = torch.ones(target.shape[0], dtype=target.dtype, device=target.device)
    if weights.ndim != 1 or weights.shape[0] != target.shape[0]:
        raise ValueError("weights must have one value per target sample")
    weighted_target = target_probabilities * weights.to(target.dtype).unsqueeze(1)

    kernels = gaussian_kernel(
        source, target, kernel_mul=kernel_mul, kernel_num=kernel_num, fix_sigma=fix_sigma
    )
    source_count = source.shape[0]
    xx = kernels[:source_count, :source_count]
    yy = kernels[source_count:, source_count:]
    xy = kernels[:source_count, source_count:]

    losses: list[torch.Tensor] = []
    for class_index in range(source_labels.shape[1]):
        source_mass = source_labels[:, class_index]
        target_mass = weighted_target[:, class_index]
        source_total = source_mass.sum()
        target_total = target_mass.sum()
        # A mini-batch cannot estimate a source class centroid when that class
        # is absent. Skipping it avoids a synthetic zero-centroid penalty.
        if source_total.detach().item() <= eps or target_total.detach().item() <= eps:
            continue
        source_mass = source_mass / source_total.clamp_min(eps)
        target_mass = target_mass / target_total.clamp_min(eps)
        loss = (
            source_mass @ xx @ source_mass
            + target_mass @ yy @ target_mass
            - 2.0 * (source_mass @ xy @ target_mass)
        )
        losses.append(loss)

    if not losses:
        return source.sum() * 0.0
    return torch.stack(losses).mean().clamp_min(0.0)
