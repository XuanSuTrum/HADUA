"""Gaussian-kernel maximum mean discrepancy."""

from __future__ import annotations

import torch


def gaussian_kernel(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
    eps: float = 1e-8,
) -> torch.Tensor:
    if source.ndim != 2 or target.ndim != 2:
        raise ValueError("source and target must be two-dimensional")
    if source.shape[1] != target.shape[1]:
        raise ValueError("source and target feature dimensions must match")
    total = torch.cat((source, target), dim=0)
    squared_distance = torch.cdist(total, total, p=2).square()
    if fix_sigma is None:
        count = total.shape[0]
        denominator = max(count * count - count, 1)
        bandwidth = squared_distance.detach().sum() / denominator
    else:
        bandwidth = torch.as_tensor(
            fix_sigma, dtype=total.dtype, device=total.device
        )
    bandwidth = bandwidth.clamp_min(eps) / (kernel_mul ** (kernel_num // 2))
    return sum(
        torch.exp(-squared_distance / (bandwidth * (kernel_mul**index)))
        for index in range(kernel_num)
    )


def mmd_rbf_noaccelerate(
    source: torch.Tensor,
    target: torch.Tensor,
    kernel_mul: float = 2.0,
    kernel_num: int = 5,
    fix_sigma: float | None = None,
) -> torch.Tensor:
    kernels = gaussian_kernel(source, target, kernel_mul, kernel_num, fix_sigma)
    source_count = source.shape[0]
    xx = kernels[:source_count, :source_count]
    yy = kernels[source_count:, source_count:]
    xy = kernels[:source_count, source_count:]
    yx = kernels[source_count:, :source_count]
    return (xx.mean() + yy.mean() - xy.mean() - yx.mean()).clamp_min(0.0)


# Compatibility with older experiment scripts.
mmd_rbf_accelerate = mmd_rbf_noaccelerate
