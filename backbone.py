"""Modality-specific MLP feature extractors used by HADUA."""

from __future__ import annotations

import torch
from torch import nn


class MLPFeatureExtractor(nn.Module):
    """Map a fixed-length modality vector to a 64-dimensional embedding."""

    def __init__(self, input_dim: int, output_dim: int = 64) -> None:
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.layers = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
            nn.Linear(128, output_dim),
            nn.BatchNorm1d(output_dim),
            nn.LeakyReLU(negative_slope=0.01, inplace=True),
        )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 2 or inputs.shape[1] != self.input_dim:
            raise ValueError(
                f"expected [batch, {self.input_dim}] input, got {tuple(inputs.shape)}"
            )
        return self.layers(inputs)


class CFE(MLPFeatureExtractor):
    """EEG feature extractor for 62 channels x 5 frequency bands."""

    def __init__(self) -> None:
        super().__init__(input_dim=310)


class CFEEye(MLPFeatureExtractor):
    """Eye-movement feature extractor for the 33 published features."""

    def __init__(self) -> None:
        super().__init__(input_dim=33)


# Preserve the names used by the released training artifact.
CFE_eye = CFEEye
network_dict = {"CFE": CFE, "CFE_eye": CFEEye}
