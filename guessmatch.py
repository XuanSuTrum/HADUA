"""Stateful Soft Gaussian Weighting and Uniform Alignment."""

from __future__ import annotations

import math

import torch
from torch import nn


class MatchWeighting(nn.Module):
    """Refine target probabilities and compute continuous confidence weights.

    The running confidence and class-distribution statistics are buffers so they
    persist across mini-batches, move with the model device, and are stored in
    checkpoints. Statistics are estimated only from target predictions.
    """

    def __init__(
        self,
        num_classes: int,
        momentum: float = 0.999,
        lambda_max: float = 1.0,
        temperature: float = 1.0,
        alignment_strength: float = 0.3,
        alignment_midpoint: float = 20.0,
        alignment_slope: float = 6.0,
        initial_variance: float = 1.0,
        eps: float = 1e-8,
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        if not 0.0 <= momentum < 1.0:
            raise ValueError("momentum must be in [0, 1)")
        if initial_variance <= 0.0:
            raise ValueError("initial_variance must be positive")
        if temperature <= 0.0:
            raise ValueError("temperature must be positive")
        if not 0.0 <= alignment_strength <= 1.0:
            raise ValueError("alignment_strength must be in [0, 1]")
        if alignment_slope <= 0.0:
            raise ValueError("alignment_slope must be positive")

        self.num_classes = num_classes
        self.momentum = momentum
        self.lambda_max = lambda_max
        self.temperature = temperature
        self.alignment_strength = alignment_strength
        self.alignment_midpoint = alignment_midpoint
        self.alignment_slope = alignment_slope
        self.eps = eps

        self.register_buffer("confidence_mean", torch.tensor(1.0 / num_classes))
        self.register_buffer("confidence_variance", torch.tensor(initial_variance))
        self.register_buffer(
            "class_distribution", torch.full((num_classes,), 1.0 / num_classes)
        )
        self.register_buffer("num_updates", torch.tensor(0, dtype=torch.long))

    @torch.no_grad()
    def update_statistics(self, probabilities: torch.Tensor) -> None:
        """Update EMA statistics from an unlabeled target mini-batch."""
        self._validate_probabilities(probabilities)
        detached = probabilities.detach()
        confidence = detached.amax(dim=1)
        batch_mean = confidence.mean()
        # Equation (12) applies the B/(B-1) correction to the biased variance.
        batch_variance = confidence.var(unbiased=False)
        if confidence.numel() > 1:
            batch_variance = batch_variance * confidence.numel() / (confidence.numel() - 1)
        batch_distribution = detached.mean(dim=0)

        one_minus_m = 1.0 - self.momentum
        self.confidence_mean.mul_(self.momentum).add_(batch_mean, alpha=one_minus_m)
        self.confidence_variance.mul_(self.momentum).add_(
            batch_variance, alpha=one_minus_m
        )
        self.confidence_variance.clamp_(min=self.eps)
        self.class_distribution.mul_(self.momentum).add_(
            batch_distribution, alpha=one_minus_m
        )
        self.class_distribution.div_(self.class_distribution.sum().clamp_min(self.eps))
        self.num_updates.add_(1)

    def alignment_alpha(self, epoch: float) -> float:
        """Sigmoid-decay schedule from Equation (17) of the paper."""
        exponent = (float(epoch) - self.alignment_midpoint) / self.alignment_slope
        exponent = min(max(exponent, -60.0), 60.0)
        return self.alignment_strength / (1.0 + math.exp(exponent))

    def uniform_alignment(
        self, probabilities: torch.Tensor, epoch: float
    ) -> torch.Tensor:
        """Move the running class prior toward uniform before Equation (16)."""
        self._validate_probabilities(probabilities)
        uniform = torch.full_like(self.class_distribution, 1.0 / self.num_classes)
        alpha = self.alignment_alpha(epoch)
        adjusted_distribution = (
            alpha * uniform + (1.0 - alpha) * self.class_distribution
        )
        ratio = (
            adjusted_distribution / self.class_distribution.clamp_min(self.eps)
        ).pow(
            self.temperature
        )
        refined = probabilities * ratio
        return refined / refined.sum(dim=1, keepdim=True).clamp_min(self.eps)

    def compute_weights(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Compute the truncated Gaussian weights in Equation (8)."""
        self._validate_probabilities(probabilities)
        confidence = probabilities.amax(dim=1)
        mean = self.confidence_mean.to(dtype=probabilities.dtype)
        variance = self.confidence_variance.to(dtype=probabilities.dtype).clamp_min(
            self.eps
        )
        decay = torch.exp(-((confidence - mean) ** 2) / (2.0 * variance))
        return torch.where(
            confidence < mean,
            self.lambda_max * decay,
            torch.full_like(confidence, self.lambda_max),
        )

    def forward(
        self, probabilities: torch.Tensor, epoch: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        self.update_statistics(probabilities)
        weights = self.compute_weights(probabilities)
        refined = self.uniform_alignment(probabilities, epoch)
        return refined, weights

    def _validate_probabilities(self, probabilities: torch.Tensor) -> None:
        if probabilities.ndim != 2 or probabilities.shape[1] != self.num_classes:
            raise ValueError(
                "probabilities must have shape "
                f"[batch, {self.num_classes}], got {tuple(probabilities.shape)}"
            )
        if probabilities.shape[0] == 0:
            raise ValueError("target mini-batch must not be empty")
