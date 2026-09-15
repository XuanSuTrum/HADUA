import torch
import torch.nn as nn
import numpy as np


class MatchWeighting(nn.Module):
    """SoftMatch-style persistent prediction-statistics estimator.

    The running confidence mean/variance and target class distribution are
    maintained across mini-batches through EMA.  They are registered as
    buffers, so they follow the model across CPU/GPU devices but are not
    optimized by gradient descent.
    """

    def __init__(
        self,
        num_classes,
        momentum=0.999,
        lambda_max=1.0,
        temperature=1.0,
        alpha=0.3,
        alpha_midpoint=20.0,
        alpha_scale=6.0,
        eps=1e-8,
    ):
        super().__init__()
        self.num_classes = num_classes
        self.momentum = momentum
        self.lambda_max = lambda_max
        self.temperature = temperature
        self.alpha = alpha
        self.alpha_midpoint = alpha_midpoint
        self.alpha_scale = alpha_scale
        self.eps = eps

        # Match the manuscript / SoftMatch initialization:
        # mu_0 = 1/C, sigma_0^2 = 1.0, pbar_0 = uniform distribution.
        self.register_buffer("mu", torch.tensor(1.0 / num_classes, dtype=torch.float32))
        self.register_buffer("var", torch.tensor(1.0, dtype=torch.float32))
        self.register_buffer(
            "class_dist",
            torch.full((num_classes,), 1.0 / num_classes, dtype=torch.float32),
        )

    @torch.no_grad()
    def update_gaussian_params(self, probabilities):
        """Update running confidence mean and variance with EMA.

        The manuscript defines the batch variance with denominator B and then
        applies B/(B-1) before the EMA update.  This is equivalent to using an
        unbiased mini-batch variance in the EMA update.
        """
        max_probs = torch.max(probabilities.detach(), dim=1)[0]
        batch_mu = torch.mean(max_probs)

        batch_size = max_probs.numel()
        batch_var_biased = torch.var(max_probs, unbiased=False)
        if batch_size > 1:
            batch_var_unbiased = batch_var_biased * batch_size / (batch_size - 1)
        else:
            batch_var_unbiased = torch.zeros_like(batch_var_biased)

        batch_var_unbiased = torch.clamp(batch_var_unbiased, min=self.eps)

        self.mu.mul_(self.momentum).add_(batch_mu * (1.0 - self.momentum))
        self.var.mul_(self.momentum).add_(batch_var_unbiased * (1.0 - self.momentum))
        self.var.clamp_(min=self.eps)

    @torch.no_grad()
    def update_class_dist(self, probabilities):
        """EMA update of the target-domain predicted class distribution."""
        batch_class_dist = torch.mean(probabilities.detach(), dim=0)
        self.class_dist.mul_(self.momentum).add_(
            batch_class_dist * (1.0 - self.momentum)
        )
        self.class_dist.div_(self.class_dist.sum().clamp_min(self.eps))

    def uniform_alignment(self, probabilities, current_epoch):
        """Dynamic uniform alignment used by HADUA."""
        effective_alpha = self.alpha / (
            1.0 + np.exp((current_epoch - self.alpha_midpoint) / self.alpha_scale)
        )
        uniform_dist = torch.full_like(
            self.class_dist, 1.0 / self.num_classes
        )
        adjusted_dist = (
            effective_alpha * uniform_dist
            + (1.0 - effective_alpha) * self.class_dist
        )

        adjust_factor = (
            adjusted_dist / (self.class_dist + self.eps)
        ) ** self.temperature
        adjusted_probs = probabilities * adjust_factor.unsqueeze(0)
        adjusted_probs = adjusted_probs / adjusted_probs.sum(
            dim=1, keepdim=True
        ).clamp_min(self.eps)
        return adjusted_probs

    def compute_weights(self, probabilities, current_epoch):
        """Compute confidence weights and persist EMA state across batches.

        As in the existing HADUA implementation, the Gaussian statistics are
        estimated from the original target probabilities, whereas the sample
        confidence used in the weighting function is obtained after UA.
        """
        # 1) Update target class-distribution EMA and perform UA.
        self.update_class_dist(probabilities)
        adjusted_probs = self.uniform_alignment(probabilities, current_epoch)
        adjusted_confidence = torch.max(adjusted_probs, dim=1)[0]

        # 2) Update confidence-statistics EMA using original predictions.
        self.update_gaussian_params(probabilities)

        # 3) Truncated-Gaussian weighting. self.var stores sigma_t^2.
        weights = torch.full_like(adjusted_confidence, self.lambda_max)
        mask = adjusted_confidence < self.mu
        weights[mask] = self.lambda_max * torch.exp(
            -((adjusted_confidence[mask] - self.mu) ** 2)
            / (2.0 * self.var.clamp_min(self.eps))
        )
        return weights

    @torch.no_grad()
    def get_stats(self, current_epoch=None):
        """Return detached statistics for debugging/logging only."""
        stats = {
            "mu": float(self.mu.item()),
            "var": float(self.var.item()),
            "sigma": float(torch.sqrt(self.var).item()),
            "class_dist": self.class_dist.detach().cpu().tolist(),
        }
        if current_epoch is not None:
            stats["alpha"] = float(
                self.alpha
                / (1.0 + np.exp((current_epoch - self.alpha_midpoint) / self.alpha_scale))
            )
        return stats