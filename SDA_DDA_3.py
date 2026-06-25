"""HADUA model: hierarchical fusion with MMD and weighted CMMD."""

from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F

import backbone
import cmmd
import mmd
from guessmatch import MatchWeighting


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        scores = torch.bmm(queries, keys.transpose(1, 2)) / math.sqrt(
            queries.shape[-1]
        )
        attention = self.dropout(torch.softmax(scores, dim=-1))
        return torch.bmm(attention, values)


def _split_heads(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch, tokens, hidden = inputs.shape
    if hidden % num_heads:
        raise ValueError("hidden dimension must be divisible by num_heads")
    inputs = inputs.reshape(batch, tokens, num_heads, hidden // num_heads)
    return inputs.permute(0, 2, 1, 3).reshape(
        batch * num_heads, tokens, hidden // num_heads
    )


def _merge_heads(inputs: torch.Tensor, num_heads: int) -> torch.Tensor:
    batch_heads, tokens, head_dim = inputs.shape
    if batch_heads % num_heads:
        raise ValueError("attention batch is incompatible with num_heads")
    batch = batch_heads // num_heads
    inputs = inputs.reshape(batch, num_heads, tokens, head_dim)
    return inputs.permute(0, 2, 1, 3).reshape(batch, tokens, num_heads * head_dim)


class MultiHeadAttention(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        hidden_dim: int,
        num_heads: int,
        dropout: float,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads:
            raise ValueError("hidden_dim must be divisible by num_heads")
        self.num_heads = num_heads
        self.attention = ScaledDotProductAttention(dropout)
        self.query = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.key = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.value = nn.Linear(feature_dim, hidden_dim, bias=False)
        self.output = nn.Linear(hidden_dim, hidden_dim, bias=False)

    def forward(
        self, queries: torch.Tensor, keys: torch.Tensor, values: torch.Tensor
    ) -> torch.Tensor:
        projected_queries = _split_heads(self.query(queries), self.num_heads)
        projected_keys = _split_heads(self.key(keys), self.num_heads)
        projected_values = _split_heads(self.value(values), self.num_heads)
        attended = self.attention(
            projected_queries, projected_keys, projected_values
        )
        return self.output(_merge_heads(attended, self.num_heads))


class Transfer_Net(nn.Module):
    """Official HADUA architecture reconstructed from the training artifact."""

    def __init__(
        self,
        num_class: int,
        base_net: str = "CFE",
        base_net_eye: str = "CFE_eye",
        transfer_loss: str = "mmd",
        width: int = 32,
        num_hiddens: int = 128,
        num_heads: int = 16,
        dropout: float = 0.5,
        gaussian_momentum: float = 0.999,
        gaussian_initial_variance: float = 1.0,
        ua_temperature: float = 1.0,
        ua_strength: float = 0.3,
        ua_midpoint: float = 20.0,
        ua_slope: float = 6.0,
    ) -> None:
        super().__init__()
        if base_net not in backbone.network_dict or base_net_eye not in backbone.network_dict:
            raise ValueError("unknown backbone name")
        if transfer_loss != "mmd":
            raise ValueError("HADUA currently supports only MMD marginal alignment")

        self.num_class = num_class
        self.transfer_loss = transfer_loss
        self.base_network = backbone.network_dict[base_net]()
        self.base_network_eye = backbone.network_dict[base_net_eye]()
        feature_dim = 64
        self.self_attention_eeg = MultiHeadAttention(
            feature_dim, num_hiddens, num_heads, dropout
        )
        self.self_attention_eye = MultiHeadAttention(
            feature_dim, num_hiddens, num_heads, dropout
        )
        self.cross_attention = MultiHeadAttention(
            feature_dim, num_hiddens, num_heads, dropout
        )
        self.classifier = nn.Sequential(
            nn.Linear(num_hiddens * 3, width),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(width, num_class),
        )
        self.match_weighting = MatchWeighting(
            num_classes=num_class,
            momentum=gaussian_momentum,
            temperature=ua_temperature,
            alignment_strength=ua_strength,
            alignment_midpoint=ua_midpoint,
            alignment_slope=ua_slope,
            initial_variance=gaussian_initial_variance,
        )

    @staticmethod
    def _split_modalities(inputs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        if inputs.ndim != 2 or inputs.shape[1] != 343:
            raise ValueError(f"expected [batch, 343] EEG+eye input, got {tuple(inputs.shape)}")
        return inputs[:, :310], inputs[:, 310:]

    def encode(self, inputs: torch.Tensor) -> torch.Tensor:
        eeg, eye = self._split_modalities(inputs)
        eeg_features = self.base_network(eeg).unsqueeze(1)
        eye_features = self.base_network_eye(eye).unsqueeze(1)
        eeg_self = self.self_attention_eeg(eeg_features, eeg_features, eeg_features)
        eye_self = self.self_attention_eye(eye_features, eye_features, eye_features)
        # Published direction: EEG is query; eye movement is key/value.
        cross = self.cross_attention(eeg_features, eye_features, eye_features)
        return torch.cat(
            (eeg_self.squeeze(1), eye_self.squeeze(1), cross.squeeze(1)), dim=1
        )

    def forward(
        self,
        epoch: float,
        source: torch.Tensor,
        target: torch.Tensor,
        source_labels: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        source_features = self.encode(source)
        target_features = self.encode(target)
        source_logits = self.classifier(source_features)
        target_probabilities = torch.softmax(self.classifier(target_features), dim=1)

        refined_probabilities, weights = self.match_weighting(
            target_probabilities, epoch
        )
        if source_labels.ndim == 1:
            source_soft_labels = F.one_hot(
                source_labels.long(), num_classes=self.num_class
            ).to(dtype=source_features.dtype)
        elif source_labels.ndim == 2 and source_labels.shape[1] == self.num_class:
            source_soft_labels = source_labels.to(dtype=source_features.dtype)
        else:
            raise ValueError("source_labels must be class indices or one-hot labels")

        marginal_loss = mmd.mmd_rbf_noaccelerate(
            source_features, target_features
        )
        conditional_loss = cmmd.cmmd(
            source_features,
            target_features,
            source_soft_labels,
            refined_probabilities,
            weights,
        )
        return source_logits, marginal_loss, conditional_loss

    def predict(self, inputs: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.encode(inputs))


# Clearer alias for new integrations while preserving the released class name.
HADUA = Transfer_Net
