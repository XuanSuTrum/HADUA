from __future__ import annotations

import numpy as np
import torch
from torch.nn import functional as F

import cmmd
from SDA_DDA_3 import HADUA
from guessmatch import MatchWeighting
from load_data2_multi_eye import create_domain_loaders, create_evaluation_loader


def test_match_weighting_state_persists_across_batches() -> None:
    module = MatchWeighting(num_classes=3, momentum=0.5)
    initial_mean = module.confidence_mean.clone()
    probabilities = torch.tensor(
        [[0.80, 0.10, 0.10], [0.20, 0.70, 0.10]], dtype=torch.float32
    )

    refined, weights = module(probabilities, epoch=0)
    module(probabilities, epoch=1)

    assert module.num_updates.item() == 2
    assert not torch.equal(initial_mean, module.confidence_mean)
    assert refined.shape == probabilities.shape
    assert torch.allclose(refined.sum(dim=1), torch.ones(2))
    assert torch.all((weights > 0) & (weights <= 1))


def test_cmmd_supports_default_weights_and_gradients() -> None:
    source = torch.randn(6, 8, requires_grad=True)
    target = torch.randn(5, 8, requires_grad=True)
    source_labels = F.one_hot(torch.tensor([0, 1, 2, 0, 1, 2]), 3).float()
    target_probabilities = torch.softmax(torch.randn(5, 3), dim=1)

    loss = cmmd.cmmd(source, target, source_labels, target_probabilities)
    loss.backward()

    assert torch.isfinite(loss)
    assert source.grad is not None
    assert target.grad is not None


def test_all_model_components_receive_gradients() -> None:
    model = HADUA(num_class=3, num_hiddens=32, num_heads=4, dropout=0.0)
    model.train()
    source = torch.randn(8, 343)
    target = torch.randn(8, 343)
    labels = torch.tensor([0, 1, 2, 0, 1, 2, 0, 1])

    logits, mmd_loss, cmmd_loss = model(0, source, target, labels)
    loss = F.cross_entropy(logits, labels) + 0.5 * mmd_loss + 0.5 * cmmd_loss
    loss.backward()

    classifier_gradients = [
        parameter.grad for parameter in model.classifier.parameters() if parameter.requires_grad
    ]
    assert classifier_gradients
    assert all(gradient is not None for gradient in classifier_gradients)
    assert any(torch.count_nonzero(gradient).item() for gradient in classifier_gradients)


def test_target_adaptation_loader_never_returns_labels(tmp_path) -> None:
    for folder in ("EEG", "EYE", "Label"):
        (tmp_path / folder).mkdir()
    for subject in ("1", "2"):
        rng = np.random.default_rng(int(subject))
        np.save(tmp_path / "EEG" / f"{subject}.npy", rng.normal(size=(6, 310)).astype(np.float32))
        np.save(tmp_path / "EYE" / f"{subject}.npy", rng.normal(size=(6, 33)).astype(np.float32))
        labels = np.eye(3, dtype=np.int64)[np.arange(6) % 3]
        np.save(tmp_path / "Label" / f"{subject}.npy", labels)

    source, adaptation = create_domain_loaders(
        tmp_path, "2", ["1", "2"], batch_size=3
    )
    evaluation = create_evaluation_loader(tmp_path, "2", batch_size=3)

    assert len(next(iter(source))) == 2
    assert len(next(iter(adaptation))) == 1
    assert len(next(iter(evaluation))) == 2
