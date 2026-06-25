"""Leakage-safe command-line training entry point for HADUA."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    roc_auc_score,
)
from torch import nn

from SDA_DDA_3 import HADUA
from load_data2_multi_eye import create_domain_loaders, create_evaluation_loader


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train HADUA with LOSO transductive UDA")
    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
        help="directory containing EEG/, EYE/, and Label/ preprocessed arrays",
    )
    parser.add_argument(
        "--subjects",
        default="1,2,3,4,5,8,9,10,11,12,13,14",
        help="comma-separated subject file stems",
    )
    parser.add_argument(
        "--target-subject",
        default=None,
        help="run only one target; by default every listed subject is held out once",
    )
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=5e-5)
    parser.add_argument("--gamma-mmd", type=float, default=0.5)
    parser.add_argument("--gamma-cmmd", type=float, default=0.5)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--gaussian-momentum", type=float, default=0.999)
    parser.add_argument("--gaussian-initial-variance", type=float, default=1.0)
    parser.add_argument("--ua-temperature", type=float, default=1.0)
    parser.add_argument("--ua-strength", type=float, default=0.3)
    parser.add_argument("--ua-midpoint", type=float, default=20.0)
    parser.add_argument("--ua-slope", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument(
        "--device", default="auto", help="auto, cpu, cuda, or an explicit torch device"
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs"),
        help="directory for aggregate JSON metrics",
    )
    return parser.parse_args(argv)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> torch.device:
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(requested)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    return device


def build_model(args: argparse.Namespace) -> HADUA:
    return HADUA(
        num_class=args.num_classes,
        num_hiddens=args.hidden_dim,
        num_heads=args.num_heads,
        gaussian_momentum=args.gaussian_momentum,
        gaussian_initial_variance=args.gaussian_initial_variance,
        ua_temperature=args.ua_temperature,
        ua_strength=args.ua_strength,
        ua_midpoint=args.ua_midpoint,
        ua_slope=args.ua_slope,
    )


def train_fixed_epochs(
    model: HADUA,
    source_loader: torch.utils.data.DataLoader,
    target_loader: torch.utils.data.DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epochs: int,
    gamma_mmd: float,
    gamma_cmmd: float,
) -> list[dict[str, float]]:
    """Train without reading target labels or target evaluation metrics."""
    criterion = nn.CrossEntropyLoss()
    history: list[dict[str, float]] = []
    for epoch in range(epochs):
        model.train()
        target_iterator = iter(target_loader)
        totals = {"classification": 0.0, "mmd": 0.0, "cmmd": 0.0, "total": 0.0}
        samples = 0
        for source_features, source_labels in source_loader:
            try:
                (target_features,) = next(target_iterator)
            except StopIteration:
                target_iterator = iter(target_loader)
                (target_features,) = next(target_iterator)

            source_features = source_features.to(device)
            source_labels = source_labels.to(device)
            target_features = target_features.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits, mmd_loss, cmmd_loss = model(
                epoch, source_features, target_features, source_labels
            )
            classification_loss = criterion(logits, source_labels)
            total_loss = (
                classification_loss
                + gamma_mmd * mmd_loss
                + gamma_cmmd * cmmd_loss
            )
            total_loss.backward()
            optimizer.step()

            batch_size = source_features.shape[0]
            samples += batch_size
            totals["classification"] += classification_loss.item() * batch_size
            totals["mmd"] += mmd_loss.item() * batch_size
            totals["cmmd"] += cmmd_loss.item() * batch_size
            totals["total"] += total_loss.item() * batch_size

        epoch_metrics = {name: value / samples for name, value in totals.items()}
        epoch_metrics["epoch"] = float(epoch + 1)
        history.append(epoch_metrics)
        print(
            f"epoch {epoch + 1:03d}/{epochs:03d} "
            f"cls={epoch_metrics['classification']:.6f} "
            f"mmd={epoch_metrics['mmd']:.6f} "
            f"cmmd={epoch_metrics['cmmd']:.6f} "
            f"total={epoch_metrics['total']:.6f}"
        )
    return history


@torch.no_grad()
def evaluate_once(
    model: HADUA,
    evaluation_loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
) -> dict[str, object]:
    """Read target labels once, after optimization and model selection are over."""
    model.eval()
    labels: list[np.ndarray] = []
    predictions: list[np.ndarray] = []
    probabilities: list[np.ndarray] = []
    for features, batch_labels in evaluation_loader:
        logits = model.predict(features.to(device))
        batch_probabilities = torch.softmax(logits, dim=1)
        labels.append(batch_labels.numpy())
        predictions.append(batch_probabilities.argmax(dim=1).cpu().numpy())
        probabilities.append(batch_probabilities.cpu().numpy())
    target = np.concatenate(labels)
    predicted = np.concatenate(predictions)
    probability = np.concatenate(probabilities)
    try:
        auc = float(
            roc_auc_score(
                target,
                probability,
                labels=np.arange(num_classes),
                average="macro",
                multi_class="ovr",
            )
        )
    except ValueError:
        auc = float("nan")
    return {
        "accuracy": float(accuracy_score(target, predicted) * 100.0),
        "macro_precision": float(
            precision_score(target, predicted, average="macro", zero_division=0)
        ),
        "macro_f1": float(f1_score(target, predicted, average="macro", zero_division=0)),
        "macro_auc": auc,
        "confusion_matrix": confusion_matrix(
            target, predicted, labels=np.arange(num_classes)
        ).tolist(),
    }


def aggregate(results: list[dict[str, object]]) -> dict[str, dict[str, float]]:
    summary: dict[str, dict[str, float]] = {}
    for metric in ("accuracy", "macro_precision", "macro_f1", "macro_auc"):
        values = np.asarray([result[metric] for result in results], dtype=float)
        summary[metric] = {
            "mean": float(np.nanmean(values)),
            "std": float(np.nanstd(values, ddof=1 if values.size > 1 else 0)),
        }
    return summary


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    subjects = [value.strip() for value in args.subjects.split(",") if value.strip()]
    targets = [args.target_subject] if args.target_subject else subjects
    if any(target not in subjects for target in targets):
        raise ValueError("every target subject must occur in --subjects")
    device = resolve_device(args.device)
    all_results: list[dict[str, object]] = []

    for run_index, target_subject in enumerate(targets):
        print(f"\nTarget subject: {target_subject}")
        set_seed(args.seed + run_index)
        source_loader, adaptation_loader = create_domain_loaders(
            args.data_root,
            target_subject,
            subjects,
            args.batch_size,
            seed=args.seed + run_index,
            num_workers=args.num_workers,
        )
        model = build_model(args).to(device)
        # All trainable modules, including the classifier, are optimized.
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=args.learning_rate,
            weight_decay=args.weight_decay,
        )
        history = train_fixed_epochs(
            model,
            source_loader,
            adaptation_loader,
            optimizer,
            device,
            args.epochs,
            args.gamma_mmd,
            args.gamma_cmmd,
        )
        # Target labels are not read from disk until all parameter updates end.
        evaluation_loader = create_evaluation_loader(
            args.data_root,
            target_subject,
            args.batch_size,
            num_workers=args.num_workers,
        )
        result = evaluate_once(model, evaluation_loader, device, args.num_classes)
        result["target_subject"] = target_subject
        result["final_training_loss"] = history[-1]
        all_results.append(result)
        print(
            f"final target evaluation: accuracy={result['accuracy']:.4f}, "
            f"macro_f1={result['macro_f1']:.4f}, macro_auc={result['macro_auc']:.4f}"
        )

    payload = {
        "protocol": "LOSO transductive UDA; fixed-epoch training; one final target evaluation",
        "subjects": subjects,
        "results": all_results,
        "summary": aggregate(all_results),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output_path = args.output_dir / "results.json"
    output_path.write_text(json.dumps(payload, indent=2, allow_nan=True) + "\n")
    print(f"\nSaved metrics to {output_path}")
    print(json.dumps(payload["summary"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
