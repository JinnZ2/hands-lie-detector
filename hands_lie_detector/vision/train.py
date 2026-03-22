"""
Training loop for the hand experience classifier.
"""

from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

from .classifier import HandClassifier
from .dataset import HandDataset


class Trainer:
    """
    Training harness for HandClassifier.

    Args:
        model: HandClassifier instance.
        device: Torch device ("cuda", "cpu", or "mps").
        lr: Learning rate.
        weight_decay: L2 regularization.
    """

    def __init__(
        self,
        model: HandClassifier | None = None,
        device: str | None = None,
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
    ):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        self.model = (model or HandClassifier()).to(self.device)
        self.optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=lr,
            weight_decay=weight_decay,
        )
        self.criterion = nn.MSELoss()

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        count = 0

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(images)
            # Stack predictions in category order to match target columns
            pred_tensor = torch.stack(list(predictions.values()), dim=1)

            loss = self.criterion(pred_tensor, targets)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item() * images.size(0)
            count += images.size(0)

        return total_loss / count if count > 0 else 0.0

    @torch.no_grad()
    def validate(self, dataloader: DataLoader) -> dict[str, float]:
        """Run validation. Returns dict with loss and per-category MAE."""
        self.model.eval()
        total_loss = 0.0
        category_errors = [0.0] * self.model.num_categories
        count = 0

        for images, targets in dataloader:
            images = images.to(self.device)
            targets = targets.to(self.device)

            predictions = self.model(images)
            pred_tensor = torch.stack(list(predictions.values()), dim=1)

            loss = self.criterion(pred_tensor, targets)
            total_loss += loss.item() * images.size(0)

            # Per-category MAE
            abs_errors = (pred_tensor - targets).abs().mean(dim=0)
            for i in range(self.model.num_categories):
                category_errors[i] += abs_errors[i].item() * images.size(0)

            count += images.size(0)

        if count == 0:
            return {"loss": 0.0}

        metrics = {"loss": total_loss / count}
        for i, name in enumerate(HandClassifier.CATEGORY_NAMES):
            metrics[f"mae_{name}"] = category_errors[i] / count
        return metrics

    def fit(
        self,
        image_dir: str | Path,
        labels_csv: str | Path,
        epochs: int = 30,
        batch_size: int = 32,
        val_split: float = 0.2,
        augment: bool = True,
        save_path: str | Path | None = None,
    ) -> list[dict]:
        """
        Full training run from data paths.

        Args:
            image_dir: Path to image directory.
            labels_csv: Path to labels CSV.
            epochs: Number of training epochs.
            batch_size: Batch size.
            val_split: Fraction of data for validation.
            augment: Use data augmentation for training.
            save_path: If set, save best model weights here.

        Returns:
            List of per-epoch metrics dicts.
        """
        dataset = HandDataset(image_dir, labels_csv, augment=augment)
        val_size = int(len(dataset) * val_split)
        train_size = len(dataset) - val_size
        train_set, val_set = random_split(dataset, [train_size, val_size])

        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=2)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, num_workers=2)

        history = []
        best_val_loss = float("inf")

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.validate(val_loader)

            record = {"epoch": epoch + 1, "train_loss": train_loss, **val_metrics}
            history.append(record)

            print(
                f"Epoch {epoch + 1}/{epochs} — "
                f"train_loss: {train_loss:.4f}, "
                f"val_loss: {val_metrics['loss']:.4f}"
            )

            if save_path and val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                torch.save(self.model.state_dict(), save_path)
                print(f"  Saved best model (val_loss: {best_val_loss:.4f})")

        return history
