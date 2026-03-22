"""
Vision classifier framework for hand experience detection.

Uses a pretrained backbone (ResNet50 by default) with a multi-head
output that predicts scores for each rubric category independently.
"""

import torch
import torch.nn as nn
from torchvision import models


class HandClassifier(nn.Module):
    """
    Multi-head classifier that outputs per-category experience scores.

    Architecture:
        Pretrained backbone -> shared feature layer -> N scoring heads
        Each head outputs a single score for its rubric category.

    Args:
        backbone: Pretrained model name ("resnet50", "resnet18", "efficientnet_b0").
        category_max_points: List of max points per category (default: rubric v0.1).
        freeze_backbone: If True, freeze backbone weights for transfer learning.
        dropout: Dropout rate for scoring heads.
    """

    DEFAULT_CATEGORY_POINTS = [25, 20, 15, 15, 10, 10, 5]  # 7 categories, 100 total
    CATEGORY_NAMES = [
        "Texture Persistence",
        "Wear Localization",
        "Micro-Injury History",
        "Tendon & Vein Definition",
        "Nail Evidence",
        "Symmetry of Wear",
        "Climate & PPE Intelligence",
    ]

    def __init__(
        self,
        backbone: str = "resnet50",
        category_max_points: list[int] | None = None,
        freeze_backbone: bool = True,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.category_max_points = category_max_points or self.DEFAULT_CATEGORY_POINTS
        self.num_categories = len(self.category_max_points)

        # Load pretrained backbone
        self.backbone, backbone_out_features = self._build_backbone(backbone)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        # Shared feature layer
        self.shared = nn.Sequential(
            nn.Linear(backbone_out_features, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Per-category scoring heads — each outputs a single value in [0, max_points]
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(256, 64),
                nn.ReLU(),
                nn.Linear(64, 1),
                nn.Sigmoid(),  # Output in [0, 1], scaled to [0, max_points] in forward
            )
            for _ in self.category_max_points
        ])

    def _build_backbone(self, name: str) -> tuple[nn.Module, int]:
        """Build and return (backbone_without_fc, num_features)."""
        if name == "resnet50":
            model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, num_features

        if name == "resnet18":
            model = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
            num_features = model.fc.in_features
            model.fc = nn.Identity()
            return model, num_features

        if name == "efficientnet_b0":
            model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.DEFAULT)
            num_features = model.classifier[1].in_features
            model.classifier = nn.Identity()
            return model, num_features

        raise ValueError(f"Unsupported backbone: {name}. Use resnet50, resnet18, or efficientnet_b0.")

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """
        Forward pass.

        Args:
            x: Input images, shape (batch, 3, 224, 224).

        Returns:
            Dict mapping category name -> predicted scores (batch,).
        """
        features = self.backbone(x)
        shared = self.shared(features)

        scores = {}
        for i, (head, max_pts) in enumerate(zip(self.heads, self.category_max_points)):
            raw = head(shared).squeeze(-1)  # (batch,) in [0, 1]
            scores[self.CATEGORY_NAMES[i]] = raw * max_pts  # Scale to [0, max_pts]

        return scores

    def predict_total(self, x: torch.Tensor) -> torch.Tensor:
        """Return total score across all categories."""
        scores = self.forward(x)
        return sum(scores.values())

    def unfreeze_backbone(self, layers: int | None = None):
        """Unfreeze backbone for fine-tuning. If layers is None, unfreeze all."""
        params = list(self.backbone.parameters())
        if layers is None:
            for p in params:
                p.requires_grad = True
        else:
            for p in params[-layers:]:
                p.requires_grad = True
