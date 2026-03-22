"""
Dataset loader for hand experience images.

Expected directory structure:
    data/
    ├── images/
    │   ├── 001.jpg
    │   ├── 002.jpg
    │   └── ...
    └── labels.csv

labels.csv format:
    filename, texture_persistence, wear_localization, micro_injury_history,
    tendon_vein_definition, nail_evidence, symmetry_of_wear, climate_ppe
"""

from pathlib import Path

import torch
from torch.utils.data import Dataset
from torchvision import transforms

try:
    from PIL import Image
except ImportError:
    Image = None  # Deferred — error at runtime if used without Pillow


# Standard ImageNet normalization for pretrained backbones
DEFAULT_TRANSFORM = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

AUGMENT_TRANSFORM = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.RandomCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class HandDataset(Dataset):
    """
    PyTorch dataset for hand images with per-category scores.

    Args:
        image_dir: Path to directory containing images.
        labels_csv: Path to CSV with columns [filename, ...7 score columns].
        transform: Torchvision transform to apply. Defaults to resize + normalize.
        augment: If True, use augmentation transforms instead.
    """

    SCORE_COLUMNS = [
        "texture_persistence",
        "wear_localization",
        "micro_injury_history",
        "tendon_vein_definition",
        "nail_evidence",
        "symmetry_of_wear",
        "climate_ppe",
    ]

    def __init__(
        self,
        image_dir: str | Path,
        labels_csv: str | Path,
        transform: transforms.Compose | None = None,
        augment: bool = False,
    ):
        if Image is None:
            raise ImportError("Pillow is required for HandDataset. Install with: pip install Pillow")

        self.image_dir = Path(image_dir)
        self.transform = transform or (AUGMENT_TRANSFORM if augment else DEFAULT_TRANSFORM)

        self.samples = self._load_labels(Path(labels_csv))

    def _load_labels(self, csv_path: Path) -> list[dict]:
        """Parse labels CSV into list of {filename, scores} dicts."""
        import csv

        samples = []
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                filename = row.get("filename", "").strip()
                if not filename:
                    continue
                scores = []
                for col in self.SCORE_COLUMNS:
                    scores.append(float(row.get(col, 0.0)))
                samples.append({"filename": filename, "scores": scores})
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        sample = self.samples[idx]
        image_path = self.image_dir / sample["filename"]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        scores = torch.tensor(sample["scores"], dtype=torch.float32)
        return image, scores
