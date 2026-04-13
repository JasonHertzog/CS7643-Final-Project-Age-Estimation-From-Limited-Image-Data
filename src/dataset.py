"""
UTKFace PyTorch Dataset for use with CS7643 Final Project: Age Estimation from Limited Image Data

Filename format: AGE_GENDER_RACE_TIMESTAMP.jpg
  AGE    : integer 0–116
  GENDER : 0 = Male, 1 = Female
  RACE   : 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Other
"""
import csv
from pathlib import Path

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image

class UTKFaceDataset(Dataset):
    """PyTorch Dataset for UTKFace dataset

    Expects a CSV in the directory data/processed with columns:
        path   – absolute path to the image file
        age    – integer age label
        gender – 0 (male) or 1 (female)
        race   – 0–4 where 0 = White, 1 = Black, 2 = Asian, 3 = Indian, 4 = Other
    """

    # Age bin boundaries (right-inclusive upper edges)
    AGE_BINS = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
    AGE_BIN_LABELS = ["0-9", "10-19", "20-29", "30-39",
                      "40-49", "50-59", "60-69", "70-79", "80+"]

    def __init__(self, csv_path: str | Path, transform=None):
        """
        Args:
            csv_path  : Path to split CSV (train.csv / val.csv / test.csv).
            transform : torchvision transform applied to each PIL image.
        """
        self.transform = transform
        self.records: list[dict] = []

        with open(csv_path, newline="") as f:
            for row in csv.DictReader(f):
                self.records.append({
                    "path":   row["path"],
                    "age":    int(row["age"]),
                    "gender": int(row["gender"]),
                    "race":   int(row["race"]),
                })

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int):
        rec = self.records[idx]
        image = Image.open(rec["path"]).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)

        labels = {
            "age":     torch.tensor(rec["age"],                      dtype=torch.float32),
            "age_bin": torch.tensor(self.age_to_bin(rec["age"]),     dtype=torch.long),
            "gender":  torch.tensor(rec["gender"],                   dtype=torch.long),
            "race":    torch.tensor(rec["race"],                     dtype=torch.long),
        }
        return image, labels


    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------
    @classmethod
    def age_to_bin(cls, age: int) -> int:
        """Map a raw age value to its bin index (for classification tasks)."""
        for i in range(len(cls.AGE_BINS) - 1):
            if age < cls.AGE_BINS[i + 1]:
                return i
        return len(cls.AGE_BINS) - 2  # clamp to last bin

# ---------------------------------------------------------------------------
# Transform factories
# ---------------------------------------------------------------------------

def get_transforms(split: str = "train", image_size: int = 224) -> transforms.Compose:
    """Return standard image transforms for the given split.

    Training augmentations: resize + horizontal flip + colour jitter.
    Val / test: deterministic resize only.
    """
    # Normalization values for ImageNet (since UTKFace images are similar in style to ImageNet).
    # this is easier than computing the mean and std of the UTKFace dataset, and should work well enough for our purposes.
    _IMAGENET_MEAN = [0.485, 0.456, 0.406]
    _IMAGENET_STD = [0.229, 0.224, 0.225]
    normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STD)

    if split == "train":
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize,
        ])

    # for test or validation dataset, there is no augmentation
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        normalize,
    ])


# ---------------------------------------------------------------------------
# DataLoader factory
# ---------------------------------------------------------------------------
# This can be called directly from the training script to get DataLoaders for all splits, which can then be passed to the training loop.

def get_dataloaders(
    processed_dir: str | Path,
    batch_size: int = 32,
    image_size: int = 224,
    num_workers: int = 4,
) -> dict[str, DataLoader]:
    """Create DataLoaders for all three splits.

    Args:
        processed_dir : Directory containing train.csv, val.csv, test.csv
        batch_size    : Mini-batch size.
        num_workers   : Parallel workers for data loading.

    Returns:
        ``{"train": ..., "val": ..., "test": ...}``
    """
    processed_dir = Path(processed_dir)
    loaders: dict[str, DataLoader] = {}

    for split in ("train", "test", "val"):
        csv_path = processed_dir / f"{split}.csv"
        if not csv_path.exists():
            raise FileNotFoundError(
                f"Split CSV not found: {csv_path}\n"
                "Run `python scripts/preprocess_data.py` first."
            )

        dataset = UTKFaceDataset(csv_path, transform=get_transforms(split, image_size))
        loaders[split] = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=(split == "train"),
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available(),
            drop_last=(split == "train"),
        )

    return loaders
