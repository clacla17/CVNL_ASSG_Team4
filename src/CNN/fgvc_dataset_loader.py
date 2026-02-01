"""
FGVC Aircraft Dataset Loader
Adapted to match your CNN notebook style
"""

import os
from pathlib import Path
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class FGVCAircraftDataset(Dataset):
    """
    Load FGVC Aircraft dataset from official splits.
    Each split file line is: image_name aircraft_model
    """

    def __init__(
        self,
        root_dir,
        split_file,
        transform=None,
        label_to_idx=None,
        idx_to_label=None,
        allow_new_labels=True,
        limit=None,
        verbose=True,
    ):
        self.root_dir = Path(root_dir)
        self.image_dir = self.root_dir / 'data' / 'images'
        self.transform = transform
        self.allow_new_labels = allow_new_labels
        self.verbose = verbose

        # Load image names and labels from split file
        self.image_list = []
        self.labels = []
        self.label_to_idx = label_to_idx or {}
        self.idx_to_label = idx_to_label or {}

        self._load_split(split_file, limit=limit)

    def _load_split(self, split_file, limit=None):
        if not os.path.exists(split_file):
            raise FileNotFoundError("Split file not found: {}".format(split_file))

        # Map individual aircraft models to families (only keep target families)
        family_mapping = {
            # A320 Family (A318, A319, A320, A321)
            'A320': 'A320_Family',
            
            # B737 Family (includes 737 variants and 717 which is similar)
            'Boeing 737': 'B737_Family',
            'Boeing 717': 'B737_Family',
            
            # A330 Family (A330-200, A330-300, etc.)
            'A330': 'A330_Family',
            
            # B777 Family (777-200, 777-300, etc.)
            'Boeing 777': 'B777_Family',
            
            # B787 Family (787-8, 787-9, 787-10)
            'Boeing 787': 'B787_Family',
        }

        label_idx = len(self.label_to_idx)

        with open(split_file, 'r') as f:
            for line in f:
                parts = line.strip().split(' ')
                if len(parts) < 2:
                    continue

                img_name = parts[0]
                model_name = ' '.join(parts[1:])
                
                # Map to family if possible, otherwise skip this image
                family_name = None
                for key, family in family_mapping.items():
                    if key in model_name:
                        family_name = family
                        break
                
                # Skip images that don't belong to our target families
                if family_name is None:
                    continue

                if family_name not in self.label_to_idx:
                    if not self.allow_new_labels:
                        raise ValueError(
                            "Unknown label in split file: {}".format(family_name)
                        )
                    self.label_to_idx[family_name] = label_idx
                    self.idx_to_label[label_idx] = family_name
                    label_idx += 1

                self.image_list.append(img_name)
                self.labels.append(self.label_to_idx[family_name])

                if limit is not None and len(self.image_list) >= limit:
                    break

        if self.verbose:
            print(
                "Loaded {} images with {} aircraft models".format(
                    len(self.image_list), len(self.label_to_idx)
                )
            )

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, idx):
        img_name = self.image_list[idx]
        img_path = self.image_dir / '{}.jpg'.format(img_name)

        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print("Error loading {}: {}".format(img_path, e))
            image = Image.new('RGB', (224, 224), color='white')

        if self.transform:
            image = self.transform(image)

        label = self.labels[idx]
        return image, label

    def get_model_name(self, idx):
        return self.idx_to_label[idx]


def get_fgvc_loaders(
    dataset_root,
    batch_size=32,
    num_workers=2,
    train_transform=None,
    val_transform=None,
    split_type='family',
    limit_per_split=None,
    pin_memory=None,
    verbose=True,
):
    """
    Load FGVC aircraft dataset and create train/val/test DataLoaders.

    Args:
        dataset_root: Path to FGVC dataset root (contains 'data' folder)
        batch_size: Batch size for DataLoaders
        num_workers: Number of workers for data loading
        train_transform: Transform pipeline for training data
        val_transform: Transform pipeline for validation/test data
        split_type: 'family', 'manufacturer', 'variant', or '' for base splits
        limit_per_split: Optional int to limit samples per split (for quick tests)
        pin_memory: Override pin_memory (default: torch.cuda.is_available())
        verbose: Print dataset stats

    Returns:
        Dictionary with loaders and metadata.
    """

    dataset_path = Path(dataset_root)
    data_dir = dataset_path / 'data'

    if not data_dir.exists():
        raise FileNotFoundError(
            "Data directory not found at: {}".format(data_dir)
        )

    split_prefix = 'images'
    if split_type in {'family', 'manufacturer', 'variant'}:
        split_prefix = 'images_{}'.format(split_type)

    train_file = data_dir / '{}_train.txt'.format(split_prefix)
    val_file = data_dir / '{}_val.txt'.format(split_prefix)
    test_file = data_dir / '{}_test.txt'.format(split_prefix)

    if train_transform is None:
        train_transform = transforms.Compose([
            transforms.RandomAffine(
                degrees=5, translate=(0.05, 0.05), scale=(0.98, 1.02)
            ),
            transforms.ToTensor(),
        ])

    if val_transform is None:
        val_transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    if pin_memory is None:
        pin_memory = torch.cuda.is_available()

    train_dataset = FGVCAircraftDataset(
        dataset_root,
        str(train_file),
        transform=train_transform,
        allow_new_labels=True,
        limit=limit_per_split,
        verbose=verbose,
    )

    val_dataset = FGVCAircraftDataset(
        dataset_root,
        str(val_file),
        transform=val_transform,
        label_to_idx=train_dataset.label_to_idx,
        idx_to_label=train_dataset.idx_to_label,
        allow_new_labels=False,
        limit=limit_per_split,
        verbose=verbose,
    )

    test_dataset = FGVCAircraftDataset(
        dataset_root,
        str(test_file),
        transform=val_transform,
        label_to_idx=train_dataset.label_to_idx,
        idx_to_label=train_dataset.idx_to_label,
        allow_new_labels=False,
        limit=limit_per_split,
        verbose=verbose,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    class_names = [train_dataset.idx_to_label[i] for i in range(len(train_dataset.idx_to_label))]

    return {
        'train': train_loader,
        'val': val_loader,
        'test': test_loader,
        'num_classes': len(train_dataset.label_to_idx),
        'idx_to_label': train_dataset.idx_to_label,
        'label_to_idx': train_dataset.label_to_idx,
        'class_names': class_names,
    }
