from glob import glob
from pathlib import Path
from typing import Optional

import numpy as np
from PIL import Image
from albumentations import Compose, HorizontalFlip, VerticalFlip, Normalize
from albumentations.pytorch import ToTensorV2
from pytorch_lightning import LightningDataModule
from torch import Tensor
from torch.utils.data import DataLoader, Dataset


class SegmentationDataset(Dataset):
    def __init__(
            self,
            dataset_path_regex: str,
            labels_base_path: str,
            transforms: Compose,
    ) -> None:
        # primitives attributes
        self.labels_base_path = Path(labels_base_path)
        self.transforms = transforms
        self.image_files = [Path(path) for path in glob(dataset_path_regex)]
        # prune unlabeled files
        self.image_files = [
            path
            for path in self.image_files
            if (
                (self.labels_base_path / path.name)
                .with_suffix('.ppm')
                .is_file()
            )
        ]

    def __len__(self) -> int:
        return len(self.image_files)

    def __getitem__(self, index: int) -> tuple[Tensor]:
        image_path = self.image_files[index]
        label_path = (
            (self.labels_base_path / image_path.name)
            .with_suffix('.ppm')
        )
        # load images
        image = np.array(Image.open(image_path).convert('RGB'))
        label = np.array(Image.open(label_path).convert('RGB'))
        # augment (if needed) and transform to Tensor
        augmented = self.transforms(image=image, mask=label)
        image = augmented['image']
        label = augmented['mask']
        return image, label


class SegmentationDataModule(LightningDataModule):
    def __init__(
            self,
            dataset_path_regex: str,
            labels_base_path: str,
            batch_size: int,
            num_workers: int,
            train_transforms: Compose,
            val_transforms: Compose,
    ) -> None:
        super().__init__()
        self.dataset_path_regex = dataset_path_regex
        self.labels_base_path = labels_base_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.train_transforms = train_transforms
        self.val_transforms = val_transforms

    def setup(self, stage: Optional[str] = None) -> None:
        self.train_dataset = SegmentationDataset(
            self.dataset_path_regex,
            self.labels_base_path,
            transforms=self.train_transforms,
        )
        self.val_dataset = SegmentationDataset(
            self.dataset_path_regex,
            self.labels_base_path,
            transforms=self.val_transforms,
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
        )

train_transforms = Compose(
    [
        HorizontalFlip(),
        VerticalFlip(),
        Normalize(),
        ToTensorV2(),
    ]
)

val_transforms = Compose(
    [
        Normalize(),
        ToTensorV2(),
    ]
)

dm = SegmentationDataModule(
    dataset_path_regex='data/lfw-funneled/lfw_funneled/*/*.jpg',
    labels_base_path='data/parts_lfw_funneled_gt_images',
    batch_size=32,
    num_workers=8,
    train_transforms=train_transforms,
    val_transforms=val_transforms,
)
dm.setup('train')
train_dl = dm.train_dataloader()
for x, target in train_dl:
    break
