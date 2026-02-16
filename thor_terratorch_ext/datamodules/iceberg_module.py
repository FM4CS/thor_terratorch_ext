# Copyright (c) Norwegian Computing Center (NR)
# Based on Apache-2.0 and/or MIT-licensed files from the Terratorch project.

from collections.abc import Sequence
from typing import Any

import albumentations as A
import kornia.augmentation as K  # noqa: N812
import numpy as np
import torch
from kornia.augmentation import AugmentationSequential
from terratorch.datamodules.utils import wrap_in_compose_is_list
from terratorch.io.file import load_from_file_or_attribute
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule

from thor_terratorch_ext.datasets import (
    IcebergDataset,
)


def to_numpy(d):
    new_dict = {}
    for k, v in d.items():
        if not isinstance(v, torch.Tensor):
            new_dict[k] = v
        else:
            v = v.numpy()
            if k == "image":
                if len(v.shape) == 5:
                    raise NotImplementedError("Temporal axis not implemented")
                    # v = np.moveaxis(v, 1, -1)
                elif len(v.shape) == 4:
                    v = np.moveaxis(v, 1, -1)
                elif len(v.shape) == 3:
                    v = np.moveaxis(v, 0, -1)
            new_dict[k] = v
    return new_dict


class IcebergDataModule(NonGeoDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        means: list[float] | str,
        stds: list[float] | str,
        bands: Sequence[str] = IcebergDataset.all_band_names,
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        predict_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        no_data_replace: str | float | None = 0,
        no_label_replace: int | None = -1,
        use_metadata: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            IcebergDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.data_root = data_root

        self.bands = bands

        self.drop_last = drop_last
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.use_metadata = use_metadata

        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        self.aug = AugmentationSequential(
            K.Normalize(means, stds),
            data_keys=None,
        )

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)

    def setup(self, stage: str):
        if stage in ["fit"]:
            self.train_dataset = IcebergDataset(
                self.data_root,
                split="train",
                bands=self.bands,
                transform=self.train_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = IcebergDataset(
                self.data_root,
                split="val",
                bands=self.bands,
                transform=self.val_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

        if stage in ["test"]:
            self.test_dataset = IcebergDataset(
                self.data_root,
                split="test",
                bands=self.bands,
                transform=self.test_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

        if stage == "predict":
            self.predict_dataset = IcebergDataset(
                self.data_root,
                split="predict",
                bands=self.bands,
                transform=self.predict_transform,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

    def on_after_batch_transfer(
        self, batch: dict[str, Tensor], dataloader_idx: int
    ) -> dict[str, Tensor]:
        # Temporarily remove keys that kornia would misinterpret as
        # geometric data (bboxes, class_labels, etc.) so that
        # AugmentationSequential only processes image/mask tensors.
        # This is fine as long as the transforms don't need to modify those keys, which is true for our current setup.
        # We only apply mean and std normalization in AugmentationSequential, which doesn't need to modify bboxes or class labels.
        extra_keys = {}
        for key in list(batch.keys()):
            if key not in ("image", "mask"):
                extra_keys[key] = batch.pop(key)

        batch = super().on_after_batch_transfer(batch, dataloader_idx)

        batch.update(extra_keys)
        return batch

    def _collate_fn(self, batch):
        """
        batch: A list of dicts, e.g. [{'image': img1, 'bboxes': box1}, {'image': img2, ...}]
        """

        # Stack images (they are fixed size after transform)
        images = torch.stack([item["image"] for item in batch])
        masks = torch.stack([item["mask"] for item in batch])

        # Keep bboxes and labels as lists (because they vary in size).
        bboxes = []
        class_labels = []
        for item in batch:
            # Convert from COCO [x, y, w, h] to torchvision [x1, y1, x2, y2].
            boxes_coco = torch.tensor(item["bboxes"], dtype=torch.float32)
            if boxes_coco.numel() > 0:
                boxes_xyxy = boxes_coco.clone()
                boxes_xyxy[:, 2] = boxes_coco[:, 0] + boxes_coco[:, 2]  # x2 = x + w
                boxes_xyxy[:, 3] = boxes_coco[:, 1] + boxes_coco[:, 3]  # y2 = y + h
                bboxes.append(boxes_xyxy)
            else:
                bboxes.append(torch.zeros((0, 4), dtype=torch.float32))
            class_labels.append(torch.tensor(item["class_labels"], dtype=torch.int64))
        filenames = [item["filename"] for item in batch]
        source_files = [item.get("source_file", None) for item in batch]

        return {
            "image": images,  # Shape: (Batch_Size, Channels, H, W)
            "mask": masks,  # Shape: (Batch_Size, H, W)
            "bboxes": bboxes,  # List of N tensors, each shape (Num_Boxes, 4)
            "class_labels": class_labels,  # List of N tensors
            "filenames": filenames,  # List of N training images
            "source_file": source_files,  # List of SAR image source files
        }

    def _dataloader_factory(self, split: str) -> DataLoader[dict[str, Tensor]]:
        """Implement one or more PyTorch DataLoaders.

        Args:
            split: Either 'train', 'val', 'test', or 'predict'.

        Returns:
            A collection of data loaders specifying samples.

        Raises:
            MisconfigurationException: If :meth:`setup` does not define a
                dataset or sampler, or if the dataset or sampler has length 0.
        """
        dataset = self._valid_attribute(f"{split}_dataset", "dataset")
        batch_size = self._valid_attribute(f"{split}_batch_size", "batch_size")

        return DataLoader(
            dataset=dataset,
            batch_size=batch_size,
            shuffle=split == "train",
            num_workers=self.num_workers,
            collate_fn=self._collate_fn,
            drop_last=self.drop_last,
        )
