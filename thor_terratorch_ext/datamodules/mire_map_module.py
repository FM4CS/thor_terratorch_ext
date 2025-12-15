from collections.abc import Sequence
from typing import Any, Literal

import albumentations as A
import kornia.augmentation as K  # noqa: N812
from kornia.augmentation import AugmentationSequential

from terratorch.datamodules.utils import wrap_in_compose_is_list
from thor_terratorch_ext.datasets import (
    MireMapDataset,
)
from terratorch.io.file import load_from_file_or_attribute
from torch import Tensor
from torch.utils.data import DataLoader
from torchgeo.datamodules import NonGeoDataModule


class MireMapNonGeoDataModule(NonGeoDataModule):
    def __init__(
        self,
        data_root: str,
        batch_size: int,
        num_workers: int,
        means: list[float] | str,
        stds: list[float] | str,
        bands: Sequence[str] = MireMapDataset.all_band_names,
        mode: Literal["multiclass", "binary"] = "multiclass",
        train_transform: A.Compose | None | list[A.BasicTransform] = None,
        val_transform: A.Compose | None | list[A.BasicTransform] = None,
        test_transform: A.Compose | None | list[A.BasicTransform] = None,
        predict_transform: A.Compose | None | list[A.BasicTransform] = None,
        drop_last: bool = True,
        constant_scale: float = 1.0,
        no_data_replace: str | float | None = 0,
        no_label_replace: int | None = -1,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            MireMapDataset, batch_size=batch_size, num_workers=num_workers, **kwargs
        )

        self.data_root = data_root

        self.bands = bands

        self.drop_last = drop_last
        self.constant_scale = constant_scale
        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace
        self.mode = mode

        means = load_from_file_or_attribute(means)
        stds = load_from_file_or_attribute(stds)

        self.aug = AugmentationSequential(K.Normalize(means, stds), data_keys=None)
        # self.aug = Normalize(means, stds)

        self.train_transform = wrap_in_compose_is_list(train_transform)
        self.val_transform = wrap_in_compose_is_list(val_transform)
        self.test_transform = wrap_in_compose_is_list(test_transform)
        self.predict_transform = wrap_in_compose_is_list(predict_transform)

    def setup(self, stage: str):
        if stage in ["fit"]:
            self.train_dataset = MireMapDataset(
                self.data_root,
                split="train",
                bands=self.bands,
                mode=self.mode,
                transform=self.train_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )
        if stage in ["fit", "validate"]:
            self.val_dataset = MireMapDataset(
                self.data_root,
                split="val",
                bands=self.bands,
                mode=self.mode,
                transform=self.val_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

        if stage in ["test"]:
            self.test_dataset = MireMapDataset(
                self.data_root,
                split="test",
                bands=self.bands,
                mode=self.mode,
                transform=self.test_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

        if stage == "predict":
            self.predict_dataset = MireMapDataset(
                self.data_root,
                split="val",
                bands=self.bands,
                mode=self.mode,
                transform=self.predict_transform,
                constant_scale=self.constant_scale,
                no_data_replace=self.no_data_replace,
                no_label_replace=self.no_label_replace,
            )

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
            persistent_workers=False,
            collate_fn=self.collate_fn,
            # drop_last=split == "train" and self.drop_last,
            drop_last=self.drop_last,
        )
