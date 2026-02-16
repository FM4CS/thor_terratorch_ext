# Copyright (c) Norwegian Computing Center (NR)
# Based on Apache-2.0 and/or MIT-licensed files from the Terratorch project.

import glob
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import albumentations as A
import h5py
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from matplotlib import colors
from matplotlib.figure import Figure
from matplotlib.patches import Rectangle
from terratorch.datasets.utils import (
    clip_image_percentile,
    default_transform,
    validate_bands,
)
from torch import Tensor
from torchgeo.datasets import NonGeoDataset


class IcebergDataset(NonGeoDataset):
    all_band_names = (
        "IW_VV",
        "IW_VH",
    )
    band_mapping = {  # noqa: RUF012
        "B01": "IW_VV",
        "B02": "IW_VH",
    }

    rgb_bands = ("IW_VV", "IW_VV", "IW_VH")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}  # noqa: RUF012

    def __init__(
        self,
        data_root: str,
        split="train",
        bands: Sequence[str] = BAND_SETS["all"],
        transform: A.Compose | None = None,
        labelvar: str = "mask",  # "mask_no_ignore_boundary",
        no_data_replace: float | str | None = 0,
        no_label_replace: int | None = -1,
    ):
        super().__init__()

        if split not in ["train", "test", "val", "predict"]:
            msg = "Split must be one of train, test, val, predict."
            raise Exception(msg)
        self.split = split

        validate_bands(bands, self.all_band_names)
        self.bands = bands

        # reversed_band_mapping = {v: k for k, v in self.band_mapping.items()}
        # self.band_netcdf_names = [reversed_band_mapping[b] for b in bands]
        self.data_root = Path(data_root)
        self.labelvar = labelvar
        # self.valid_mask = ["valid_mask"] #["s3_valid_mask"]  # TODO read from config file
        self.target_bands = ["mask"]  # ["fsc"] # ["sgs"] # ["ssw"]

        self.data_root = Path(data_root)

        # self.rgb_indices = [self.all_band_names.index(b) for b in self.rgb_bands]
        if len(self.bands) >= 3:
            self.rgb_indices = [0, 1, 2]
        else:
            self.rgb_indices = [0, 1, 1]
        all_files = glob.glob(str(self.data_root / "**/*.nc"), recursive=True)

        if self.split == "train":
            ann_file = self.data_root / "annotations_train.json"
        elif self.split == "val":
            ann_file = self.data_root / "annotations_val.json"
        elif self.split == "test":
            ann_file = self.data_root / "annotations_test.json"
        else:
            ann_file = None

        if ann_file and ann_file.exists():
            with open(ann_file, "r") as f:
                coco_data = json.load(f)
            # Use 'file_name' from COCO to build the absolute paths
            img_files = [
                str(self.data_root / "all_files" / img["file_name"])
                for img in coco_data["images"]
            ]
        elif ann_file is None and self.split == "predict":
            print("No annotation files for 'predict' split, using all files.")
            img_files = all_files
        else:
            print("No annotation files found, splitting data randomly.")
            all_files = glob.glob(str(self.data_root / "**/*.nc"), recursive=True)
            np.random.seed(42)
            np.random.shuffle(all_files)

            val_fraction = 0.1
            test_fraction = 0.25
            num_test = int(len(all_files) * test_fraction)
            num_val = int(len(all_files) * val_fraction)

            if self.split == "val":
                img_files = all_files[:num_val]
            elif self.split == "test":
                img_files = all_files[num_val : num_val + num_test]
            else:
                # This covers 'train' (if no JSON) and 'predict'
                img_files = all_files[num_val + num_test :]

        self.files = sorted(img_files)

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        self.transform = transform if transform else default_transform

        bbox_params = A.BboxParams(
            format="coco", min_visibility=0.1, label_fields=["class_labels"]
        )
        self.transform = A.Compose(
            transforms=self.transform.transforms, bbox_params=bbox_params
        )

        self.shape = (512, 512)  # TODO: read from config or data

    def _load_netcdf(
        self,
        f: str | Path,
        labelvar: str,
        # bands: list[str],
        nan_replace: int | float | str | None = None,
        label_replace: int | None = -1,
        # valid_mask: str | None = None,
    ) -> tuple[np.ndarray, np.ndarray]:

        # with xr.open_dataset(f, decode_coords="all", engine="h5netcdf") as ds:
        #     # im = np.stack(
        #     #     [ds[k].transpose("y", "x").to_numpy().astype("float32") for k in bands],
        #     #     axis=-1,
        #     # )
        #     im = ds["image"].transpose("y", "x", ...).to_numpy()
        with h5py.File(f, "r") as hf:
            im_ds = hf["image"]
            im = im_ds[:]
            # Assuming (C, H, W) layout in file, transpose to (H, W, C)
            if im.ndim == 3 and im.shape[0] < im.shape[1] and im.shape[0] < im.shape[2]:
                im = np.moveaxis(im, 0, -1)

            # Handle FillValue -> NaN for float images to match xarray behavior
            if "_FillValue" in im_ds.attrs:
                fill_val = im_ds.attrs["_FillValue"]
                if np.issubdtype(im.dtype, np.floating):
                    im[im == fill_val] = np.nan

            mask = np.isnan(im)
            if nan_replace is not None:
                if isinstance(nan_replace, str):
                    if nan_replace == "mean":
                        nan_replace = float(np.nanmean(im))
                    elif nan_replace == "median":
                        nan_replace = float(np.nanmedian(im))
                    else:
                        err_msg = f"Unknown nan_replace value: {nan_replace}"
                        raise ValueError(err_msg)

                im[mask] = nan_replace

            # lbl = ds[labelvar].fillna(label_replace).astype("long")
            # lbl = lbl.transpose("y", "x").to_numpy()
            lbl_ds = hf[labelvar]
            lbl = lbl_ds[:]

            # Handle dimensions for label
            if lbl.ndim == 3 and lbl.shape[0] == 1:
                lbl = np.moveaxis(lbl, 0, -1)

            # Handle FillValue for label
            if "_FillValue" in lbl_ds.attrs:
                fill_val = lbl_ds.attrs["_FillValue"]
                # if uint convert to int larger than uint max to avoid overflow when replacing
                if np.issubdtype(lbl.dtype, np.unsignedinteger):
                    lbl = lbl.astype(np.int64)
                lbl[lbl == fill_val] = label_replace
            elif np.issubdtype(lbl.dtype, np.floating):
                lbl[np.isnan(lbl)] = label_replace

            lbl = lbl.astype("long")
            lbl[mask.any(axis=-1)] = label_replace

            # Get the bouding box, if any
            # bbox_str = ds.attrs["bbox"]
            bbox_str = hf.attrs["bbox"]
            if isinstance(bbox_str, bytes):
                bbox_str = bbox_str.decode("utf-8")
            bboxes = json.loads(bbox_str)
            # source_file = ds.attrs["source_file"]

            source_file = hf.attrs["source_file"]
            if isinstance(source_file, bytes):
                source_file = source_file.decode("utf-8")

        return im, lbl, bboxes, source_file

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        f = self.files[idx]

        im, lbl, bboxes, source_file = self._load_netcdf(
            f,
            self.labelvar,
            # self.band_netcdf_names,
            self.no_data_replace,
            self.no_label_replace,
        )

        im = im.copy()
        lbl = lbl.copy()
        bboxes = bboxes.copy()
        class_labels = []
        for bbox in bboxes:
            class_labels.append(1)

        output = {
            "image": im,
            "mask": lbl,
            "bboxes": bboxes,
            "class_labels": class_labels,
            # "filename": f,
        }

        if self.transform:
            output = self.transform(**output)  # type: ignore

        # if len(bboxes) > 0:
        #     target_boxes = torch.tensor(bboxes, dtype=torch.float32)
        #     target_labels = torch.tensor(class_labels, dtype=torch.long)
        # else:
        #     # Handle images with no boxes
        #     target_boxes = torch.zeros((0, 4), dtype=torch.float32)
        #     target_labels = torch.zeros((0,), dtype=torch.long)

        output["filename"] = str(f)
        output["source_file"] = source_file

        return output

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """

        image = sample["image"][self.rgb_indices, ...].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()

        image = clip_image_percentile(image)

        return self._plot_sample(
            image,
            mask,
            prediction=sample.get("prediction", None),
            suptitle=suptitle,
        )

    @staticmethod
    def _plot_sample(image, label, prediction=None, suptitle=None):
        num_images = 5 if prediction is not None else 4
        fig, ax = plt.subplots(1, num_images, figsize=(8, 6))

        # for legend
        ax[0].axis("off")
        norm = colors.Normalize(vmin=0, vmax=255)
        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(label, cmap="jet", norm=norm)
        # ax[2].imshow(label, cmap=cmap)  # , norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(label, cmap="jet", alpha=0.3, norm=norm)
        # ax[3].imshow(label, cmap=cmap, alpha=0.3)  # , norm=norm)

        if prediction is not None:
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction, cmap="jet", norm=norm)
            # ax[4].imshow(prediction, cmap=cmap)  # , norm=norm)

        cmap = plt.get_cmap("jet")
        # legend_data = []
        # for i, _ in enumerate(range(0, num_classes + 1)):
        #     if i < num_classes:
        #         class_name = class_names[i] if class_names else str(i)
        #     else:
        #         class_name = "Ignore"
        #     # class_name = class_names[i] if class_names and i < len(class_names) else str(i)
        #     data = [i, cmap(norm(i)), class_name]
        #     # data = [i, cmap(i), class_name]
        #     legend_data.append(data)
        # handles = [Rectangle((0, 0), 1, 1, color=tuple(v for v in c)) for k, c, n in legend_data]
        # labels = [n for k, c, n in legend_data]
        # ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)
        return fig
