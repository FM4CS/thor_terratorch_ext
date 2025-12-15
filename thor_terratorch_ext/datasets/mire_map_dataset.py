import glob
import logging
import os
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path
from typing import Any, Literal

import albumentations as A
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

logger = logging.getLogger(__name__)

# logger.setLevel(logging.DEBUG)


@lru_cache(maxsize=256)
def load_netcdf(
    f: str,
    label_var: str,
    bands: tuple[str, ...],
    nan_replace: int | float | str | None = None,
    label_replace: int | None = 255,
    label_mapping: tuple[tuple[int, int], ...] | None = None,
    ignore_classes: tuple[int, ...] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(f, decode_coords="all") as ds:
        im = np.stack(
            [ds[k].transpose("y", "x").to_numpy().astype("float32") for k in bands],
            axis=-1,
        )
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

        lbl = ds[label_var].fillna(label_replace).astype("long")
        lbl = lbl.transpose("y", "x").to_numpy()
        lbl[mask.any(axis=-1)] = label_replace

        # logger.debug(f"unique labels before ignore mapping {np.unique(lbl)}")
        # logger.debug(
        #     f"ignore classes: {ignore_classes}, label_replace: {label_replace}"
        # )
        if ignore_classes is not None:
            for k in ignore_classes:
                # logger.debug(f"Ignoring class {k}, mapping to {label_replace}")
                lbl[lbl == k] = label_replace

        # logger.debug(f"unique labels before mapping {np.unique(lbl)}")
        # logger.debug(f"label mapping: {label_mapping}")
        if label_mapping is not None:
            for k, v in label_mapping:
                # logger.debug(f"Mapping class {k} to {v}")
                lbl[lbl == k] = v
        # logger.debug(f"unique labels {np.unique(lbl)}")

    return im, lbl


class MireMapDataset(NonGeoDataset):
    TEST_TILES = ["32VKL", "32VPR", "33WXT"]
    VAL_TILES = ["32WPA", "33WVQ", "33WVS"]

    band_mapping = {  # noqa: RUF012
        "B01": "COASTAL_AEROSOL",
        "B02": "BLUE",
        "B03": "GREEN",
        "B04": "RED",
        "B05": "RED_EDGE_1",
        "B06": "RED_EDGE_2",
        "B07": "RED_EDGE_3",
        "B08": "NIR_BROAD",
        "B8A": "NIR_NARROW",
        "B09": "WATER_VAPOR",
        "B11": "SWIR_1",
        "B12": "SWIR_2",
    }

    all_band_names = tuple(band_mapping.values())

    CLASS_NAMES = {  # noqa: RUF012
        10: "Bebygd areal",
        20: "Dyrka mark",
        26: "Dyrka mark med myr",
        30: "Skog",
        36: "Skog med myr",
        50: "Snaumark",
        56: "Snaumark med myr",
        60: "Myr",
        61: "Myr med bebygd areal",  # "Myr med bebygd areal",
        62: "Myr med dyrka mark",  # "Myr med dyrka mark",
        63: "Myr med skog",  # "Myr med skog",
        # 64: "Blautmyr",  # "Myr med snaumark",
        65: "Myr med snaumark",  # "Kombimyr",
        66: "Kombimyr",
        80: "Vann",
        255: "Ignore",
    }

    LABEL_COLORMAP = {  # noqa: RUF012
        10: (0x66, 0x66, 0x66),  # Bebygd areal
        20: (0xFF, 0xEE, 0x00),  # Dyrka mark og beitevoller
        26: (0xBB, 0xAA, 0x00),  # Dyrka mark og beitevoller med innslag av myr
        30: (0x00, 0x88, 0x00),  # Skog
        36: (0x55, 0x88, 0x00),  # Skog med innslag av myr
        50: (0x00, 0xCC, 0x55),  # Snaumark
        56: (0x66, 0xAA, 0x00),  # Snaumark med innslag av myr
        60: (0x77, 0x77, 0x00),  # Myr
        61: (0x7A, 0x7A, 0x44),  # Myr med innslag av bebygd areal
        62: (0xAA, 0x99, 0x11),  # Myr med innslag av dyrka mark og beitevoller
        63: (0x77, 0x88, 0x00),  # Myr med innslag av skog
        65: (0x88, 0xAA, 0x00),  # Myr med innslag av snaumark
        66: (0x77, 0x77, 0x44),  # Kombinasjon av ulike myrtyper
        80: (0x11, 0x00, 0xFF),  # Vann
        255: (0x00, 0x00, 0x00),  # Ignore
    }

    IGNORE_CLASSES = (
        26,  # mixed class
        36,  # mixed class
        56,  # mixed class
        61,  # mixed class, merge with 60?
        62,  # mixed class, merge with 60?
        63,  # mixed class, merge with 60?
        65,  # mixed class, merge with 60?
        66,  # mixed class, merge with 60?
    )

    # TODO: implement multi label mode
    # MAJOR_RATIO = 0.7
    # CLASS_PROBS = {
    #     10: {10: 1.0},
    #     20: {20: 1.0},
    #     26: {20: MAJOR_RATIO, 60: 1.0 - MAJOR_RATIO},
    #     30: {30: 1.0},
    #     36: {30: MAJOR_RATIO, 60: 1.0 - MAJOR_RATIO},
    #     50: {50: 1.0},
    #     56: {50: MAJOR_RATIO, 60: 1.0 - MAJOR_RATIO},
    #     60: {60: 1.0},
    #     61: {60: MAJOR_RATIO, 10: 1.0 - MAJOR_RATIO},
    #     62: {60: MAJOR_RATIO, 20: 1.0 - MAJOR_RATIO},
    #     63: {60: MAJOR_RATIO, 30: 1.0 - MAJOR_RATIO},
    #     65: {60: MAJOR_RATIO, 50: 1.0 - MAJOR_RATIO},
    #     # 66: {66: 1.0}, # TODO: or ignore?
    #     80: {80: 1.0},
    # }

    rgb_bands = ("RED", "GREEN", "BLUE")

    BAND_SETS = {"all": all_band_names, "rgb": rgb_bands}  # noqa: RUF012

    label_var = "mire"

    def __init__(
        self,
        data_root: str,
        split="train",
        bands: Sequence[str] = BAND_SETS["all"],
        mode: Literal["binary", "multiclass"] = "multiclass",
        transform: A.Compose | None = None,
        constant_scale: float = 1.0,
        no_data_replace: float | str | None = 0,
        no_label_replace: int | None = -1,
    ):
        super().__init__()

        validate_bands(bands, self.all_band_names)
        self.bands = bands
        reversed_band_mapping = {v: k for k, v in self.band_mapping.items()}
        self.band_netcdf_names = [reversed_band_mapping[b] for b in bands]
        self.constant_scale = constant_scale
        self.data_root = Path(data_root)
        self.split = split
        if split not in ["train", "test", "val", "predict"]:
            msg = "Split must be one of train, test, val."
            raise Exception(msg)

        self.data_root = Path(data_root)

        self.rgb_indices = [self.bands.index(b) for b in self.rgb_bands]

        img_files = glob.glob(str(self.data_root / "**/*.nc"), recursive=True)
        img_files = [f for f in img_files if not os.path.isdir(f)]
        if self.split == "val":
            # filter on validation tiles
            img_files = [f for f in img_files if self._filter_func(f, self.VAL_TILES)]
        elif self.split == "test":
            # filter on test tiles
            img_files = [f for f in img_files if self._filter_func(f, self.TEST_TILES)]
        elif self.split == "train":
            img_files = [
                f
                for f in img_files
                if not self._filter_func(f, self.VAL_TILES + self.TEST_TILES)
            ]

        self.files = sorted(img_files)

        self.mode = mode
        if mode == "multiclass":
            labels_to_use = sorted(
                set(self.LABEL_COLORMAP.keys()) - set(self.IGNORE_CLASSES)
            )
            self.label_mapping = tuple([
                (key, index) if key != 255 else (key, 255)
                for index, key in enumerate(labels_to_use)
            ])
            self.num_classes = len(labels_to_use) - 1
        elif mode == "binary":
            self.label_mapping = (
                (10, 0),  # Bebygd areal
                (20, 0),  # Dyrka mark
                (30, 0),  # Skog
                (50, 0),  # Snaumark
                (60, 1),  # Myr
                (80, 0),  # Vann
                (255, 255),  # Ignore
            )
            self.num_classes = 2

        self.no_data_replace = no_data_replace
        self.no_label_replace = no_label_replace

        self.transform = transform if transform else default_transform

    def _filter_func(self, image_file, split):
        tile = self._get_tile(image_file)

        if tile in split:
            return True
        else:
            return False

    def _get_tile(self, file_path):
        # find the tile name from the file path
        path = Path(file_path)
        tile = path.parents[0].name.split("_")[-2].removeprefix("T")
        return tile

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, index: int) -> dict[str, Any]:
        im, lbl = load_netcdf(
            self.files[index],
            self.label_var,
            tuple(self.band_netcdf_names),
            self.no_data_replace,
            self.no_label_replace,
            label_mapping=self.label_mapping,
            ignore_classes=tuple(self.IGNORE_CLASSES),
        )

        output = {
            "image": im.astype(np.float32),
            "mask": lbl,
        }
        if self.transform:
            output = self.transform(**output)  # type: ignore

        # print unique labels
        # logger.debug(
        #     f"unique labels after transform {np.unique(output['mask'].numpy())}"
        # )

        output["mask"] = output["mask"].long()
        return output

    def plot(self, sample: dict[str, Tensor], suptitle: str | None = None) -> Figure:
        """Plot a sample from the dataset.

        Args:
            sample: a sample returned by :meth:`__getitem__`
            suptitle: optional string to use as a suptitle

        Returns:
            a matplotlib Figure with the rendered sample
        """
        num_images = 4

        rgb_indices = [self.bands.index(band) for band in self.rgb_bands]
        if len(rgb_indices) != 3:
            msg = "Dataset doesn't contain some of the RGB bands"
            raise ValueError(msg)

        # RGB -> channels-last
        image = sample["image"][rgb_indices, ...].permute(1, 2, 0).numpy()
        mask = sample["mask"].numpy()
        mask = np.squeeze(mask)

        image = clip_image_percentile(image, 0.5, 99.5)

        if "prediction" in sample:
            prediction = sample["prediction"].numpy()
            prediction = np.squeeze(prediction)
            num_images += 1
        else:
            prediction = None

        fig, ax = plt.subplots(1, num_images, figsize=(12, 5), layout="compressed")

        ax[0].axis("off")

        if self.mode == "binary":
            mire_colormap = {0: (139, 69, 19), 1: (0, 255, 0), 255: (0, 0, 0)}
            class_names = ["Non-mire", "Mire", "Ignore"]
        else:
            mire_colormap = {i: self.LABEL_COLORMAP[k] for (k, i) in self.label_mapping}
            class_names = [
                self.CLASS_NAMES[class_key]
                for (class_key, class_ind) in self.label_mapping
            ]
        logger.debug(f"mire color map: {mire_colormap}")
        logger.debug(f"class names: {class_names}")
        logger.debug(f"unique labels plot {np.unique(mask)}")
        mask_display = mask.copy()
        mask_display[mask == 255] = self.num_classes
        if prediction is not None:
            prediction_display = prediction.copy()
            prediction_display[prediction == 255] = self.num_classes
        else:
            prediction_display = None
        logger.debug(f"unique labels plot after mapping {np.unique(mask_display)}")
        cmap = colors.ListedColormap([
            np.array(color) / 255.0 for color in mire_colormap.values()
        ])
        norm = colors.Normalize(vmin=0, vmax=len(mire_colormap) - 1)

        ax[1].axis("off")
        ax[1].title.set_text("Image")
        ax[1].imshow(image)

        ax[2].axis("off")
        ax[2].title.set_text("Ground Truth Mask")
        ax[2].imshow(mask_display, cmap=cmap, norm=norm)

        ax[3].axis("off")
        ax[3].title.set_text("GT Mask on Image")
        ax[3].imshow(image)
        ax[3].imshow(mask_display, cmap=cmap, alpha=0.3, norm=norm)

        if prediction_display is not None:
            ax[4].axis("off")
            ax[4].title.set_text("Predicted Mask")
            ax[4].imshow(prediction_display, cmap=cmap, norm=norm)
        legend_data = [
            [i, cmap(norm(i)), class_names[i]] for i in range(len(class_names))
        ]
        handles = [
            Rectangle((0, 0), 1, 1, color=tuple(v for v in c))
            for k, c, n in legend_data
        ]
        labels = [n for k, c, n in legend_data]
        ax[0].legend(handles, labels, loc="center")
        if suptitle is not None:
            plt.suptitle(suptitle)

        return fig
