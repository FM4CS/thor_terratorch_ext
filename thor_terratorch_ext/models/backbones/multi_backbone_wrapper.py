"""Multi-backbone wrapper.

Builds a backbone composed of multiple other backbones and merges their features so the
resulting module can be used like a normal backbone by EncoderDecoderFactory.
"""

from __future__ import annotations

from typing import Literal

import torch
import torch.nn.functional as F  # noqa: N812
from terratorch.datasets import HLSBands, OpticalBands, SARBands
from terratorch.models.utils import _get_backbone, extract_prefix_keys

from torch import nn

try:
    from thor_terratorch_ext.datasets.utils import OLCIBands, SARThorBands, SLSTRBands
except Exception:  # pragma: no cover - optional bands
    OLCIBands = None
    SARThorBands = None
    SLSTRBands = None

import logging

from terratorch.models.necks import Neck, build_neck_list
from terratorch.registry import (
    TERRATORCH_BACKBONE_REGISTRY,
)

logger = logging.getLogger(__name__)


def _flatten_bands(bands):
    if bands is None:
        return None
    if isinstance(bands, dict):
        flattened = []
        for value in bands.values():
            if isinstance(value, list):
                flattened.extend(value)
            else:
                flattened.append(value)
        return flattened
    return list(bands)


def freeze_module(module: nn.Module):
    for param in module.parameters():
        param.requires_grad_(False)


@TERRATORCH_BACKBONE_REGISTRY.register
class MultiBackboneWrapper(nn.Module):
    """Backbone that combines multiple backbones into a single feature pyramid."""

    def __init__(
        self,
        backbones: list[dict],
        bands: list[
            HLSBands
            | OpticalBands
            | SARBands
            | SARThorBands
            | OLCIBands
            | SLSTRBands
            | int
        ]
        | None = None,
        necks: list[dict] | None = None,
        merge_features: Literal["concat", "sum", "mean", "flatten"] = "concat",
        rescale_features: Literal["down", "up"]
        | int
        | list[int]
        | tuple[int]
        | None = None,
        **kwargs,
    ) -> None:
        super().__init__()

        if hasattr(backbones, "items"):
            raise TypeError(
                "`backbones` must be a list/sequence of backbone configuration dictionaries."
            )

        backbone_entries = list(backbones)
        if not backbone_entries:
            raise ValueError("At least one backbone configuration must be provided.")

        bands = _flatten_bands(bands)

        default_entry_kwargs = dict(kwargs)
        if "merge_features" in default_entry_kwargs:
            merge_features = default_entry_kwargs.pop("merge_features")
        if "rescale_features" in default_entry_kwargs:
            rescale_features = default_entry_kwargs.pop("rescale_features")

        backbone_modules: list[nn.Module] = []
        backbone_necks: list[Neck | None] = []
        backbone_bands: list[list[int]] = []

        for entry in backbone_entries:
            logger.debug("=== Building backbone ===")
            if hasattr(entry, "items") or isinstance(entry, dict):
                if len(entry) == 1:
                    # Assume single-entry dict is {name: {...}}
                    key, value = list(entry.items())[0]
                    entry_cfg = dict(value)
                    entry_cfg["backbone"] = key
                else:
                    entry_cfg = dict(entry).copy()
            else:
                raise TypeError("Each backbone entry must be a mapping/dict.")

            for key, value in default_entry_kwargs.items():
                entry_cfg.setdefault(key, value)

            backbone_name = entry_cfg.pop("backbone", None) or entry_cfg.pop(
                "name", None
            )
            if backbone_name is None:
                raise ValueError(
                    "Each backbone entry must define `backbone` or `name`."
                )

            necks_cfg = entry_cfg.pop("necks", None)

            backbone_kwargs, remaining_kwargs = extract_prefix_keys(
                entry_cfg, "backbone_"
            )
            backbone_kwargs.update(remaining_kwargs)

            freeze_backbone = backbone_kwargs.pop("freeze_backbone", False)

            assert not (
                "model_bands" in backbone_kwargs and "bands" in backbone_kwargs
            ), "Backbone entry cannot both `bands` and `model_bands`."

            model_bands = backbone_kwargs.get("model_bands", None)

            if model_bands is None:
                model_bands = backbone_kwargs.pop("bands", None)

            if model_bands is None:
                model_bands = bands.copy()
                logger.warning(
                    f"did not find bands for backbone {backbone_name}. Bands set to global bands: {model_bands}"
                )

            model_bands = _flatten_bands(model_bands)
            logger.info(f"backbone {backbone_name} bands: {model_bands}")
            if model_bands is None:
                raise ValueError(
                    "Bands must be specified either globally (`bands`) or per-backbone "
                    "(`bands`, or `model_bands`)."
                )

            backbone_bands.append(model_bands)

            try:
                backbone = _get_backbone(
                    backbone_name,
                    **backbone_kwargs,
                )
            except KeyError as exc:
                raise KeyError(
                    f"Backbone {backbone_name} not found in the registry."
                ) from exc

            if necks_cfg is not None:
                neck_list, _ = build_neck_list(necks_cfg, backbone.out_channels)
                backbone_neck = nn.Sequential(*neck_list)
            else:
                backbone_neck = None

            backbone_modules.append(backbone)
            backbone_necks.append(backbone_neck)

            if freeze_backbone:
                freeze_module(backbone)
            logger.info(f"freeze_backbone: {freeze_backbone}")

        self.bands = bands
        self.backbones = nn.ModuleList(backbone_modules)
        self.backbone_bands = backbone_bands
        self.backbone_necks = nn.ModuleList(backbone_necks)

        def _neck_process_channels(ops: nn.Sequential, channel_list):
            cur_channel_list = channel_list.copy()
            if ops is None or isinstance(ops, nn.Identity):
                return cur_channel_list

            for cur_op in ops:
                cur_channel_list = cur_op.process_channel_list(cur_channel_list)
            return cur_channel_list

        processed_out_channels = [
            _neck_process_channels(neck, backbone.out_channels)
            for neck, backbone in zip(self.backbone_necks, self.backbones, strict=False)
        ]
        self.backbone_out_channels = processed_out_channels

        assert all(
            len(poc) == len(processed_out_channels[0]) for poc in processed_out_channels
        ), (
            "Necks must have the same number of output channels. "
            f"Got {[len(poc) for poc in processed_out_channels]}, {processed_out_channels}."
        )

        self.merge_features = merge_features
        if self.merge_features in ["concat", "sum", "mean"]:
            assert all(
                len(poc) == len(processed_out_channels[0])
                for poc in processed_out_channels
            ), (
                "Necks must have the same number of output channels. "
                f"Got {[len(poc) for poc in processed_out_channels]}, {processed_out_channels}."
            )

            backbone_out_indices = list(range(len(processed_out_channels[0])))
            self.out_indices = backbone_out_indices
        else:
            self.out_indices = None

        if self.merge_features in ["sum", "mean"]:
            for backbone_out_channels in self.backbone_out_channels:
                for other_backbone_out_channels in self.backbone_out_channels:
                    assert all(
                        boc == oboc
                        for boc, oboc in zip(
                            backbone_out_channels, other_backbone_out_channels
                        )
                    ), (
                        "When using 'sum' or 'mean' to merge features, "
                        "all backbones must have the same output channels. "
                        f"Got {self.backbone_out_channels}."
                    )

        assert rescale_features is None or isinstance(
            rescale_features, int | str | list
        ), "rescale_features must be None, int, str or list."
        if isinstance(rescale_features, str):
            assert rescale_features in ["down", "up"], (
                "rescale_features must be 'down' or 'up'."
            )
        if isinstance(rescale_features, int):
            assert rescale_features >= 0, "rescale_features must be a positive integer."
            assert rescale_features < len(self.backbones), (
                "rescale_features must be less than the number of backbones."
            )
        if isinstance(rescale_features, list | tuple) and self.out_indices is not None:
            assert len(rescale_features) == len(self.out_indices), (
                "rescale_features must have the same length as the number of output features."
            )
        self.rescale_features = rescale_features

        if necks is None:
            necks = []
        neck_list, channel_list = build_neck_list(necks, self.out_channels)
        self.out_necks = nn.Sequential(*neck_list)

    @property
    def out_channels(self):
        if self.merge_features == "concat":
            return [
                sum([bc[i] for bc in self.backbone_out_channels])
                for i in self.out_indices
            ]
        elif self.merge_features in ["sum", "mean"]:
            return self.backbone_out_channels[0]
        elif self.merge_features in ["flatten"]:
            out = []
            for b in range(len(self.backbones)):
                for i in range(len(self.backbone_out_channels[b])):
                    out.append(self.backbone_out_channels[b][i])
            return out
        else:
            raise ValueError(f"Unknown merge_features method: {self.merge_features}.")

    def _preprocess_input(self, x):
        xs = []
        for bands in self.backbone_bands:
            if bands is None:
                xs.append(x)
            else:
                band_indices = [self.bands.index(band) for band in bands]
                xs.append(x[:, band_indices])
        return xs

    def inner_forward(self, xs, indices=None):
        backbone_features = []
        for backbone, neck, x in zip(
            self.backbones, self.backbone_necks, xs, strict=False
        ):
            backbone_out = backbone(x)
            # TODO: remove once deprecated
            if hasattr(backbone, "prepare_features_for_image_model"):
                backbone_out = backbone.prepare_features_for_image_model(backbone_out)
            if neck is not None:
                backbone_out = neck(backbone_out)
            backbone_features.append(backbone_out)

        features = []
        if self.merge_features in ["concat", "sum", "mean"]:
            num_features = len(backbone_features[0])
            for i in range(num_features):
                shape = None
                if self.rescale_features == "up":
                    max_shape = max([bf[i].shape[-2:] for bf in backbone_features])
                    shape = max_shape
                elif self.rescale_features == "down":
                    min_shape = min([bf[i].shape[-2:] for bf in backbone_features])
                    shape = min_shape
                elif isinstance(self.rescale_features, list | tuple):
                    shape = self.rescale_features[i]
                elif isinstance(self.rescale_features, int):
                    shape = backbone_features[self.rescale_features][i].shape[-2:]

                if self.merge_features in ["sum", "mean"]:
                    if shape is not None:
                        features_reshaped = torch.stack(
                            [
                                F.interpolate(
                                    bf[i],
                                    size=shape,
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                if bf[i].shape[-2:] != shape
                                else bf[i]
                                for bf in backbone_features
                            ],
                            dim=0,
                        )
                    else:
                        features_reshaped = torch.stack(
                            [bf[i] for bf in backbone_features], dim=0
                        )

                    if self.merge_features == "sum":
                        feature = torch.sum(
                            features_reshaped,
                            dim=0,
                        )
                    elif self.merge_features == "mean":
                        feature = torch.mean(
                            features_reshaped,
                            dim=0,
                        )
                    features.append(feature)

                elif self.merge_features == "concat":
                    if shape is not None:
                        feature = torch.cat(
                            [
                                F.interpolate(
                                    bf[i],
                                    size=shape,
                                    mode="bilinear",
                                    align_corners=False,
                                )
                                if bf[i].shape[-2:] != shape
                                else bf[i]
                                for bf in backbone_features
                            ],
                            dim=1,
                        )
                    else:
                        feature = torch.cat([bf[i] for bf in backbone_features], dim=1)

                    features.append(feature)

        elif self.merge_features == "flatten":
            for bf in backbone_features:
                features.extend(bf)

        if indices is not None:
            features = [features[i] for i in indices]

        # for i in range(len(features)):
        #     logger.debug(f"MultiBackboneWrapper feature {i} shape: {features[i].shape}")

        return features

    def forward(self, x, **kwargs):
        xs = self._preprocess_input(x)
        x = self.inner_forward(xs, indices=self.out_indices)
        x = self.out_necks(x)
        return x

    def summary(self):
        print(self)
