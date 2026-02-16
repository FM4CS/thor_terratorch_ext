# Copyright (c) Norwegian Computing Center (NR)
# Based on Apache-2.0 and/or MIT-licensed files from the Terratorch project.

from __future__ import annotations

import albumentations as A


class CropNonEmptyMaskIfExistsCompat(A.CropNonEmptyMaskIfExists):
    """Version-tolerant wrapper for CropNonEmptyMaskIfExists.

    Some albumentations versions use ``height``/``width`` while others use
    ``size=(height, width)``. This wrapper keeps a stable constructor for
    config parsing and forwards to whichever base signature is available.
    """

    def __init__(
        self,
        height: int,
        width: int,
        ignore_values: list[int] | None = None,
        ignore_channels: list[int] | None = None,
        p: float = 1.0,
    ) -> None:
        try:
            super().__init__(
                height=height,
                width=width,
                ignore_values=ignore_values,
                ignore_channels=ignore_channels,
                p=p,
            )
        except TypeError:
            super().__init__(
                size=(height, width),
                ignore_values=ignore_values,
                ignore_channels=ignore_channels,
                p=p,
            )
