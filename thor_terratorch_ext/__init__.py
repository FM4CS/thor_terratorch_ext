import jsonargparse._typehints as _jth

_original_adapt = _jth.adapt_typehints

def _patched_adapt(val, tp, *args, **kwargs):
    try:
        return _original_adapt(val, tp, *args, **kwargs)
    except ValueError:
        if 'list[str]' in str(tp) and isinstance(val, list):
            return val
        raise

_jth.adapt_typehints = _patched_adapt

import logging

logger = logging.getLogger(__name__)
try:
    import thor  # noqa: F401

except (ImportError, ModuleNotFoundError) as e:
    logger.error(f"ImportError: {e}")
    error_msg = (
        "THOR package not found. Please install it from https://github.com/fm4cs/thor"
    )
    raise ImportError(error_msg)

from .datamodules import MireMapNonGeoDataModule  # noqa: E402
from .datasets import MireMapDataset, utils  # noqa: E402
from .models.backbones import thor_vit, multi_backbone_wrapper  # noqa: E402
from viksat.necks import LinearProjectionNeck
from viksat.custom_losses import InstanceAwareLoss, BlobLoss 
import terratorch.tasks.segmentation_tasks as seg_tasks
from viksat.segmentation_task import ThresholdedSegmentationTask
import os 
from typing import Union
import inspect 

_original_seg_init = seg_tasks.SemanticSegmentationTask.__init__

def _patched_seg_init(self, *args, loss=None, **kwargs):
    if loss is not None:
        processed_loss = []
        for l in (loss if isinstance(loss, list) else [loss]):
            if isinstance(l, dict) and 'class_path' in l:
                cls_path = l['class_path']
                init_args = l.get('init_args', {})
                if 'InstanceAwareLoss' in cls_path:
                    processed_loss.append(InstanceAwareLoss(**init_args))
                elif 'BlobLoss' in cls_path:
                    processed_loss.append(BlobLoss(**init_args))
                else:
                    processed_loss.append(l)
            else:
                processed_loss.append(l)
        loss = processed_loss
    _original_seg_init(self, *args, loss=loss, **kwargs)

seg_tasks.SemanticSegmentationTask.__init__ = _patched_seg_init

_original_init_loss = seg_tasks.init_loss

_LOSS_OVERRIDE_PARAMS = {}

def set_loss_params(**kwargs):
    global _LOSS_OVERRIDE_PARAMS
    _LOSS_OVERRIDE_PARAMS = kwargs

_INSTANCE_LOSS_DEFAULTS = {
    "instance_aware": dict(min_size=100, fp_weight=0.5, ce_weight=0.5, iou_threshold=0.5),
    "blob_loss": dict(min_size=100, ce_weight=0.5),
}

def _patched_init_loss(loss, ignore_index=None, class_weights=None):
    if loss in _INSTANCE_LOSS_DEFAULTS:
        params = {**_INSTANCE_LOSS_DEFAULTS[loss], **_LOSS_OVERRIDE_PARAMS}
        if loss == "blob_loss":
            return BlobLoss(ignore_index=ignore_index or 100, **params)
        return InstanceAwareLoss(ignore_index=ignore_index or 100, **params)
    return _original_init_loss(loss, ignore_index=ignore_index, class_weights=class_weights)

seg_tasks.init_loss = _patched_init_loss

__all__ = [
    "utils",
    "thor_vit",
    "multi_backbone_wrapper",
    "MireMapDataset",
    "MireMapNonGeoDataModule",
    "necks"
    "InstanceAwareLoss"
    "BlobLoss",
    "ThresholdedSegmentationTask",
    "set_loss_params"
]
