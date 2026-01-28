"""Monkey patch EmbeddingGenerationTask to add THORGroupReshapeTokensToImage neck when using THOR ViT backbones,
as EmbeddingGenerationTask does not natively support this necks.
"""

import logging

import torch
from terratorch.models.utils import TemporalWrapper, get_image_size
from terratorch.registry import BACKBONE_REGISTRY, NECK_REGISTRY
from terratorch.tasks import EmbeddingGenerationTask

logger = logging.getLogger(__name__)

logger.info(
    "Applying monkey patch to EmbeddingGenerationTask.configure_models to add THORGroupReshapeTokensToImage neck when using THOR ViT backbones"
)


class THORModelWrapper(torch.nn.Module):
    def __init__(self, model, neck):
        super().__init__()
        self.model = model
        self.neck = neck
        self.out_channels = (
            neck.process_channel_list(model.out_channels)
            if hasattr(neck, "process_channel_list")
            else model.out_channels
        )

    def forward(self, x):
        input_size = get_image_size(x)
        features = self.model(x)
        out = self.neck(features, image_size=input_size)
        return out


original_configure_models = EmbeddingGenerationTask.configure_models


def patched_configure_models(self) -> None:
    """Instantiate backbone and optional temporal wrapper."""
    self.model = BACKBONE_REGISTRY.build(
        self.model,
        **(self.model_args or {}),
    )
    logger.info("Model configured, checking for THOR backbone...")

    if (
        hasattr(self.model, "model")
        and "thorvit" in self.model.model.__class__.__name__.lower()
    ):
        logger.info(
            "Patching EmbeddingGenerationTask model to add THORGroupReshapeTokensToImage neck for THOR ViT backbone"
        )
        neck_config = {
            "name": "THORGroupReshapeTokensToImage",
            "init_args": {
                "merge": "mean",
            },
        }
        neck = NECK_REGISTRY.build(
            neck_config["name"], self.model.out_channels, **neck_config["init_args"]
        )
        self.model = THORModelWrapper(self.model, neck)

    if self.temporal_cfg.get("temporal_wrapper", False):
        self.model = TemporalWrapper(
            self.model,
            pooling=self.temporal_cfg.get("temporal_pooling", "keep"),
        )
    self.model.eval()


EmbeddingGenerationTask.configure_models = patched_configure_models
