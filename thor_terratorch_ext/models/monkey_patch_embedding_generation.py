"""Monkey patch EmbeddingGenerationTask to call self.model(x) instead of self.model.encoder(x) to get post neck outputs for configuration summary."""

import logging
from datetime import datetime, timezone
import json

import torch
from terratorch.tasks import EmbeddingGenerationTask

logger = logging.getLogger(__name__)


def save_configuration_summary(
    self,
    x: torch.Tensor | dict[str, torch.Tensor],
) -> None:
    """
    Saves a JSON containing model, layer configuration, and output specs.
    """
    if self._config_saved:
        return

    outputs = self.model(x)

    if not isinstance(outputs, list):
        outputs = [outputs]
    n_outputs = len(outputs)

    resolved_indices = [
        (idx if idx >= 0 else n_outputs + idx) for idx in self.embedding_indices
    ]

    total_params = sum(p.numel() for p in self.model.parameters()) / 1e6
    config_summary = {
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "output_dir": str(self.output_path.absolute()),
        "output_format": self.output_format,
        "backbone": self.model_args["backbone"],
        "backbone_total_params_million": total_params,
        "has_cls": self.has_cls,
        "embedding_pooling": self.embedding_pooling,
        "model_layer_count": n_outputs,
        "n_layers_saved": len(self.embedding_indices),
        "layers": [
            {
                "output_folder_name": f"layer_{i:02d}",
                "requested_index": folder,
                "layer_number": res + 1,
                "layer_output_shape": list(outputs[res][0].shape),
            }
            for i, (folder, res) in enumerate(
                zip(self.embedding_indices, resolved_indices)
            )
        ],
    }

    out_path = self.output_path / "configuration_summary.json"
    try:
        with open(out_path, "w") as f:
            json.dump(config_summary, f, indent=2)
        logger.info(f"Configuration summary saved to {out_path}")
    except IOError as e:
        logger.error(f"Failed to write configuration summary: {e}")

    self._config_saved = True


logger.info(
    "Monkey patching EmbeddingGenerationTask.save_configuration_summary to call self.model(x) instead of self.model.encoder(x) to get post neck outputs for configuration summary."
)

EmbeddingGenerationTask.save_configuration_summary = save_configuration_summary
