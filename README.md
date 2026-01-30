# THOR TerraTorch Extension

# Install using uv
```bash
uv sync --group log
```

# Usage

As an example, to run finetuning on the sen1floods11 dataset with Sentinel-2 and -1 data using the THOR ViT Base backbone, use the following command:

## Finetuning
```bash
uv run terratorch fit --config config/sen1floods11_thor_multimodal.yaml
```
The config file uses the `custom_modules_path` key which should point to the `thor_terratorch_ext` module.

Alternatively, you can point to the `thor_terratorch_ext` module using the `--custom-modules-path` argument:
```bash
uv run terratorch fit --config config/sen1floods11_thor_multimodal.yaml --custom-modules-path thor_terratorch_ext
```

## Loading the backbone directly
Alternatively, you can load the THOR ViT backbone directly in Python as follows:

```python
# Example usage of THOR ViT backbone with terratorch

# Import our custom thor_terratorch_ext module to register THOR backbones
import thor_terratorch_ext  # noqa: F401

# Load the backbone registry
from terratorch import BACKBONE_REGISTRY

# List available THOR backbones
print([b for b in list(BACKBONE_REGISTRY) if "thor" in b])

# Build a THOR ViT model with specific bands
model = BACKBONE_REGISTRY.build(
    "thor_v1_base",
    pretrained=True,
    model_bands=["BLUE", "GREEN", "RED", "VV", "VH"],
    input_params=dict(  # Optional input parameters to customize
        ground_covers=[
            2880
        ],  # Ground cover in meters (typically input image size * input image resolution)
        flexivit_patch_size_seqs=[8],  # Patch size in pixels
    ),
)
```