# THOR TerraTorch Extension

[![arXiv](https://img.shields.io/badge/arXiv-2601.16011-b31b1b?logo=arxiv)](https://arxiv.org/abs/2601.16011)
[![Models](https://img.shields.io/badge/Models-HuggingFace-FFD21E?logo=huggingface)](https://huggingface.co/FM4CS)
[![Website](https://img.shields.io/badge/Website-THOR-0F62FE)](https://thor-model.notion.site/THOR-Foundation-Model-Showcase-2ee64c7f3cb78087bf77feb6350bdcc6)

This repository provides the [TerraTorch](https://github.com/terrastackai/terratorch) extension for using the **THOR** (Transformer based foundation model for Heterogeneous Observation and Resolution) model in downstream applications. 

The model and pretraining code for THOR can be found [here](https://github.com/FM4CS/THOR).

## Key Features

- **Multi-sensor support**: Sentinel-1 (SAR), Sentinel-2 (MSI), Sentinel-3 OLCI & SLSTR
- **Flexible resolution**: 10 m to 1000 m native resolutions
- **Compute-adaptive**: Flexible patch sizes and ground covers (1000 m to +100,000 m)
- **Data-efficient**: State-of-the-art performance in data-limited regimes

## Setup

Install using `uv`:
```bash
uv sync
```

## Usage

### Finetuning with TerraTorch

To run finetuning on the sen1floods11 dataset with Sentinel-2 and Sentinel-1 data using the THOR ViT Base backbone:

```bash
uv run terratorch fit --config config/sen1floods11_thor_multimodal.yaml
```

The config file uses the `custom_modules_path` key which should point to the `thor_terratorch_ext` module. Alternatively, you can specify the module path using the `--custom-modules-path` argument:

```bash
uv run terratorch fit --config config/sen1floods11_thor_multimodal.yaml --custom-modules-path thor_terratorch_ext
```

### Loading the Backbone Directly

You can load the THOR ViT backbone directly in Python:

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
        ],  # Ground cover in meters (typically input image size [px] * input image resolution)
        flexivit_patch_size_seqs=[8],  # Patch size in pixels
    ),
)
```

## Attribution

The development of THOR was funded and supported by European Space Agency (ESA) Φ-lab (FM4CS project, contract no. 4000143489/24/I-DT), and the Research Council of Norway (KnowEarth project no. 337481).

## Citation

If you use THOR in your research, please cite the [paper](https://arxiv.org/abs/2601.16011):

```bibtex
@article{forgaard2026thor,
      title={THOR: A Versatile Foundation Model for Earth Observation Climate and Society Applications}, 
      author={Theodor Forgaard and Jarle H. Reksten and Anders U. Waldeland and Valerio Marsocci and Nicolas Longépé and Michael Kampffmeyer and Arnt-Børre Salberg},
      year={2026},
      eprint={2601.16011},
      archivePrefix={arXiv},
      primaryClass={eess.IV},
      url={https://arxiv.org/abs/2601.16011}, 
}
```