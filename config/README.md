
# Example configs

## TerraTorch based dataset configurations
Can be run as-is with the THOR ViT backbone. You can also use these as templates for your own experiments.


### Sen1floods11 segmentation examples

**Note**: to run the sen1floods11 examples you first need to download the sen1floods11 dataset and place it in the correct location (default is root of project). The dataset can be downloaded using the following code snippet:

```python
import gdown

gdown.download(
    "https://drive.google.com/uc?id=1lRw3X7oFNq_WyzBO6uyUJijyTuYm23VS",
    "sen1floods11_v1.1.tar.gz",
    quiet=False,
    fuzzy=True
)
```

followed by extracting the downloaded archive and placing the extracted folder in the correct location:

```bash
tar -xvf sen1floods11_v1.1.tar.gz
```

**[sen1floods11_thor_S2.yaml](sen1floods11_thor_S2.yaml)**: This config is for training a THOR ViT backbone on the sen1floods11 dataset using only Sentinel-2 data. 

**[sen1floods11_thor_S1.yaml](sen1floods11_thor_S1.yaml)**: This config is for training a THOR ViT backbone on the sen1floods11 dataset using only Sentinel-1 data.

**[sen1floods11_thor_multimodal.yaml](sen1floods11_thor_multimodal.yaml)**: This config is for training a THOR ViT backbone on the sen1floods11 dataset using both Sentinel-1 and Sentinel-2 data.

**[sen1floods11_thor_multibackbone_multimodal.yaml](sen1floods11_thor_multibackbone_multimodal.yaml)** and **[sen1floods11_timm_multibackbone_multimodal.yaml](sen1floods11_timm_multibackbone_multimodal.yaml)**: These are example configs for how one can use the `MultibackboneWrapper` to train a multi-backbone model on the sen1floods11 dataset using both Sentinel-1 and Sentinel-2 data, feeding each modality to a separate backbone. Note that they are only example configs for template purposes.


## Embedding generation

**[embedding_generation_sen1floods11_S2.yaml](embedding_generation_sen1floods11_S2.yaml)**: This config is for generating embeddings using a pre-trained THOR ViT backbone. The example config generates embeddings for the sen1floods11 dataset (only Sentinel-2 data) . See above or one of the notebooks for how to download the dataset. The config can be adapted for your own embedding generation experiments.

**[embedding_generation_burnscars.yaml](embedding_generation_burnscars.yaml)**: This config is for generating embeddings using a pre-trained THOR ViT backbone. The example config generates embeddings for the burnscars dataset. The config can be adapted for your own embedding generation experiments. See the notebook [thor_embedding_generation_burnscars.ipynb](../notebooks/thor_embedding_generation_burnscars.ipynb) for an example of how to run the embedding generation using this config.

## Unreleased datasets configs 
The following configs are for datasets that are not yet publicly released, but will be released in the near future.

- [iceberg_thor_S1.yaml](iceberg_thor_S1.yaml)
- [mire_thor_S2_binary.yaml](mire_thor_S2_binary.yaml)
- [mire_thor_S2_multiclass.yaml](mire_thor_S2_multiclass.yaml)

