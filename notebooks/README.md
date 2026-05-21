# Example notebooks

## TerraTorch based dataset examples

### Sen1floods11 segmentation
**[thor_segmentation_sen1floods11.ipynb](thor_segmentation_sen1floods11.ipynb)**

### Embedding generation
**[thor_embedding_generation_sen1floods11.ipynb](thor_embedding_generation_sen1floods11.ipynb)**

## THOR data preprocessing and inference examples

### Phidown Sentinel-3 only data preprocessing and inference workflow with THOR
**[thor_inference_sentinel3_phidown.ipynb](thor_inference_sentinel3_phidown.ipynb)**: Phidown-based workflow for Sentinel-3 data preprocessing and inference with THOR. 

### Phidown Sentinel-1, -2 and -3 data processing workflow for THOR 
**[9_s1_s2_s3_thor_workflow.ipynb](https://github.com/ESA-PhiLab/phidown/blob/main/notebooks/9_s1_s2_s3_thor_workflow.ipynb)**: Phidown workflow for extracting Sentinel-1, Sentinel-2, and Sentinel-3 data for a given location and time and preparing it for use with THOR.
Exports the data to geotiff files which can be used for inference with THOR or any other model.

### THOR Sentinel-1, -2 and -3 inference 

**[10_multimodal_thor_inference.ipynb](10_multimodal_thor_inference.ipynb)**: Example of inference on Sentinel-1, -2 and -3 data using the above phidown data processing pipeline.