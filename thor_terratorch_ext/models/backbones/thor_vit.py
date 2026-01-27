import logging
import warnings
from copy import deepcopy
from functools import partial
from pathlib import Path
from typing import Any, Iterable, Literal

import torch
import torch.nn.functional as F  # noqa: N812
import yaml
from terratorch.datasets import HLSBands, OpticalBands, SARBands
from terratorch.models.necks import Neck
from terratorch.registry import TERRATORCH_BACKBONE_REGISTRY, TERRATORCH_NECK_REGISTRY
from thor.core.model_registry import MODELS
from torch import nn

from thor_terratorch_ext.datasets.utils import OLCIBands, SARThorBands, SLSTRBands

logger = logging.getLogger(__name__)


_default_input_params = {
    "ground_covers": [2880],  # m
    "aggr_type": "subsetmean",
    "use_superposition_encoding": False,
    "use_fuzzy_encoding": False,
    "encoder_pos_type": "alibi",
    "cls_token_type": "pooled",
    "use_flexivit": False,  # True equals drawing random patch sizes for each group, not possible when finetuning
    "flexivit_ref_patch_size": 4,  # Smallest possible patch size
    "flexivit_patch_size_seqs": [
        8
    ],  # List of patch sizes to use for each group, or list of length 1 with patch size to use
    "flexivit_ref_grid_size": 14,  # If using positional encoding, this is the reference grid size, we use alibi encoding instead
    "select_patch_strategy": "min",  # min, max, "equal-min" or "equal-max"
    # "equal-min" and "equal-max" means that we try to select patch sizes giving equal number of patches per group, raises error if not possible
    # i.e., patch size 10 for GSD 60m, and patch size 30 for GSD 20m and patch size 60 for GSD 10m
    # NOTE: even though all groups are defined here, only the ones specified in the input to the model will be used
    "groups": [
        ["S2:Red", "S2:Green", "S2:Blue", "S2:NIR"],
        ["S2:RE1", "S2:RE2", "S2:RE3", "S2:RE4", "S2:SWIR1", "S2:SWIR2"],
        ["S2:CoastAerosal", "S2:WaterVapor"],
        ["S1:IW-VH", "S1:IW-VV", "S1:EW-VH", "S1:EW-VV"],
        ["S1:IW-HV", "S1:IW-HH", "S1:EW-HV", "S1:EW-HH"],
        [
            "S3:Oa01_reflectance",
            "S3:Oa02_reflectance",
            "S3:Oa03_reflectance",
            "S3:Oa04_reflectance",
            "S3:Oa05_reflectance",
            "S3:Oa06_reflectance",
            "S3:Oa07_reflectance",
        ],
        [
            "S3:Oa08_reflectance",
            "S3:Oa09_reflectance",
            "S3:Oa10_reflectance",
            "S3:Oa11_reflectance",
            "S3:Oa12_reflectance",
            "S3:Oa13_reflectance",
            "S3:Oa14_reflectance",
        ],
        [
            "S3:Oa15_reflectance",
            "S3:Oa16_reflectance",
            "S3:Oa17_reflectance",
            "S3:Oa18_reflectance",
            "S3:Oa19_reflectance",
            "S3:Oa20_reflectance",
            "S3:Oa21_reflectance",
        ],
        [
            "S3:S1_reflectance_an",
            "S3:S2_reflectance_an",
            "S3:S3_reflectance_an",
            "S3:S4_reflectance_an",
            "S3:S5_reflectance_an",
            "S3:S6_reflectance_an",
        ],
        ["S3:S7_BT_in", "S3:S8_BT_in", "S3:S9_BT_in"],
    ],
    # NOTE: the patch sizes are used for the reference patch embedding weights, the actual patch sizes depends on the flexivit patch size
    "channels": {
        "S2:Red": {
            "GSD": 10,  # m
            "patch_size": 16,  # px
        },
        "S2:Green": {
            "GSD": 10,  # m
            "patch_size": 16,  # px
        },
        "S2:Blue": {
            "GSD": 10,
            "patch_size": 16,  # px
        },
        "S2:NIR": {
            "GSD": 10,
            "patch_size": 16,  # px
        },
        "S2:RE1": {
            "GSD": 20,
            "patch_size": 16,  # px
        },
        "S2:RE2": {
            "GSD": 20,
            "patch_size": 16,  # px
        },
        "S2:RE3": {
            "GSD": 20,
            "patch_size": 16,  # px
        },
        "S2:RE4": {
            "GSD": 20,
            "patch_size": 16,  # px
        },
        "S2:SWIR1": {
            "GSD": 20,
            "patch_size": 16,  # px
        },
        "S2:SWIR2": {
            "GSD": 20,
            "patch_size": 16,  # px
        },
        "S2:CoastAerosal": {
            "GSD": 60,
            "patch_size": 16,  # px
        },
        "S2:WaterVapor": {
            "GSD": 60,
            "patch_size": 16,  # px
        },
        # NOTE: new models use patch_embed_name to
        # use the same patch embedding weights for both IW and EW
        "S1:IW-VV": {
            "GSD": 10,
            "patch_size": 16,  # px
            "patch_embed_name": "S1:VV",
        },
        "S1:IW-VH": {
            "GSD": 10,
            "patch_size": 16,  # px
            "patch_embed_name": "S1:VH",
        },
        "S1:IW-HV": {
            "GSD": 10,
            "patch_size": 16,  # px
            "patch_embed_name": "S1:HV",
        },
        "S1:IW-HH": {
            "GSD": 10,
            "patch_size": 16,  # px
            "patch_embed_name": "S1:HH",
        },
        "S1:EW-VV": {
            "GSD": 10,
            "patch_size": 16,  # px
            "patch_embed_name": "S1:VV",
        },
        "S1:EW-VH": {
            "GSD": 10,
            "patch_size": 16,  # px
            "patch_embed_name": "S1:VH",
        },
        "S1:EW-HV": {
            "GSD": 10,  # 250  # 10
            "patch_size": 16,  # px
            "patch_embed_name": "S1:HV",
        },
        "S1:EW-HH": {
            "GSD": 10,  # 250 # 10
            "patch_size": 16,  # px
            "patch_embed_name": "S1:HH",
        },
        "S3:Oa01_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa02_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa03_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa04_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa05_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa06_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa07_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa08_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa09_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa10_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa11_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa12_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa13_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa14_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa15_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa16_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa17_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa18_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa19_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa20_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:Oa21_reflectance": {
            "GSD": 240,  # GSD 240 interp
            "patch_size": 16,  # px
        },
        "S3:S1_reflectance_an": {
            "GSD": 480,  # GSD 480 interp
            "patch_size": 16,  # px
        },
        "S3:S2_reflectance_an": {
            "GSD": 480,  # GSD 480 interp
            "patch_size": 16,  # px
        },
        "S3:S3_reflectance_an": {
            "GSD": 480,  # GSD 480 interp
            "patch_size": 16,  # px
        },
        "S3:S4_reflectance_an": {
            "GSD": 480,  # GSD 480 interp
            "patch_size": 16,  # px
        },
        "S3:S5_reflectance_an": {
            "GSD": 480,  # GSD 480 interp
            "patch_size": 16,  # px
        },
        "S3:S6_reflectance_an": {
            "GSD": 480,  # GSD 480 interp
            "patch_size": 16,  # px
        },
        "S3:S7_BT_in": {
            "GSD": 960,  # GSD 960 interp
            "patch_size": 16,  # px
        },
        "S3:S8_BT_in": {
            "GSD": 960,  # GSD 960 interp
            "patch_size": 16,  # px
        },
        "S3:S9_BT_in": {
            "GSD": 960,  # GSD 960 interp
            "patch_size": 16,  # px
        },
    },
}

_user_overridable_params = {
    "ground_covers": "List of ground cover sizes (m) that define the crop fed to the ViT",
    "flexivit_patch_size_seqs": "Per-group or shared FlexiViT patch sizes to use",
    "flexivit_ref_patch_size": "Reference patch size for FlexiViT patch embedding",
    "select_patch_strategy": "Strategy for choosing patch sizes across groups (min, max, equal-*)",
}


def _format_overridable_params() -> str:
    return ", ".join(
        f"{key}: {_user_overridable_params[key]}"
        for key in sorted(_user_overridable_params)
    )


def _ensure_allowed_input_params(keys: Iterable[str]) -> None:
    disallowed_keys = sorted(set(keys) - set(_user_overridable_params))
    if disallowed_keys:
        allowed_str = _format_overridable_params()
        msg = (
            "Cannot override the following THOR ViT input params via configuration: "
            f"{disallowed_keys}. Allowed keys: {allowed_str}"
        )
        raise ValueError(msg)


#############################################################################################
# NOTE: these are the calculated mean std values from the *pretraining* dataset used in THOR
# They are only included here for reference.
# You should calculate dataset specific mean and std values for fine-tuning
THOR_NORMALIZATION_PARAMS = {
    # S2-10m #################
    "S2:Blue": {"mean": 0.176620, "std": 0.264520},
    "S2:Green": {"mean": 0.195923, "std": 0.252949},
    "S2:Red": {"mean": 0.213948, "std": 0.259180},
    "S2:NIR": {"mean": 0.308133, "std": 0.226434},
    # S2-20m #################
    "S2:RE1": {"mean": 0.263378, "std": 0.272771},
    "S2:RE2": {"mean": 0.300818, "std": 0.248175},
    "S2:RE3": {"mean": 0.313144, "std": 0.235432},
    "S2:RE4": {"mean": 0.320993, "std": 0.223274},
    "S2:SWIR1": {"mean": 0.221550, "std": 0.171606},
    "S2:SWIR2": {"mean": 0.175772, "std": 0.156223},
    # S2-60m #################
    "S2:CoastAerosal": {"mean": 0.182569, "std": 0.282463},
    "S2:WaterVapor": {"mean": 0.322589, "std": 0.235857},
    # S1-IW-VV-60m #################
    "S1:IW-VH_60": {"mean": -20.6672, "std": 5.9634},
    "S1:IW-VV_60": {"mean": -13.0095, "std": 5.2439},
    # S1-IW-VV-10m #################
    "S1:IW-VH_10": {"mean": -20.6958, "std": 5.8688},
    "S1:IW-VV_10": {"mean": -12.9850, "std": 5.0062},
    # S1-IW-HH-60m #################
    "S1:IW-HH_60": {"mean": -13.0126, "std": 7.1580},
    "S1:IW-HV_60": {"mean": -21.2863, "std": 7.6221},
    # S1-IW-HH-10m #################
    "S1:IW-HH_10": {"mean": -13.9485, "std": 6.8015},
    "S1:IW-HV_10": {"mean": -22.4575, "std": 6.9106},
    # S1-EW-VV-60m #################
    "S1:EW-VH_60": {"mean": -23.2024, "std": 7.0584},
    "S1:EW-VV_60": {"mean": -13.1240, "std": 5.5013},
    # S1-EW-VV-10m #################
    "S1:EW-VH_10": {"mean": -23.5719, "std": 6.8895},
    "S1:EW-VV_10": {"mean": -13.9046, "std": 6.3085},
    # S1-EW-HH-60m #################
    "S1:EW-HH_60": {"mean": -12.1138, "std": 6.5830},
    "S1:EW-HV_60": {"mean": -21.7450, "std": 7.4658},
    # S1-EW-HH-10m #################
    "S1:EW-HH_10": {"mean": -12.7691, "std": 6.6416},
    "S1:EW-HV_10": {"mean": -22.6922, "std": 7.2472},
    # S3-250m #################
    "S3:Oa01_reflectance": {"mean": 0.418360, "std": 0.271155},
    "S3:Oa02_reflectance": {"mean": 0.407687, "std": 0.276213},
    "S3:Oa03_reflectance": {"mean": 0.384837, "std": 0.286278},
    "S3:Oa04_reflectance": {"mean": 0.356827, "std": 0.291605},
    "S3:Oa05_reflectance": {"mean": 0.342093, "std": 0.284960},
    "S3:Oa06_reflectance": {"mean": 0.313561, "std": 0.259517},
    "S3:Oa07_reflectance": {"mean": 0.305947, "std": 0.260534},
    "S3:Oa08_reflectance": {"mean": 0.326632, "std": 0.286289},
    "S3:Oa09_reflectance": {"mean": 0.329984, "std": 0.290304},
    "S3:Oa10_reflectance": {"mean": 0.331824, "std": 0.291953},
    "S3:Oa11_reflectance": {"mean": 0.348113, "std": 0.281539},
    "S3:Oa12_reflectance": {"mean": 0.394400, "std": 0.265238},
    "S3:Oa13_reflectance": {"mean": 0.101062, "std": 0.066239},
    "S3:Oa14_reflectance": {"mean": 0.188930, "std": 0.122599},
    "S3:Oa15_reflectance": {"mean": 0.352127, "std": 0.233035},
    "S3:Oa16_reflectance": {"mean": 0.394914, "std": 0.258835},
    "S3:Oa17_reflectance": {"mean": 0.402817, "std": 0.251323},
    "S3:Oa18_reflectance": {"mean": 0.398475, "std": 0.244922},
    "S3:Oa19_reflectance": {"mean": 0.312429, "std": 0.213306},
    "S3:Oa20_reflectance": {"mean": 0.166702, "std": 0.156878},
    "S3:Oa21_reflectance": {"mean": 0.380989, "std": 0.209298},
    # S3-500m #################
    "S3:S1_reflectance_an": {"mean": 0.338953, "std": 0.280222},
    "S3:S2_reflectance_an": {"mean": 0.339034, "std": 0.294075},
    "S3:S3_reflectance_an": {"mean": 0.405393, "std": 0.262460},
    "S3:S4_reflectance_an": {"mean": 0.010942, "std": 0.030842},
    "S3:S5_reflectance_an": {"mean": 0.184803, "std": 0.140885},
    "S3:S6_reflectance_an": {"mean": 0.137453, "std": 0.109526},
    # S3-1000m #################
    "S3:S7_BT_in": {"mean": 286.190167, "std": 21.070093},
    "S3:S8_BT_in": {"mean": 278.423815, "std": 22.109991},
    "S3:S9_BT_in": {"mean": 277.117165, "std": 21.526558},
}
###########################################################################################


lookup_band = {
    # Optical bands
    "COASTAL_AEROSOL": "S2:CoastAerosal",
    "BLUE": "S2:Blue",
    "GREEN": "S2:Green",
    "RED": "S2:Red",
    "RED_EDGE_1": "S2:RE1",
    "RED_EDGE_2": "S2:RE2",
    "RED_EDGE_3": "S2:RE3",
    "NIR_BROAD": "S2:NIR",
    "NIR_NARROW": "S2:RE4",
    "SWIR_1": "S2:SWIR1",
    "SWIR_2": "S2:SWIR2",
    "WATER_VAPOR": "S2:WaterVapor",
    # SAR bands
    "VV": "S1:IW-VV",
    "VH": "S1:IW-VH",
    "ASC_VV": "S1:IW-VV",
    "ASC_VH": "S1:IW-VH",
    "DSC_VV": "S1:IW-VV",
    "DSC_VH": "S1:IW-VH",
    "IW_VV": "S1:IW-VV",
    "IW_VH": "S1:IW-VH",
    "IW_HV": "S1:IW-HV",
    "IW_HH": "S1:IW-HH",
    "EW_VV": "S1:EW-VV",
    "EW_VH": "S1:EW-VH",
    "EW_HV": "S1:EW-HV",
    "EW_HH": "S1:EW-HH",
    # OLCI bands
    "OA01_REFLECTANCE": "S3:Oa01_reflectance",
    "OA02_REFLECTANCE": "S3:Oa02_reflectance",
    "OA03_REFLECTANCE": "S3:Oa03_reflectance",
    "OA04_REFLECTANCE": "S3:Oa04_reflectance",
    "OA05_REFLECTANCE": "S3:Oa05_reflectance",
    "OA06_REFLECTANCE": "S3:Oa06_reflectance",
    "OA07_REFLECTANCE": "S3:Oa07_reflectance",
    "OA08_REFLECTANCE": "S3:Oa08_reflectance",
    "OA09_REFLECTANCE": "S3:Oa09_reflectance",
    "OA10_REFLECTANCE": "S3:Oa10_reflectance",
    "OA11_REFLECTANCE": "S3:Oa11_reflectance",
    "OA12_REFLECTANCE": "S3:Oa12_reflectance",
    "OA13_REFLECTANCE": "S3:Oa13_reflectance",
    "OA14_REFLECTANCE": "S3:Oa14_reflectance",
    "OA15_REFLECTANCE": "S3:Oa15_reflectance",
    "OA16_REFLECTANCE": "S3:Oa16_reflectance",
    "OA17_REFLECTANCE": "S3:Oa17_reflectance",
    "OA18_REFLECTANCE": "S3:Oa18_reflectance",
    "OA19_REFLECTANCE": "S3:Oa19_reflectance",
    "OA20_REFLECTANCE": "S3:Oa20_reflectance",
    "OA21_REFLECTANCE": "S3:Oa21_reflectance",
    # SLSTR bands
    "S1_REFLECTANCE_AN": "S3:S1_reflectance_an",
    "S2_REFLECTANCE_AN": "S3:S2_reflectance_an",
    "S3_REFLECTANCE_AN": "S3:S3_reflectance_an",
    "S4_REFLECTANCE_AN": "S3:S4_reflectance_an",
    "S5_REFLECTANCE_AN": "S3:S5_reflectance_an",
    "S6_REFLECTANCE_AN": "S3:S6_reflectance_an",
    "S7_BT_IN": "S3:S7_BT_in",
    "S8_BT_IN": "S3:S8_BT_in",
    "S9_BT_IN": "S3:S9_BT_in",
}

# NOTE: this gets edited in the THOREncoderWrapper to match the actual groups used
AVAILABLE_GROUPS = {
    f"group{i}": group for i, group in enumerate(_default_input_params["groups"])
}
###################################################################################


def process_thor_bands(
    bands: list[
        HLSBands | OpticalBands | SARBands | SARThorBands | OLCIBands | SLSTRBands
    ],
) -> tuple[list[str], dict[str, dict[str, Any]]]:
    """
    Process a list of bands and map them to THOR band names. For SAR bands, handle GSD suffixes and update channel parameters accordingly.
    Args:
        bands (list[HLSBands | OpticalBands | SARBands | SARThorBands | OLCIBands | SLSTRBands]): List of bands to process.
    Returns:
        tuple[list[str], dict[str, dict[str, Any]]]: A tuple containing the list of THOR band names and the updated channel parameters.
    Raises:
        NotImplementedError: If a band is not implemented in THOR.
        ValueError: If group for a band is not found or if duplicate bands are found after processing.
    """
    thor_bands = []
    channel_params = deepcopy(_default_input_params["channels"])
    for band in bands:
        if not isinstance(band, str):
            band = band.value

        try:
            # Handle SAR bands with GSD suffixes
            if (gsd_str := band.split("_")[-1]).isdigit() and any(
                (part in band.split("_") for part in ["VV", "VH", "HH", "HV"])
            ):
                band = "_".join(band.split("_")[:-1])
                thor_band = lookup_band[band]
                logger.info(
                    f"Detected GSD suffix: {gsd_str} for band: {band}, mapped to THOR band: {thor_band}"
                )
                if int(gsd_str) == channel_params[thor_band]["GSD"]:
                    logger.info(
                        f"GSD {gsd_str} matches default GSD for band {thor_band}, no update needed."
                    )
                    thor_bands.append(thor_band)
                    continue
                else:
                    logger.info(
                        f"Original channel params: {_default_input_params['channels'][thor_band]}, updating GSD to {gsd_str}"
                    )
                channel_params[thor_band]["GSD"] = int(gsd_str)
                group_channel_members = None
                for group in AVAILABLE_GROUPS.values():
                    if thor_band in group:
                        group_channel_members = group
                        break
                if group_channel_members is None:
                    msg = f"Could not find group for band: {thor_band}"
                    raise ValueError(msg)
                logger.info(
                    f"Updating GSD for group members if necessary: {group_channel_members}"
                )
                for member in group_channel_members:
                    if channel_params[member]["GSD"] != int(gsd_str):
                        logger.info(
                            f"Updating GSD for group member: {member} to {gsd_str}"
                        )
                        channel_params[member]["GSD"] = int(gsd_str)

                thor_bands.append(thor_band)
            else:
                thor_band = lookup_band[band]
                thor_bands.append(thor_band)

        except KeyError:
            msg = f"This band is not implemented in THOR: {band}"
            raise NotImplementedError(msg)

    if any(thor_bands.count(b) > 1 for b in set(thor_bands)):
        duplicates = [
            (b, bands[i]) for i, b in enumerate(thor_bands) if thor_bands.count(b) > 1
        ]
        msg = f"Duplicate bands are not allowed/implemented. Make sure to only use one sigma0 product. Duplicates found: {duplicates}"
        raise ValueError(msg)
    return thor_bands, channel_params


@TERRATORCH_NECK_REGISTRY.register
class THORGroupReshapeTokensToImage(Neck):
    def __init__(
        self,
        channel_list: list[int],
        merge: Literal["concat", "sum", "mean"] = "concat",
        remove_cls_token=False,
    ):
        """THOR specific neck to transform sequence of tokens into a feature map.

        First extracts the embeddings for each group, then reshapes each group into feature maps.
        Finally, interpolates the feature maps to the highest num_patches and concatenates the feature maps.

        Args:
            channel_list (list[int]): List of input channel sizes
            merge (str): Method to merge the feature maps from different groups, either 'concat', 'sum' or 'mean'
            remove_cls_token (bool): Whether to remove the cls token from the input features

        """

        super().__init__(channel_list)

        # TODO: add support for or make other necks which allow for alternatives to interpolation and channel wise concat
        # For example: generate a feature pyramid from one single input, by extracting embeddings for different groups

        assert all(self.channel_list[0] == c for c in self.channel_list), (
            "All channels must have the same embedding size"
        )

        self.single_embedding_shape = self.channel_list[0]
        self.groups = AVAILABLE_GROUPS
        assert merge in ["concat", "sum", "mean"], (
            "Merge must be either 'concat', 'sum' or 'mean'"
        )
        self.merge = merge
        self.remove_cls_token = remove_cls_token
        self.highest_num_patch = None

    def forward(
        self,
        features: list[torch.Tensor] | tuple[list[torch.Tensor], dict[str, Any]],
        **kwargs,
    ) -> list[torch.Tensor]:
        """Stack embeddings for each group, requires interpolation to the highest num_patch."""

        if isinstance(features, tuple):
            features, channel_params = features
            highest_num_patch = 0
            for _channel, params in channel_params.items():
                num_patch = params["num_patch"]
                highest_num_patch = max(highest_num_patch, num_patch)
                self.highest_num_patch = highest_num_patch
        else:
            msg = (
                "THORGroupReshapeTokensToImage requires channel_params to be passed during forward"
                "please set return_channel_params=True in the THOREncoderWrapper"
            )
            raise ValueError(msg)

        out_features = []
        for feature in features:
            if self.remove_cls_token:
                x = feature[:, 1:]
            else:
                x = feature

            start_idx = 0
            out = []
            # Important that we iterate through this in the same order we encoded
            for group_members in self.groups.values():
                member = next((m for m in group_members if m in channel_params), None)
                if member is None:
                    msg = f"None of the group members {group_members} found in channel_params"
                    raise ValueError(msg)

                num_patch = channel_params[member]["num_patch"]

                x_ = x[:, start_idx : start_idx + num_patch**2, :].reshape(
                    -1, num_patch, num_patch, self.single_embedding_shape
                )  # B, num_patch, num_patch, C
                x_ = x_.permute(0, 3, 1, 2)  # B, C, H, W
                # TODO: maybe add support for learned interpolation and/or learned channel reduction
                if num_patch != self.highest_num_patch:
                    x_ = F.interpolate(
                        x_,
                        size=(self.highest_num_patch, self.highest_num_patch),
                        mode="bilinear",
                    )

                out.append(x_)
                start_idx += num_patch**2

            if start_idx != x.shape[1]:
                msg = f"Number of patches used: {start_idx} does not match input shape {x.shape[-1]}"
                raise ValueError(msg)

            if self.merge == "sum":
                out = torch.sum(torch.stack(out), dim=0)
            elif self.merge == "mean":
                out = torch.mean(torch.stack(out), dim=0)
            elif self.merge == "concat":
                out = torch.cat(out, dim=1)
            out_features.append(out)

        return out_features

    def process_channel_list(self, channel_list: list[int]) -> list[int]:
        if self.merge in ["sum", "mean"]:
            return [c for c in channel_list]
        elif self.merge == "concat":
            return [c * len(self.groups) for c in channel_list]


class THOREncoderWrapper(nn.Module):
    def __init__(
        self,
        model: nn.Module = None,
        bands: list[str] | None = None,
        out_indices: list[int] | None = None,
        return_channel_params: bool = False,
    ) -> None:
        super().__init__()

        self.model = model
        self.return_channel_params = return_channel_params
        self.channels = self.model.channels

        if bands is None:
            logger.info("Bands not provided, using model default bands")
            bands = list(self.model.channels.keys())
        self.bands = bands
        self.band_index = list(range(len(bands)))
        self.groups = self.model.get_available_groups(dict.fromkeys(self.bands, None))
        ########NOTE: DUBIOUS CODE BELOW ############
        all_group_keys = list(AVAILABLE_GROUPS.keys())
        removed_groups = []
        for group_name in all_group_keys:
            if group_name not in self.groups:
                removed_groups.append(group_name)
                del AVAILABLE_GROUPS[group_name]
        if removed_groups:
            logger.info(
                f"Removed groups: {removed_groups}. Remaining groups: {list(AVAILABLE_GROUPS.keys())}"
            )
        #########################################

        logger.info(f"Groups: {self.groups}")
        logger.info(f"Bands: {self.bands}")

        self.single_embedding_shape = self.model.state_dict()["norm.bias"].shape[0]

        self.inner_forward = self.model.forward_intermediates
        self.forward = self._forward_vit

        lowest_gsd = 1000000
        for group_member in self.groups.values():
            product_band = group_member[0]
            lowest_gsd = min(lowest_gsd, self.channels[product_band]["GSD"])
        self.lowest_gsd = lowest_gsd

        if out_indices is None:
            num_blocks = len(self.model.blocks)
            out_indices = list(range(0, num_blocks, 1))  # every block
        self.out_indices = out_indices

        # TODO: do this a better way
        if hasattr(self.model, "ground_covers"):
            assert len(self.model.ground_covers) == 1, (
                "Multiple ground covers found, please specify which one to use"
            )
            ground_cover = self.model.ground_covers[0]
        else:
            msg = "No ground cover found in model"
            raise ValueError(msg)
        self.ground_cover = ground_cover
        self.input_size = ground_cover // self.lowest_gsd

    @property
    def out_channels(self):
        return [self.single_embedding_shape] * len(self.out_indices)

    def _preprocess_input(self, x):
        x = {
            channel: F.interpolate(
                x[:, [band_index], :, :],
                (
                    int(self.ground_cover / self.channels[channel]["GSD"]),
                    int(self.ground_cover / self.channels[channel]["GSD"]),
                ),
                mode="bilinear",
            )
            for band_index, channel in zip(self.band_index, self.bands, strict=False)
        }

        return x

    def _forward_vit(self, x, **kwargs):
        x = self._preprocess_input(x)
        x = self.inner_forward(
            x,
            indices=self.out_indices,
            intermediates_only=True,
            return_channel_params=self.return_channel_params,
        )
        return x

    def summary(self):
        pass


def load_thor_model(
    model_name: str,
    model_bands: list[
        HLSBands | OpticalBands | SARBands | SARThorBands | OLCIBands | SLSTRBands
    ],
    out_indices: list[int] | None = None,
    pretrained: bool = False,
    **kwargs,
):
    # Map to THOR bands names and get updated channel params if necessary
    bands, channel_params_updated = process_thor_bands(model_bands)
    logger.info(f"bands mapped to thor: {bands}")

    config = kwargs.pop("config", None)
    if isinstance(config, str | Path):
        logger.info(f"Loading backbone config from {config}")
        config = yaml.safe_load(open(config))
    ckpt_path = kwargs.pop(
        "ckpt", None
    )  # path to checkpoint to load, will override config if provided
    input_params = kwargs.pop(
        "input_params", {}
    )  # dict with input params to override config if provided

    return_channel_params = kwargs.pop("return_channel_params", True)

    default_input_params = deepcopy(_default_input_params)
    default_input_params["channels"].update(channel_params_updated)
    model_config = {
        "type": model_name,
        "strict": False,  # Whether to strictly enforce that the keys in state_dict match the keys returned by the model's state_dict function.
        "ckpt": None,  # Path to checkpoint to load, if None, will not load any checkpoint
        "ckpt_ignore": [
            "decoder_.*",
            "decode_.*",
            "ref_pos_embed.*",
            "pos_embed.*",
        ],  # regex ignore list, removing mae head
        "input_params": default_input_params,
    }
    model_checkpoint_key = kwargs.pop("model_ckpt_type", "mae")

    if config is not None:
        if "models" in config:
            config = config["models"]
        if model_checkpoint_key in config:
            config = config[model_checkpoint_key]
        else:
            msg = f"Model {model_checkpoint_key} not found in config file {config}. Available models are {config.keys()}"
            raise ValueError(msg)
        # Merge loaded config with backbone_config, giving priority to loaded_config
        model_config.update(dict(config))

    if model_name != model_config["type"]:
        msg = f"Backbone name {model_name} does not match model type {model_config['type']}"
        raise ValueError(msg)

    if ckpt_path is not None:
        if model_config["ckpt"]:
            warnings.warn(
                f"Overwriting checkpoint path for model {model_checkpoint_key} with {ckpt_path}",
                stacklevel=2,
            )
        else:
            logger.info(
                f"Setting checkpoint path for model {model_checkpoint_key} to {ckpt_path}"
            )
        model_config["ckpt"] = ckpt_path

    if input_params:
        _ensure_allowed_input_params(input_params.keys())
        for k, v in input_params.items():
            if k in model_config["input_params"]:
                warnings.warn(
                    f"Overwriting input param {k} for model {model_checkpoint_key} with {v}",
                    stacklevel=2,
                )
            else:
                logger.info(
                    f"Setting input param {k} for model {model_checkpoint_key} to {v}"
                )
            model_config["input_params"][k] = v

    if pretrained and model_config["ckpt"] is None:
        msg = f"Pretrained is set to True, but no checkpoint was provided for model {model_checkpoint_key}. Please provide a checkpoint path."
        raise ValueError(msg)

    logger.info(f"Instantiating model {model_checkpoint_key} of type {model_name}")
    model = MODELS.build(model_cfgs={model_checkpoint_key: model_config})[
        model_checkpoint_key
    ]

    return THOREncoderWrapper(
        model=model,
        bands=bands,
        out_indices=out_indices,
        return_channel_params=return_channel_params,
    )


def _partial_with_name(func, /, *args, **keywords):
    partial_with_name = partial(func, *args, **keywords)
    partial_with_name.__name__ = keywords.get("model_name", "none")
    return partial_with_name


def register_thor_models():
    if getattr(register_thor_models, "_done", False):
        return
    logger.debug("Registering THOR models in TERRATORCH_BACKBONE_REGISTRY")
    for model_name in MODELS.models.keys():
        if model_name.startswith("thor_vit"):
            logger.debug(f"-> registering {model_name} as THOR ViT model")
            TERRATORCH_BACKBONE_REGISTRY.register(
                _partial_with_name(load_thor_model, model_name=model_name)
            )
    register_thor_models._done = True


register_thor_models()
