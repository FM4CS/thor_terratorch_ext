from enum import Enum


class ThorModalities(Enum):
    """Supported modality keys for THOR."""

    # -- Sentinel-1 GRD -------------------------------------------------------
    # V-transmit dual-pol (VV + VH) – land / ocean globally
    S1GRD = "S1GRD"  # shorthand alias for S1GRD_VV_VH
    S1GRD_VV_VH = "S1GRD_VV_VH"
    # H-transmit dual-pol (HH + HV) – sea ice, polar, some agricultural
    S1GRD_HH_HV = "S1GRD_HH_HV"

    # -- Sentinel-2 -----------------------------------------------------------
    # L2A surface reflectance, 12 bands B01–B12 (excl. B10) in wavelength order
    S2L2A = "S2L2A"

    # -- Sentinel-3 -----------------------------------------------------------
    S3 = "S3"  # shorthand alias for all Sentinel-3 bands (OLCI + SLSTR)
    # OLCI Level-1 reflectance, 21 bands Oa01–Oa21
    S3OLCI = "S3OLCI"
    S3SLSTR = "S3SLSTR"  # SLSTR Level-1 reflectance (S1–S6) + BT (S7–S9)
    # SLSTR nadir-view reflectance bands S1–S6 (6 channels)
    S3SLSTR_REFL = "S3SLSTR_REFL"
    # SLSTR brightness-temperature bands S7–S9 (3 channels)
    S3SLSTR_BT = "S3SLSTR_BT"


class SARThorBands(Enum):
    # NOTE SAR bands are appendable with GSD suffixes like _10, _60, _240 to indicate resolution
    # For example: IW_VV_40 or VV_40 may be used to specify that the band is at 40m GSD.
    # Default is 10m if no suffix is provided.

    IW_VV = "IW_VV"
    IW_VH = "IW_VH"
    IW_HV = "IW_HV"
    IW_HH = "IW_HH"

    EW_VV = "EW_VV"
    EW_VH = "EW_VH"
    EW_HV = "EW_HV"
    EW_HH = "EW_HH"


class S2L2ABands(Enum):
    COASTAL_AEROSOL = "COASTAL_AEROSOL"
    BLUE = "BLUE"
    GREEN = "GREEN"
    RED = "RED"
    RED_EDGE_1 = "RED_EDGE_1"
    RED_EDGE_2 = "RED_EDGE_2"
    RED_EDGE_3 = "RED_EDGE_3"
    NIR_BROAD = "NIR_BROAD"
    NIR_NARROW = "NIR_NARROW"
    WATER_VAPOR = "WATER_VAPOR"
    SWIR_1 = "SWIR_1"
    SWIR_2 = "SWIR_2"


class S3OLCIBands(Enum):
    OA01_REFLECTANCE = "OA01_REFLECTANCE"
    OA02_REFLECTANCE = "OA02_REFLECTANCE"
    OA03_REFLECTANCE = "OA03_REFLECTANCE"
    OA04_REFLECTANCE = "OA04_REFLECTANCE"
    OA05_REFLECTANCE = "OA05_REFLECTANCE"
    OA06_REFLECTANCE = "OA06_REFLECTANCE"
    OA07_REFLECTANCE = "OA07_REFLECTANCE"
    OA08_REFLECTANCE = "OA08_REFLECTANCE"
    OA09_REFLECTANCE = "OA09_REFLECTANCE"
    OA10_REFLECTANCE = "OA10_REFLECTANCE"
    OA11_REFLECTANCE = "OA11_REFLECTANCE"
    OA12_REFLECTANCE = "OA12_REFLECTANCE"
    OA13_REFLECTANCE = "OA13_REFLECTANCE"
    OA14_REFLECTANCE = "OA14_REFLECTANCE"
    OA15_REFLECTANCE = "OA15_REFLECTANCE"
    OA16_REFLECTANCE = "OA16_REFLECTANCE"
    OA17_REFLECTANCE = "OA17_REFLECTANCE"
    OA18_REFLECTANCE = "OA18_REFLECTANCE"
    OA19_REFLECTANCE = "OA19_REFLECTANCE"
    OA20_REFLECTANCE = "OA20_REFLECTANCE"
    OA21_REFLECTANCE = "OA21_REFLECTANCE"


class S3SLSTRBands(Enum):
    S1_REFLECTANCE_AN = "S1_REFLECTANCE_AN"
    S2_REFLECTANCE_AN = "S2_REFLECTANCE_AN"
    S3_REFLECTANCE_AN = "S3_REFLECTANCE_AN"
    S4_REFLECTANCE_AN = "S4_REFLECTANCE_AN"
    S5_REFLECTANCE_AN = "S5_REFLECTANCE_AN"
    S6_REFLECTANCE_AN = "S6_REFLECTANCE_AN"
    S7_BT_IN = "S7_BT_IN"
    S8_BT_IN = "S8_BT_IN"
    S9_BT_IN = "S9_BT_IN"
