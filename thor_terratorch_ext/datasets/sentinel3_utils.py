"""
Sentinel-3 preprocessing utilities.

Provides radiance-to-reflectance conversion for OLCI and SLSTR, and other helpers.

**Example pipeline** (scale factor already applied)::

    from thor_terratorch_ext.datasets.sentinel3_utils import (
        radiance_to_reflectance, compute_sza,
        OLCI_SOLAR_FLUX, SLSTR_SOLAR_FLUX,
    )
    cos_sza = np.cos(np.deg2rad(compute_sza(lat, lon, scene_datetime)))
    refl = radiance_to_reflectance(radiance_2d, OLCI_SOLAR_FLUX[band_idx], cos_sza)
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    import datetime

    from numpy.typing import NDArray

# ── Band constants ─────────────────────────────────────────────────────────────

#: Mean solar spectral irradiance for OLCI bands Oa01-Oa21, in mW m⁻² nm⁻¹.
#: Tabulated from instrument_data.nc (mean over all detectors, typical scene).
OLCI_SOLAR_FLUX: list[float] = [
    1500.96,
    1672.84,
    1849.84,
    1893.60,
    1879.35,
    1758.86,
    1614.05,
    1497.63,
    1463.22,
    1438.04,
    1373.91,
    1239.14,
    1221.16,
    1213.50,
    1202.85,
    1149.11,
    938.91,
    911.24,
    876.70,
    808.90,
    684.75,
]

#: Mean solar spectral irradiance for SLSTR bands S1-S6, in mW m⁻² nm⁻¹.
#: Used as fallback when per-detector irradiance is not available.
SLSTR_SOLAR_FLUX: list[float] = [1798.94, 1488.56, 936.94, 358.14, 240.20, 75.82]


# ── Core radiometry ────────────────────────────────────────────────────────────


def radiance_to_reflectance(
    radiance: NDArray,
    solar_flux: float | NDArray,
    cos_sza: float | NDArray,
    clip: bool = True,
) -> NDArray:
    """Convert TOA radiance to TOA reflectance.

    Uses the standard formula:

        ρ = π · L / (F₀ · cos(SZA))

    Parameters
    ----------
    radiance:
        Radiance values in mW m⁻² sr⁻¹ nm⁻¹.  Any shape.
    solar_flux:
        Solar spectral irradiance in the same units.  Scalar or broadcastable
        array (e.g. per-detector shape (H, W)).
    cos_sza:
        Cosine of the solar zenith angle.  Scalar or broadcastable array.
        Values are internally clamped to [0.01, 1] to avoid near-terminator
        blow-up.
    clip:
        If True (default), clamp the result to [0, 2].

    Returns
    -------
    np.ndarray
        TOA reflectance, same shape as *radiance*, dtype float32.
    """
    cos_sza = np.clip(cos_sza, 0.01, 1.0)
    refl = (np.pi * np.asarray(radiance, dtype=np.float32)) / (solar_flux * cos_sza)
    return np.clip(refl, 0.0, 2.0) if clip else refl


def compute_sza(lat_deg: float, lon_deg: float, dt: datetime.datetime) -> float:
    """Approximate solar zenith angle (degrees) at a point and UTC time.

    Uses a simplified astronomical formula (accurate to ≈ 1°).  Suitable for
    single-point SZA estimates; for spatially-varying SZA over a full scene,
    use the tie-point grids in the SEN3 ancillary files instead (see
    :func:`load_olci_reflectance`).

    Parameters
    ----------
    lat_deg, lon_deg:
        Geographic position in decimal degrees (WGS-84).
    dt:
        UTC observation time.  Timezone-aware or timezone-naive.

    Returns
    -------
    float
        Solar zenith angle in degrees, in [0°, 90°].
    """
    import pandas as pd

    ts = pd.Timestamp(dt)
    if ts.tzinfo is not None:
        ts = ts.tz_convert(None)
    doy = ts.dayofyear
    decl = math.radians(23.45 * math.sin(math.radians(360 / 365 * (doy - 81))))
    utc_h = ts.hour + ts.minute / 60 + ts.second / 3600
    ha = math.radians(15.0 * (utc_h - 12.0 + lon_deg / 15.0))
    lat_r = math.radians(lat_deg)
    cos_sza = math.sin(lat_r) * math.sin(decl) + math.cos(lat_r) * math.cos(
        decl
    ) * math.cos(ha)
    return math.degrees(math.acos(max(0.001, min(1.0, cos_sza))))


def interp_tie_points(tie_arr: NDArray, target_shape: tuple[int, int]) -> NDArray:
    """Bilinearly interpolate a tie-point grid to *target_shape*.

    NaN values in *tie_arr* are filled with nearest-neighbour values before
    interpolation (``RectBivariateSpline`` does not handle NaN).
    """
    from scipy.interpolate import RectBivariateSpline
    from scipy.ndimage import distance_transform_edt

    if np.isnan(tie_arr).any():
        _, idx = distance_transform_edt(np.isnan(tie_arr), return_indices=True)
        tie_arr = tie_arr[tuple(idx)]

    tie_y = np.linspace(0, 1, tie_arr.shape[0])
    tie_x = np.linspace(0, 1, tie_arr.shape[1])
    full_y = np.linspace(0, 1, target_shape[0])
    full_x = np.linspace(0, 1, target_shape[1])

    return RectBivariateSpline(tie_y, tie_x, tie_arr, kx=1, ky=1)(
        full_y, full_x
    ).astype(np.float32)


# ── Geographic utilities ───────────────────────────────────────────────────────


def geo_bbox(
    centre_lat: float,
    centre_lon: float,
    extent_m: float,
) -> tuple[float, float, float, float]:
    """Return a square bounding box centred on a point.

    Uses WGS-84 geodesic distances so the box size is consistent regardless
    of latitude.

    Parameters
    ----------
    centre_lat, centre_lon:
        Centre of the box in decimal degrees.
    extent_m:
        Side length of the box in metres.

    Returns
    -------
    (lat_min, lat_max, lon_min, lon_max) in decimal degrees.
    """
    from pyproj import Geod

    geod = Geod(ellps="WGS84")
    half = extent_m / 2
    _, lat_n, _ = geod.fwd(centre_lon, centre_lat, 0, half)
    _, lat_s, _ = geod.fwd(centre_lon, centre_lat, 180, half)
    lon_e, _, _ = geod.fwd(centre_lon, centre_lat, 90, half)
    lon_w, _, _ = geod.fwd(centre_lon, centre_lat, 270, half)
    return (
        float(min(lat_s, lat_n)),
        float(max(lat_s, lat_n)),
        float(min(lon_w, lon_e)),
        float(max(lon_w, lon_e)),
    )


def geo_crop_slices(
    lat_arr: NDArray,
    lon_arr: NDArray,
    bbox: tuple[float, float, float, float],
    min_size: int = 16,
) -> tuple[slice | None, slice | None]:
    """Find tight row/col slices that enclose a geographic bounding box.

    Parameters
    ----------
    lat_arr, lon_arr:
        2-D latitude and longitude arrays of shape (H, W).
    bbox:
        ``(lat_min, lat_max, lon_min, lon_max)`` in decimal degrees.
    min_size:
        Minimum crop extent in pixels (expanded symmetrically if needed).

    Returns
    -------
    row_slice, col_slice : slice or None
        ``(None, None)`` if no pixels fall inside *bbox*.
    """
    lat_min, lat_max, lon_min, lon_max = bbox
    mask = (
        (lat_arr >= lat_min)
        & (lat_arr <= lat_max)
        & (lon_arr >= lon_min)
        & (lon_arr <= lon_max)
        & np.isfinite(lat_arr)
        & np.isfinite(lon_arr)
    )
    rows, cols = np.where(mask)
    if len(rows) == 0:
        return None, None

    r0, r1 = int(rows.min()), int(rows.max()) + 1
    c0, c1 = int(cols.min()), int(cols.max()) + 1

    if r1 - r0 < min_size:
        mid = (r0 + r1) // 2
        r0, r1 = max(mid - min_size // 2, 0), mid + min_size // 2
    if c1 - c0 < min_size:
        mid = (c0 + c1) // 2
        c0, c1 = max(mid - min_size // 2, 0), mid + min_size // 2

    return slice(r0, r1), slice(c0, c1)


# ── Array utilities ────────────────────────────────────────────────────────────


def fill_nan(arr: NDArray) -> NDArray:
    """Replace NaN values in each band with that band's finite median.

    Parameters
    ----------
    arr:
        Array of shape (C, H, W).

    Returns
    -------
    np.ndarray
        Copy of *arr* with NaN replaced by per-band median (0.0 if all-NaN).
    """
    out = arr.copy()
    for b in range(out.shape[0]):
        finite_median = np.nanmedian(out[b])
        fill_val = 0.0 if np.isnan(finite_median) else float(finite_median)
        out[b] = np.where(np.isnan(out[b]), fill_val, out[b])
    return out
