"""Tests for thor_terratorch_ext/models/backbones/thor_vit.py.

All THOREncoderWrapper tests go through load_thor_model to exercise the real
model construction path. Results are asserted against the wrapper's own
properties (e.g. wrapper.single_embedding_shape) so they remain correct even
if the architecture changes.
"""

import pytest
import torch

from thor_terratorch_ext.datasets.utils import (
    S2L2ABands,
    S3OLCIBands,
    S3SLSTRBands,
    SARThorBands,
    ThorModalities,
)
from thor_terratorch_ext.models.backbones.thor_vit import (
    _ensure_allowed_input_params,
    _normalize_patch_sizes,
    _resolve_band_key,
    _to_internal_band_name,
    THOREncoderWrapper,
    bands_from_modalities,
    load_thor_model,
    process_thor_bands,
)

# ---------------------------------------------------------------------------
# Wrapper factory with a per-session cache so the THOR model is only built
# once per (bands_key, merge_method) combination.
# ---------------------------------------------------------------------------

_BAND_MAP: dict[str, list] = {
    "s2": list(S2L2ABands),
    "s2_blue_only": [S2L2ABands.BLUE],
    "s1": [SARThorBands.IW_VV, SARThorBands.IW_VH],
}

_wrapper_cache: dict[tuple, THOREncoderWrapper] = {}


def _make_wrapper(
    bands_key: str, merge_method: str | None = None
) -> THOREncoderWrapper:
    key = (bands_key, merge_method)
    if key not in _wrapper_cache:
        _wrapper_cache[key] = load_thor_model(
            "thor_v1_tiny",
            model_bands=_BAND_MAP[bands_key],
            pretrained=False,
            merge_method=merge_method,
        )
    return _wrapper_cache[key]


# ---------------------------------------------------------------------------
# Helpers for _merge_tokens_to_image_features
# ---------------------------------------------------------------------------


def _make_channel_params(wrapper: THOREncoderWrapper, num_patch: int) -> dict:
    """Uniform num_patch for every band in every group."""
    return {
        member: {"num_patch": num_patch}
        for group in wrapper.groups.values()
        for member in group
    }


def _make_tokens(
    wrapper: THOREncoderWrapper, num_patch: int, batch: int = 2
) -> torch.Tensor:
    n_groups = len(wrapper.groups)
    return torch.randn(batch, n_groups * num_patch**2, wrapper.single_embedding_shape)


# ---------------------------------------------------------------------------
# _ensure_allowed_input_params
# ---------------------------------------------------------------------------


class TestEnsureAllowedInputParams:
    def test_allowed_keys_pass(self):
        _ensure_allowed_input_params(["ground_covers"])
        _ensure_allowed_input_params(
            [
                "flexivit_patch_size_seqs",
                "flexivit_ref_patch_size",
                "select_patch_strategy",
            ]
        )

    def test_empty_keys_pass(self):
        _ensure_allowed_input_params([])

    def test_disallowed_key_raises_value_error(self):
        with pytest.raises(ValueError, match="Cannot override"):
            _ensure_allowed_input_params(["unknown_param"])

    def test_error_names_the_bad_key(self):
        with pytest.raises(ValueError, match="totally_wrong"):
            _ensure_allowed_input_params(["totally_wrong"])

    def test_mix_of_good_and_bad_raises(self):
        with pytest.raises(ValueError):
            _ensure_allowed_input_params(["ground_covers", "not_allowed"])


# ---------------------------------------------------------------------------
# process_thor_bands
# ---------------------------------------------------------------------------


class TestProcessThorBands:
    def test_s2_enum_bands(self):
        thor_bands, _ = process_thor_bands([S2L2ABands.BLUE, S2L2ABands.RED])
        assert thor_bands == ["S2:Blue", "S2:Red"]

    def test_s2_string_bands(self):
        thor_bands, _ = process_thor_bands(["BLUE", "RED", "NIR_BROAD"])
        assert thor_bands == ["S2:Blue", "S2:Red", "S2:NIR"]

    def test_sar_enum_bands(self):
        thor_bands, _ = process_thor_bands([SARThorBands.IW_VV, SARThorBands.IW_VH])
        assert thor_bands == ["S1:IW-VV", "S1:IW-VH"]

    def test_s3_olci_band(self):
        thor_bands, _ = process_thor_bands([S3OLCIBands.OA01_REFLECTANCE])
        assert thor_bands == ["S3:Oa01_reflectance"]

    def test_s3_slstr_bt_band(self):
        thor_bands, _ = process_thor_bands([S3SLSTRBands.S7_BT_IN])
        assert thor_bands == ["S3:S7_BT_in"]

    def test_sar_default_gsd_suffix_no_param_change(self):
        """IW_VV_10 is the default GSD; channel params should not change."""
        thor_bands, channel_params = process_thor_bands(["IW_VV_10"])
        assert "S1:IW-VV" in thor_bands
        assert channel_params["S1:IW-VV"]["GSD"] == 10

    def test_sar_non_default_gsd_suffix_updates_param(self):
        """IW_VV_60 overrides GSD to 60 in channel params."""
        _, channel_params = process_thor_bands(["IW_VV_60"])
        assert channel_params["S1:IW-VV"]["GSD"] == 60

    def test_sar_gsd_suffix_updates_whole_group(self):
        """Changing the GSD on one SAR band must propagate to all same-group members."""
        _, channel_params = process_thor_bands(["IW_VV_60"])
        # IW_VV and IW_VH are in the same default group
        assert channel_params["S1:IW-VH"]["GSD"] == 60

    def test_unknown_band_raises_not_implemented(self):
        with pytest.raises(NotImplementedError, match="not implemented in THOR"):
            process_thor_bands(["TOTALLY_UNKNOWN_BAND"])

    def test_duplicate_bands_raises_value_error(self):
        with pytest.raises(ValueError, match="Duplicate"):
            process_thor_bands([SARThorBands.IW_VV, SARThorBands.IW_VV])

    def test_returns_correct_types(self):
        thor_bands, channel_params = process_thor_bands([S2L2ABands.BLUE])
        assert isinstance(thor_bands, list)
        assert isinstance(channel_params, dict)
        assert all(isinstance(b, str) for b in thor_bands)

    def test_channel_params_has_gsd_and_patch_size(self):
        _, channel_params = process_thor_bands([S2L2ABands.BLUE])
        assert "GSD" in channel_params["S2:Blue"]
        assert "patch_size" in channel_params["S2:Blue"]


# ---------------------------------------------------------------------------
# bands_from_modalities
# ---------------------------------------------------------------------------


class TestBandsFromModalitiesListForm:
    def test_s2_all_bands(self):
        result = bands_from_modalities([ThorModalities.S2L2A])
        assert set(result) == set(S2L2ABands)
        assert len(result) == 12

    def test_s1grd(self):
        result = bands_from_modalities([ThorModalities.S1GRD])
        assert list(result) == [SARThorBands.IW_VV, SARThorBands.IW_VH]

    def test_s3_shorthand(self):
        """ThorModalities.S3 must be usable after the mapping fix."""
        result = bands_from_modalities([ThorModalities.S3])
        assert len(result) == 30  # 21 OLCI + 9 SLSTR

    def test_deduplicates_across_overlapping_modalities(self):
        # S1GRD and S1GRD_VV_VH both expand to IW_VV + IW_VH
        result = bands_from_modalities(
            [
                ThorModalities.S1GRD,
                ThorModalities.S1GRD_VV_VH,
            ]
        )
        assert len(result) == 2
        assert len(result) == len(set(result))

    def test_string_modality_key(self):
        result = bands_from_modalities(["S2L2A"])
        assert set(result) == set(S2L2ABands)

    def test_invalid_modality_raises(self):
        with pytest.raises(ValueError, match="Invalid modality"):
            bands_from_modalities(["INVALID_MODALITY"])

    def test_multiple_modalities_combined(self):
        result = bands_from_modalities([ThorModalities.S2L2A, ThorModalities.S1GRD])
        assert set(result) == set(S2L2ABands) | {SARThorBands.IW_VV, SARThorBands.IW_VH}


class TestBandsFromModalitiesDictForm:
    def test_subset_is_respected(self):
        """Bug fix: only the specified subset should appear, not all modality bands."""
        subset = [S2L2ABands.RED, S2L2ABands.GREEN, S2L2ABands.BLUE]
        result = bands_from_modalities({ThorModalities.S2L2A: subset})
        assert list(result) == subset
        # Bands not in the subset must be absent
        assert S2L2ABands.NIR_BROAD not in result
        assert S2L2ABands.SWIR_1 not in result
        assert S2L2ABands.SWIR_2 not in result

    def test_subset_ordering_preserved(self):
        """Output order must match the subset order, not the canonical modality order."""
        subset = [S2L2ABands.SWIR_2, S2L2ABands.BLUE, S2L2ABands.RED]
        result = bands_from_modalities({ThorModalities.S2L2A: subset})
        assert list(result) == subset

    def test_single_band_subset(self):
        result = bands_from_modalities({ThorModalities.S1GRD: [SARThorBands.IW_VV]})
        assert list(result) == [SARThorBands.IW_VV]

    def test_band_not_in_modality_raises(self):
        with pytest.raises(ValueError, match="not part of modality"):
            bands_from_modalities({ThorModalities.S1GRD: [S2L2ABands.BLUE]})

    def test_invalid_modality_raises(self):
        with pytest.raises(ValueError, match="Invalid modality"):
            bands_from_modalities({"BAD_MODALITY": [SARThorBands.IW_VV]})

    def test_deduplicates_across_modalities(self):
        refl_bands = list(S3SLSTRBands)[:2]
        result = bands_from_modalities(
            {
                ThorModalities.S3SLSTR_REFL: refl_bands,
                ThorModalities.S3SLSTR: refl_bands,
            }
        )
        assert len(result) == len(set(result))
        assert len(result) == 2

    def test_string_modality_key(self):
        result = bands_from_modalities({"S1GRD": [SARThorBands.IW_VV]})
        assert list(result) == [SARThorBands.IW_VV]


# ---------------------------------------------------------------------------
# THOREncoderWrapper — __init__ and properties  (uses real THOR tiny model)
# ---------------------------------------------------------------------------


class TestTHOREncoderWrapperInit:
    def test_basic_init_s2(self):
        wrapper = _make_wrapper("s2")
        assert len(wrapper.bands) == 12
        assert "S2:Blue" in wrapper.bands
        assert "S2:Red" in wrapper.bands
        assert wrapper.single_embedding_shape > 0

    def test_basic_init_s1(self):
        wrapper = _make_wrapper("s1")
        assert "S1:IW-VV" in wrapper.bands
        assert "S1:IW-VH" in wrapper.bands

    def test_out_indices_default_is_all_blocks(self):
        wrapper = _make_wrapper("s2")
        num_blocks = len(wrapper.model.blocks)
        assert num_blocks > 0
        assert wrapper.out_indices == list(range(num_blocks))

    def test_out_indices_custom(self):
        wrapper = load_thor_model(
            "thor_v1_tiny",
            model_bands=list(S2L2ABands),
            pretrained=False,
            out_indices=[0, 2],
        )
        assert wrapper.out_indices == [0, 2]

    def test_invalid_merge_method_raises(self):
        with pytest.raises(ValueError, match="Unknown merge_method"):
            load_thor_model(
                "thor_v1_tiny",
                model_bands=list(S2L2ABands),
                pretrained=False,
                merge_method="invalid",
            )

    def test_out_channels_concat(self):
        wrapper = _make_wrapper("s2", merge_method="concat")
        n_groups = len(wrapper.groups)
        assert all(
            c == wrapper.single_embedding_shape * n_groups for c in wrapper.out_channels
        )

    def test_out_channels_non_concat(self):
        for method in ("sum", "mean"):
            wrapper = _make_wrapper("s2", merge_method=method)
            assert all(
                c == wrapper.single_embedding_shape for c in wrapper.out_channels
            )

    def test_lowest_gsd_s2(self):
        # S2L2A has 10m, 20m, 60m bands → lowest is 10
        wrapper = _make_wrapper("s2")
        assert wrapper.lowest_gsd == 10

    def test_lowest_gsd_s1(self):
        # S1 IW bands default to 10m
        wrapper = _make_wrapper("s1")
        assert wrapper.lowest_gsd == 10

    def test_input_size(self):
        wrapper = _make_wrapper("s2")
        assert wrapper.input_size == wrapper.ground_cover // wrapper.lowest_gsd

    def test_s2_has_three_groups(self):
        # S2L2A splits into three THOR groups: 10m / 20m / 60m
        wrapper = _make_wrapper("s2")
        assert len(wrapper.groups) == 3

    def test_s1_has_one_group(self):
        wrapper = _make_wrapper("s1")
        assert len(wrapper.groups) == 1


# ---------------------------------------------------------------------------
# THOREncoderWrapper._preprocess_input
# ---------------------------------------------------------------------------


class TestPreprocessInput:
    def test_tensor_input_returns_dict_with_all_bands(self):
        wrapper = _make_wrapper("s2")
        x = torch.zeros(2, 12, 16, 16)
        result = wrapper._preprocess_input(x)
        assert isinstance(result, dict)
        assert len(result) == 12

    def test_tensor_input_interpolates_s2_10m(self):
        wrapper = _make_wrapper("s2")
        x = torch.zeros(2, 12, 16, 16)
        result = wrapper._preprocess_input(x)
        expected_size = int(wrapper.ground_cover / wrapper.channels["S2:Blue"]["GSD"])
        b, c, h, w = result["S2:Blue"].shape
        assert (h, w) == (expected_size, expected_size)
        assert (b, c) == (2, 1)

    def test_tensor_input_interpolates_s2_60m(self):
        """60m bands must be interpolated to a smaller target than 10m bands."""
        wrapper = _make_wrapper("s2")
        x = torch.zeros(2, 12, 16, 16)
        result = wrapper._preprocess_input(x)
        size_10m = int(wrapper.ground_cover / wrapper.channels["S2:Blue"]["GSD"])
        size_60m = int(
            wrapper.ground_cover / wrapper.channels["S2:CoastAerosal"]["GSD"]
        )
        assert size_60m < size_10m
        assert result["S2:CoastAerosal"].shape[-1] == size_60m

    def test_dict_input_s1grd(self):
        wrapper = _make_wrapper("s1")
        x = {"S1GRD": torch.zeros(2, 2, 16, 16)}
        result = wrapper._preprocess_input(x)
        assert set(result.keys()) == {"S1:IW-VV", "S1:IW-VH"}

    def test_dict_input_skips_bands_not_in_model(self):
        """Only the band the model knows about should appear in the result."""
        wrapper = _make_wrapper("s2_blue_only")  # single-band model
        x = {"S2L2A": torch.zeros(2, 12, 32, 32)}
        result = wrapper._preprocess_input(x)
        assert set(result.keys()) == {"S2:Blue"}

    def test_dict_input_invalid_modality_raises(self):
        wrapper = _make_wrapper("s2")
        with pytest.raises(ValueError, match="Invalid modality key"):
            wrapper._preprocess_input({"NOT_A_MODALITY": torch.zeros(2, 1, 16, 16)})

    def test_dict_input_no_matching_bands_raises(self):
        """If none of the modality bands are in the model, should raise."""
        wrapper = _make_wrapper("s1")  # S1-only model
        with pytest.raises(ValueError, match="No valid bands"):
            wrapper._preprocess_input({"S2L2A": torch.zeros(2, 12, 16, 16)})

    def test_dict_input_too_few_channels_raises(self):
        """Tensor with fewer channels than expected for the modality should raise."""
        wrapper = _make_wrapper("s1")
        # S1GRD expects ch0=VV, ch1=VH; provide only 1 channel
        with pytest.raises(ValueError, match="channels"):
            wrapper._preprocess_input({"S1GRD": torch.zeros(2, 1, 16, 16)})


# ---------------------------------------------------------------------------
# THOREncoderWrapper._merge_tokens_to_image_features  (bug-fix tests)
# ---------------------------------------------------------------------------


class TestMergeTokensToImageFeatures:
    """Verify the routing fix: all groups must be included for sum/mean/concat,
    even when they are at the highest resolution (num_patch == highest_num_patch).

    The old bug: `if num_patch != highest_num_patch and method != "group"` sent
    highest-resolution groups to `grouped_tokens`, which is only consumed by the
    "group" method — silently dropping them for sum/mean/concat.
    """

    def test_concat_includes_all_groups_at_same_resolution(self):
        """All 3 S2 groups at equal num_patch — none should be dropped."""
        wrapper = _make_wrapper("s2", merge_method="concat")
        n_groups = len(wrapper.groups)
        cp = _make_channel_params(wrapper, num_patch=4)
        tokens = _make_tokens(wrapper, num_patch=4)

        result = wrapper._merge_tokens_to_image_features([tokens], cp)
        assert len(result) == 1
        assert result[0].shape[1] == n_groups * wrapper.single_embedding_shape

    def test_sum_includes_all_groups_at_same_resolution(self):
        wrapper = _make_wrapper("s2", merge_method="sum")
        cp = _make_channel_params(wrapper, num_patch=4)
        tokens = _make_tokens(wrapper, num_patch=4)

        result = wrapper._merge_tokens_to_image_features([tokens], cp)
        assert len(result) == 1
        assert result[0].shape[1] == wrapper.single_embedding_shape

    def test_mean_includes_all_groups_at_same_resolution(self):
        wrapper = _make_wrapper("s2", merge_method="mean")
        cp = _make_channel_params(wrapper, num_patch=4)
        tokens = _make_tokens(wrapper, num_patch=4)

        result = wrapper._merge_tokens_to_image_features([tokens], cp)
        assert len(result) == 1
        assert result[0].shape[1] == wrapper.single_embedding_shape

    def test_group_returns_per_group_dict(self):
        wrapper = _make_wrapper("s2", merge_method="group")
        n_groups = len(wrapper.groups)
        cp = _make_channel_params(wrapper, num_patch=4)
        tokens = _make_tokens(wrapper, num_patch=4)

        result = wrapper._merge_tokens_to_image_features([tokens], cp)
        assert len(result) == 1
        assert isinstance(result[0], dict)
        assert len(result[0]) == n_groups

    def test_concat_output_spatial_size(self):
        num_patch = 4
        wrapper = _make_wrapper("s2", merge_method="concat")
        n_groups = len(wrapper.groups)
        cp = _make_channel_params(wrapper, num_patch=num_patch)
        tokens = _make_tokens(wrapper, num_patch=num_patch)

        result = wrapper._merge_tokens_to_image_features([tokens], cp)
        assert result[0].shape == (
            2,
            n_groups * wrapper.single_embedding_shape,
            num_patch,
            num_patch,
        )

    def test_concat_upsample_mixed_patch_sizes(self):
        """Lower-resolution groups must be upsampled to match the largest num_patch."""
        wrapper = _make_wrapper("s2", merge_method="concat")
        assert len(wrapper.groups) == 3, "expected 3 S2 groups (10m / 20m / 60m)"

        # Assign decreasing num_patch across the three groups
        num_patches = [8, 4, 2]
        channel_params: dict = {}
        total_tokens = 0
        for group_members, np_ in zip(wrapper.groups.values(), num_patches):
            for member in group_members:
                channel_params[member] = {"num_patch": np_}
            total_tokens += np_**2

        tokens = torch.randn(2, total_tokens, wrapper.single_embedding_shape)
        result = wrapper._merge_tokens_to_image_features([tokens], channel_params)

        max_patch = max(num_patches)
        n_groups = len(wrapper.groups)
        assert result[0].shape == (
            2,
            n_groups * wrapper.single_embedding_shape,
            max_patch,
            max_patch,
        )

    def test_processes_multiple_feature_maps(self):
        """One result tensor per input feature map."""
        wrapper = _make_wrapper("s2", merge_method="sum")
        cp = _make_channel_params(wrapper, num_patch=4)
        tokens = _make_tokens(wrapper, num_patch=4)

        result = wrapper._merge_tokens_to_image_features([tokens, tokens.clone()], cp)
        assert len(result) == 2


# ---------------------------------------------------------------------------
# _to_internal_band_name
# ---------------------------------------------------------------------------


class TestToInternalBandName:
    def test_band_enum_value(self):
        assert _to_internal_band_name("BLUE") == "S2:Blue"
        assert _to_internal_band_name("IW_VV") == "S1:IW-VV"

    def test_alias(self):
        assert _to_internal_band_name("VV") == "S1:IW-VV"
        assert _to_internal_band_name("ASC_VV") == "S1:IW-VV"

    def test_internal_name_passthrough(self):
        assert _to_internal_band_name("S2:Blue") == "S2:Blue"
        assert _to_internal_band_name("S3:Oa01_reflectance") == "S3:Oa01_reflectance"

    def test_unknown_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            _to_internal_band_name("NOT_A_BAND")


# ---------------------------------------------------------------------------
# _resolve_band_key
# ---------------------------------------------------------------------------


class TestResolveBandKey:
    def test_modality_key_expands(self):
        """A ThorModalities value should expand to all bands of that modality."""
        result = _resolve_band_key("S2L2A")
        assert len(result) == 12
        assert "S2:Blue" in result

    def test_sar_modality_expands_to_all_sar_bands(self):
        """S1GRD should expand to ALL 8 SAR internal bands (IW+EW, all pols)."""
        result = _resolve_band_key("S1GRD")
        assert len(result) == 8
        for name in [
            "S1:IW-VV",
            "S1:IW-VH",
            "S1:IW-HV",
            "S1:IW-HH",
            "S1:EW-VV",
            "S1:EW-VH",
            "S1:EW-HV",
            "S1:EW-HH",
        ]:
            assert name in result, f"{name} missing from S1GRD expansion"

    def test_sar_vv_vh_modality_expands_all(self):
        result = _resolve_band_key("S1GRD_VV_VH")
        assert len(result) == 8

    def test_sar_hh_hv_modality_expands_all(self):
        result = _resolve_band_key("S1GRD_HH_HV")
        assert len(result) == 8

    def test_band_enum_value(self):
        assert _resolve_band_key("BLUE") == ["S2:Blue"]
        assert _resolve_band_key("IW_VV") == ["S1:IW-VV"]

    def test_internal_name_passthrough(self):
        assert _resolve_band_key("S2:Blue") == ["S2:Blue"]
        assert _resolve_band_key("S1:IW-VV") == ["S1:IW-VV"]

    def test_alias_band_name(self):
        assert _resolve_band_key("VV") == ["S1:IW-VV"]
        assert _resolve_band_key("ASC_VV") == ["S1:IW-VV"]

    def test_s3_modality(self):
        result = _resolve_band_key("S3OLCI")
        assert len(result) == 21
        assert "S3:Oa01_reflectance" in result

    def test_invalid_key_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            _resolve_band_key("TOTALLY_INVALID_KEY")


# ---------------------------------------------------------------------------
# _normalize_patch_sizes
# ---------------------------------------------------------------------------


class TestNormalizePatchSizes:
    def test_int_input(self):
        assert _normalize_patch_sizes(8) == [8]

    def test_list_input(self):
        assert _normalize_patch_sizes([4, 8, 16]) == [4, 8, 16]

    def test_dict_with_modality_keys(self):
        result = _normalize_patch_sizes({"S2L2A": [4, 8], "S1GRD": 16})
        assert isinstance(result, dict)
        # S2L2A expands to 12 bands, S1GRD to all 8 SAR bands
        assert result["S2:Blue"] == [4, 8]
        assert result["S2:Red"] == [4, 8]
        assert result["S1:IW-VV"] == [16]
        assert result["S1:IW-VH"] == [16]
        assert result["S1:EW-VV"] == [16]
        assert result["S1:EW-HH"] == [16]
        assert len([k for k in result if k.startswith("S1:")]) == 8

    def test_dict_with_band_enum_keys(self):
        result = _normalize_patch_sizes({"BLUE": 4, "RED": [8, 16]})
        assert isinstance(result, dict)
        assert result["S2:Blue"] == [4]
        assert result["S2:Red"] == [8, 16]

    def test_dict_with_internal_keys(self):
        result = _normalize_patch_sizes({"S2:Blue": [4, 8]})
        assert isinstance(result, dict)
        assert result["S2:Blue"] == [4, 8]

    def test_dict_int_value_wrapped_in_list(self):
        result = _normalize_patch_sizes({"BLUE": 8})
        assert result["S2:Blue"] == [8]

    def test_invalid_type_raises(self):
        with pytest.raises(TypeError, match="patch_sizes must be"):
            _normalize_patch_sizes("invalid")  # type: ignore[arg-type]

    def test_invalid_dict_key_raises(self):
        with pytest.raises(ValueError, match="Cannot resolve"):
            _normalize_patch_sizes({"TOTALLY_INVALID": [8]})


# ---------------------------------------------------------------------------
# load_thor_model — new kwargs and deprecation
# ---------------------------------------------------------------------------


class TestLoadThorModelNewKwargs:
    def test_input_params_emits_deprecation_warning(self):
        with pytest.warns(DeprecationWarning, match="input_params.*deprecated"):
            load_thor_model(
                "thor_v1_tiny",
                model_bands=[S2L2ABands.BLUE],
                pretrained=False,
                input_params={"ground_covers": [1000]},
            )

    def test_ground_cover_kwarg(self):
        wrapper = load_thor_model(
            "thor_v1_tiny",
            model_bands=list(S2L2ABands),
            pretrained=False,
            ground_cover=1000,
        )
        assert wrapper.ground_cover == 1000

    def test_select_patch_strategy_kwarg(self):
        # Just ensure it doesn't raise
        load_thor_model(
            "thor_v1_tiny",
            model_bands=list(S2L2ABands),
            pretrained=False,
            select_patch_strategy="max",
        )

    def test_new_kwargs_override_deprecated_input_params(self):
        """New top-level kwargs should take priority over deprecated input_params."""
        with pytest.warns(DeprecationWarning):
            wrapper = load_thor_model(
                "thor_v1_tiny",
                model_bands=list(S2L2ABands),
                pretrained=False,
                input_params={"ground_covers": [5000]},
                ground_cover=1000,
            )
        assert wrapper.ground_cover == 1000
