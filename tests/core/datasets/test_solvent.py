"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.datasets.solvent import (
    _SOLVENT_STATS,
    SOLVENT_DESCRIPTOR_ORDER,
    SOLVENT_DIM,
    _load_raw,
    _recompute_stats,
    get_solvent_vector,
    list_solvents,
    normalize,
)


def test_get_solvent_vector_water():
    vec = get_solvent_vector("water")
    assert vec.shape == (1, SOLVENT_DIM)
    assert vec.dtype == torch.float32
    assert vec[0, -1].item() == 1.0  # solvent-present mask


def test_get_solvent_vector_case_insensitive():
    assert torch.equal(get_solvent_vector("Water"), get_solvent_vector("water"))


@pytest.mark.parametrize("name", [None, "", "vacuum", "gas", "gas_phase"])
def test_get_solvent_vector_vacuum(name):
    vec = get_solvent_vector(name)
    assert vec.shape == (1, SOLVENT_DIM)
    assert torch.count_nonzero(vec) == 0


def test_get_solvent_vector_unknown_strict_raises():
    with pytest.raises(KeyError):
        get_solvent_vector("not_a_real_solvent", strict=True)


def test_get_solvent_vector_unknown_non_strict_is_vacuum():
    vec = get_solvent_vector("not_a_real_solvent", strict=False)
    assert torch.count_nonzero(vec) == 0


def test_solvent_vector_differs_from_vacuum():
    assert not torch.equal(get_solvent_vector("water"), get_solvent_vector(None))


def test_list_solvents():
    solvents = list_solvents()
    assert "water" in solvents
    assert len(solvents) == 179


def test_baked_stats_match_json():
    """The baked _SOLVENT_STATS must stay in sync with solvent_descriptors.json."""
    recomputed = _recompute_stats()
    for name in SOLVENT_DESCRIPTOR_ORDER:
        assert recomputed[name]["scale"] == pytest.approx(
            _SOLVENT_STATS[name]["scale"], abs=1e-5
        )
        assert recomputed[name]["transform"] == _SOLVENT_STATS[name]["transform"]


def test_encoding_is_vacuum_anchored():
    """The physical gas phase maps to exactly 0 in every descriptor channel.

    Vacuum raw values: n=1 (shift1 transform), epsilon=1 (log transform), all
    other descriptors 0 (linear). No mean subtraction may reintroduce an offset.
    """
    vacuum_raw = [
        1.0 if name in ("n", "epsilon") else 0.0 for name in SOLVENT_DESCRIPTOR_ORDER
    ]
    assert normalize(vacuum_raw) == [0.0] * len(SOLVENT_DESCRIPTOR_ORDER)


def test_normalized_columns_are_unit_scale():
    """Every normalized descriptor column has population std 1 (no centering)."""
    solvents = _load_raw()["solvents"]
    cols = [[] for _ in SOLVENT_DESCRIPTOR_ORDER]
    for s in solvents.values():
        raw = [s[name] for name in SOLVENT_DESCRIPTOR_ORDER]
        for i, v in enumerate(normalize(raw)):
            cols[i].append(v)
    for col in cols:
        mean = sum(col) / len(col)
        var = sum((v - mean) ** 2 for v in col) / len(col)
        # The baked scales are stored to ~6 decimals, hence the loose bound.
        assert var == pytest.approx(1.0, abs=1e-4)


def test_water_spot_values():
    """Water lands where transform(raw)/scale says it should."""
    import math

    water = _load_raw()["solvents"]["water"]
    vec = get_solvent_vector("water")[0]
    i_n = SOLVENT_DESCRIPTOR_ORDER.index("n")
    i_eps = SOLVENT_DESCRIPTOR_ORDER.index("epsilon")
    i_gamma = SOLVENT_DESCRIPTOR_ORDER.index("gamma")
    assert vec[i_n].item() == pytest.approx(
        (water["n"] - 1.0) / _SOLVENT_STATS["n"]["scale"], rel=1e-5
    )
    assert vec[i_eps].item() == pytest.approx(
        math.log(water["epsilon"]) / _SOLVENT_STATS["epsilon"]["scale"], rel=1e-5
    )
    assert vec[i_gamma].item() == pytest.approx(
        water["gamma"] / _SOLVENT_STATS["gamma"]["scale"], rel=1e-5
    )


def test_normalize_wrong_length_raises():
    with pytest.raises(ValueError):
        normalize([1.0, 2.0])
