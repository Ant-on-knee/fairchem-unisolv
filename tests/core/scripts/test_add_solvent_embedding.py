"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import pytest
import torch

from fairchem.core.datasets.solvent import SOLVENT_DIM
from fairchem.core.models.uma.escn_md import eSCNMDBackbone
from fairchem.core.scripts.add_solvent_embedding import add_solvent_embedding
from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

SPHERE_CHANNELS = 4
SOLVENT_EMB_HIDDEN = 8


def _build_dataset_embedding_backbone() -> eSCNMDBackbone:
    """A small backbone with charge+spin+dataset mixing (mix_csd == 3*sphere)."""
    torch.manual_seed(0)
    return eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=SPHERE_CHANNELS,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=True,
        dataset_mapping={"omol": "omol"},
        use_solvent_embedding=False,
        always_use_pbc=False,
    )


def _fake_checkpoint() -> MLIPInferenceCheckpoint:
    backbone = _build_dataset_embedding_backbone()
    model_state_dict = {f"backbone.{k}": v for k, v in backbone.state_dict().items()}
    ema_state_dict = {
        f"module.backbone.{k}": v.clone() for k, v in backbone.state_dict().items()
    }
    return MLIPInferenceCheckpoint(
        model_config={"backbone": {"use_dataset_embedding": True}},
        model_state_dict=model_state_dict,
        ema_state_dict=ema_state_dict,
        tasks_config={},
    )


def test_add_solvent_embedding_widens_mix_csd_and_zeros_solvent_columns():
    checkpoint = _fake_checkpoint()
    original_mix = checkpoint.model_state_dict["backbone.mix_csd.weight"].clone()
    assert original_mix.shape == (SPHERE_CHANNELS, 3 * SPHERE_CHANNELS)

    add_solvent_embedding(checkpoint, solvent_emb_hidden=SOLVENT_EMB_HIDDEN)

    new_mix = checkpoint.model_state_dict["backbone.mix_csd.weight"]
    assert new_mix.shape == (SPHERE_CHANNELS, 4 * SPHERE_CHANNELS)
    # Existing charge/spin/dataset columns are preserved...
    assert torch.equal(new_mix[:, : 3 * SPHERE_CHANNELS], original_mix)
    # ...and the trailing solvent columns are zero.
    assert torch.count_nonzero(new_mix[:, 3 * SPHERE_CHANNELS :]) == 0


def test_add_solvent_embedding_injects_weights_into_model_and_ema():
    checkpoint = _fake_checkpoint()
    add_solvent_embedding(checkpoint, solvent_emb_hidden=SOLVENT_EMB_HIDDEN)

    for prefix, sd in (
        ("backbone.", checkpoint.model_state_dict),
        ("module.backbone.", checkpoint.ema_state_dict),
    ):
        assert sd[f"{prefix}mix_csd.weight"].shape == (
            SPHERE_CHANNELS,
            4 * SPHERE_CHANNELS,
        )
        # SolventEmbedding is a 2-layer MLP: net.0 (in->hidden), net.2 (hidden->out).
        in_w = sd[f"{prefix}solvent_embedding.net.0.weight"]
        out_w = sd[f"{prefix}solvent_embedding.net.2.weight"]
        assert in_w.shape == (SOLVENT_EMB_HIDDEN, SOLVENT_DIM)
        assert out_w.shape == (SPHERE_CHANNELS, SOLVENT_EMB_HIDDEN)


def test_add_solvent_embedding_sets_config_flags():
    checkpoint = _fake_checkpoint()
    add_solvent_embedding(checkpoint, solvent_emb_hidden=SOLVENT_EMB_HIDDEN)

    backbone_cfg = checkpoint.model_config["backbone"]
    assert backbone_cfg["use_solvent_embedding"] is True
    assert backbone_cfg["solvent_emb_grad"] is True
    assert backbone_cfg["solvent_emb_hidden"] == SOLVENT_EMB_HIDDEN


def test_add_solvent_embedding_rejects_already_enabled():
    checkpoint = _fake_checkpoint()
    checkpoint.model_config["backbone"]["use_solvent_embedding"] = True
    with pytest.raises(ValueError, match="already has use_solvent_embedding"):
        add_solvent_embedding(checkpoint)


def test_add_solvent_embedding_rejects_missing_dataset_embedding():
    """A checkpoint without dataset mixing has mix_csd == 2*sphere, which solvent
    injection does not support."""
    torch.manual_seed(0)
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=SPHERE_CHANNELS,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        use_solvent_embedding=False,
        always_use_pbc=False,
    )
    checkpoint = MLIPInferenceCheckpoint(
        model_config={"backbone": {"use_dataset_embedding": False}},
        model_state_dict={f"backbone.{k}": v for k, v in backbone.state_dict().items()},
        ema_state_dict=None,
        tasks_config={},
    )
    with pytest.raises(ValueError, match="charge\\+spin\\+dataset"):
        add_solvent_embedding(checkpoint)
