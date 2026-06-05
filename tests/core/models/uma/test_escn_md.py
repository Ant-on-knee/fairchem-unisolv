"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import logging

import pytest
import torch
from ase import Atoms
from ase.build import molecule as get_molecule
from e3nn.math import direct_sum
from e3nn.o3 import matrix_to_angles, spherical_harmonics, wigner_D

from fairchem.core.datasets.atomic_data import AtomicData
from fairchem.core.models.uma.escn_md import eSCNMDBackbone, resolve_dataset_mapping


@pytest.mark.parametrize(
    "atoms, orthogonal_direction",
    [
        (get_molecule("Be2"), torch.eye(3, dtype=torch.float32)[:2, :]),
        (
            Atoms(symbols="C3", positions=[[0, 0, 0], [0, 0, 1], [0, 0, 2]]),
            torch.eye(3, dtype=torch.float32)[:2, :],
        ),
    ],
)
def test_escnmd_backbone_impossible_vectors(
    atoms: Atoms, orthogonal_direction: torch.Tensor
) -> None:
    # TODO test could be improved by randomly rotating input and orthogonal directions
    torch.manual_seed(42)
    lmax = 2
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=lmax,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    g = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=25,
        radius=12,
        task_name="diatomic_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    out = backbone(g)

    # L=1 -> orthogonality as in R^3
    sh1 = spherical_harmonics(1, orthogonal_direction, normalize=True)
    l1 = out["node_embedding"][:, 1:4]
    orthogonal = torch.einsum(
        "...ijk,...j->...ijk", l1, sh1
    )  # n_ortho_directions, edges, 3, channels
    orthogonal = orthogonal.norm(dim=2)  # n_ortho_directions, edges, channels
    assert torch.allclose(
        orthogonal, torch.zeros_like(orthogonal), atol=1e-6
    ), "Orthogonal directions should be zero"


@pytest.mark.parametrize(
    "atoms, symmetry_matrix",
    [
        (
            Atoms(
                symbols="C4",
                positions=[
                    [-0.5, -0.5, 0.0],
                    [0.5, -0.5, 0.0],
                    [-0.5, 0.5, 0.0],
                    [0.5, 0.5, 0.0],
                ],
            ),
            torch.tensor([[0, -1, 0], [1, 0, 0], [0, 0, 1]], dtype=torch.float32),
        ),
    ],
)
def test_escnmd_backbone_symmetries(
    atoms: Atoms, symmetry_matrix: torch.Tensor
) -> None:
    torch.manual_seed(42)
    lmax = 2
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=lmax,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    g0 = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=25,
        radius=12,
        task_name="diatomic_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    out0 = backbone(g0)

    atoms1 = atoms.copy()
    atoms1.positions = atoms1.positions @ torch.linalg.inv(symmetry_matrix).T.numpy()
    g1 = AtomicData.from_ase(
        input_atoms=atoms1,
        max_neigh=25,
        radius=12,
        task_name="diatomic_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    out1 = backbone(g1)

    wigner_d = direct_sum(
        *[wigner_D(l, *matrix_to_angles(symmetry_matrix)) for l in range(lmax + 1)]
    )
    out1["node_embedding"] = torch.einsum(
        "aj,ijk->iak", wigner_d, out1["node_embedding"]
    )

    assert (
        (out0["node_embedding"] - out1["node_embedding"]).abs().max() < 5e-4
    ), f"For this molecule {atoms.positions=}, node embeddings should be invariant under this symmetry transformation {symmetry_matrix=}."  # high tolerance due to low precision


def test_escnmd_backbone_solvent_disabled_is_unchanged():
    """With use_solvent_embedding=False (default) the backbone is structurally
    identical to before: no solvent module, mix_csd unchanged."""
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )
    assert not hasattr(backbone, "solvent_embedding")
    assert backbone.mix_csd.in_features == 2 * backbone.sphere_channels


def test_escnmd_backbone_with_solvent_embedding():
    """A backbone with use_solvent_embedding=True runs a forward pass and
    widens mix_csd to include the solvent term."""
    torch.manual_seed(42)
    backbone = eSCNMDBackbone(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        use_solvent_embedding=True,
        solvent_emb_hidden=8,
        always_use_pbc=False,
    )
    assert hasattr(backbone, "solvent_embedding")
    assert backbone.mix_csd.in_features == 3 * backbone.sphere_channels

    atoms = get_molecule("H2O")
    atoms.info["solvent"] = "water"
    g = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=25,
        radius=12,
        task_name="solvent_test",
        r_edges=False,
        r_data_keys=["spin", "charge", "solvent"],
    )
    out = backbone(g)
    assert out["node_embedding"].shape[0] == len(atoms)
    assert torch.isfinite(out["node_embedding"]).all()


def test_solvent_surgery_preserves_predictions():
    """Transplanting a solvent-free backbone's weights into a solvent-enabled
    one via the surgery's mix_csd widening (zeroed solvent columns) leaves the
    forward pass numerically unchanged -- the property add_solvent_embedding
    relies on to upgrade pretrained checkpoints without changing predictions."""
    from fairchem.core.scripts.add_solvent_embedding import _expand_mix_csd_weight

    kwargs = dict(
        max_num_elements=100,
        sphere_channels=4,
        lmax=2,
        mmax=2,
        otf_graph=True,
        edge_channels=5,
        num_distance_basis=7,
        use_dataset_embedding=False,
        always_use_pbc=False,
    )

    torch.manual_seed(0)
    ref = eSCNMDBackbone(use_solvent_embedding=False, **kwargs).eval()

    solv = eSCNMDBackbone(
        use_solvent_embedding=True, solvent_emb_hidden=8, **kwargs
    ).eval()

    # Transplant ref's weights into solv: widen mix_csd (zero solvent columns),
    # keep solv's freshly-initialized solvent_embedding.
    new_sd = dict(ref.state_dict())
    new_sd["mix_csd.weight"] = _expand_mix_csd_weight(
        ref.state_dict()["mix_csd.weight"], ref.sphere_channels
    )
    new_sd.update(
        {
            key: value
            for key, value in solv.state_dict().items()
            if key.startswith("solvent_embedding.")
        }
    )
    solv.load_state_dict(new_sd)

    atoms = get_molecule("H2O")
    g_ref = AtomicData.from_ase(
        input_atoms=atoms,
        max_neigh=25,
        radius=12,
        task_name="solvent_test",
        r_edges=False,
        r_data_keys=["spin", "charge"],
    )
    atoms_solv = atoms.copy()
    atoms_solv.info["solvent"] = "water"
    g_solv = AtomicData.from_ase(
        input_atoms=atoms_solv,
        max_neigh=25,
        radius=12,
        task_name="solvent_test",
        r_edges=False,
        r_data_keys=["spin", "charge", "solvent"],
    )

    out_ref = ref(g_ref)["node_embedding"]
    out_solv = solv(g_solv)["node_embedding"]
    assert torch.allclose(out_ref, out_solv, atol=1e-6)


def test_resolve_dataset_mapping_valid_mapping():
    mapping = {"oc20": "oc20", "oc20_subset": "oc20"}
    result = resolve_dataset_mapping(deprecated_list=None, dataset_mapping=mapping)
    assert result == {"oc20": "oc20", "oc20_subset": "oc20"}


def test_resolve_dataset_mapping_deprecated_list(caplog):
    with caplog.at_level(logging.WARNING):
        result = resolve_dataset_mapping(
            deprecated_list=["omol", "omat"], dataset_mapping=None
        )
    assert result == {"omol": "omol", "omat": "omat"}
    assert "deprecated" in caplog.text.lower()


def test_resolve_dataset_mapping_both_raises():
    with pytest.raises(ValueError, match="Both"):
        resolve_dataset_mapping(
            deprecated_list=["oc20"], dataset_mapping={"oc20": "oc20"}
        )


def test_resolve_dataset_mapping_neither_raises():
    with pytest.raises(ValueError, match="dataset_mapping"):
        resolve_dataset_mapping(deprecated_list=None, dataset_mapping=None)


def test_resolve_dataset_mapping_empty_dict_raises():
    with pytest.raises(ValueError, match="non-empty"):
        resolve_dataset_mapping(deprecated_list=None, dataset_mapping={})


def test_resolve_dataset_mapping_values_not_subset_raises():
    with pytest.raises(ValueError, match="subset"):
        resolve_dataset_mapping(
            deprecated_list=None,
            dataset_mapping={"oc20_subset1": "oc20", "oc20subset2": "oc20"},
        )


def test_resolve_dataset_mapping_custom_param_name(caplog):
    with caplog.at_level(logging.WARNING):
        resolve_dataset_mapping(
            deprecated_list=["omol"],
            dataset_mapping=None,
            deprecated_param_name="dataset_names",
        )
    assert "dataset_names" in caplog.text
