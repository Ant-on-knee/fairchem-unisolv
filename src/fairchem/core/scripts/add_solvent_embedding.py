"""
Copyright (c) Meta Platforms, Inc. and affiliates.

This source code is licensed under the MIT license found in the
LICENSE file in the root directory of this source tree.
"""

from __future__ import annotations

import argparse
import logging
from typing import TYPE_CHECKING

import numpy as np
import torch

from fairchem.core.datasets.solvent import SOLVENT_DIM
from fairchem.core.models.uma.nn.embedding import SolventEmbedding

if TYPE_CHECKING:
    from fairchem.core.units.mlip_unit.api.inference import MLIPInferenceCheckpoint

# Default solvent-embedding hidden width. This mirrors the K4L2 / K10L4 backbone
# configs and MUST match ``solvent_emb_hidden`` in the finetuning config so the
# rebuilt module and the injected weights agree.
DEFAULT_SOLVENT_EMB_HIDDEN = 16


def _expand_mix_csd_weight(weight: torch.Tensor, sphere_channels: int) -> torch.Tensor:
    """Widen a mix_csd weight to make room for the solvent term.

    The solvent embedding is concatenated last in ``eSCNMDBackbone.csd_embedding``
    so the existing charge/spin/dataset columns keep their positions. The new
    solvent columns are zero-initialized: combined with the near-zero init of the
    solvent embedding's output layer, the upgraded model reproduces the original
    predictions before any finetuning.

    Args:
        weight: The original ``[sphere_channels, n * sphere_channels]`` weight.
        sphere_channels: The backbone ``sphere_channels`` (``weight.shape[0]``).

    Returns:
        A ``[sphere_channels, (n + 1) * sphere_channels]`` weight.
    """
    out_dim, in_dim = weight.shape
    new_weight = weight.new_zeros((out_dim, in_dim + sphere_channels))
    new_weight[:, :in_dim] = weight
    return new_weight


def add_solvent_embedding(
    checkpoint: MLIPInferenceCheckpoint,
    solvent_emb_hidden: int = DEFAULT_SOLVENT_EMB_HIDDEN,
    seed: int = 42,
) -> MLIPInferenceCheckpoint:
    """Inject a solvent embedding into a solvent-free inference checkpoint.

    Edits ``checkpoint`` in place: sets the backbone solvent config flags, widens
    ``mix_csd`` and adds freshly-initialized ``solvent_embedding`` weights to both
    ``model_state_dict`` and ``ema_state_dict``. Because the new ``mix_csd``
    columns are zero, the upgraded model is numerically identical to the original
    at init and only learns solvent effects during finetuning.

    Args:
        checkpoint: A loaded ``MLIPInferenceCheckpoint`` without solvent support.
        solvent_emb_hidden: Hidden width of the solvent MLP; must match the
            finetuning backbone config.
        seed: Manual seed for the freshly-initialized solvent embedding.

    Returns:
        The modified checkpoint.

    Raises:
        ValueError: If the checkpoint already has a solvent embedding, or its
            ``mix_csd`` does not have the charge+spin+dataset layout that solvent
            injection requires.
    """
    backbone_cfg = checkpoint.model_config["backbone"]
    if backbone_cfg.get("use_solvent_embedding", False):
        raise ValueError(
            "checkpoint backbone already has use_solvent_embedding=True; nothing to do"
        )

    weight = checkpoint.model_state_dict["backbone.mix_csd.weight"]
    sphere_channels, in_dim = weight.shape
    if in_dim != 3 * sphere_channels:
        raise ValueError(
            "add_solvent_embedding expects a checkpoint with charge+spin+dataset "
            "mixing (mix_csd in_features == 3 * sphere_channels == "
            f"{3 * sphere_channels}), but found in_features={in_dim} "
            f"(sphere_channels={sphere_channels}). Solvent injection is only "
            "supported on dataset-embedding checkpoints."
        )

    # Freshly-initialized solvent embedding weights. The final layer is near-zero
    # by construction in SolventEmbedding; the same weights are injected into both
    # the model and EMA state dicts.
    torch.manual_seed(seed)
    solvent_embedding = SolventEmbedding(
        solvent_input_dim=SOLVENT_DIM,
        embedding_size=sphere_channels,
        hidden_size=solvent_emb_hidden,
        grad=True,
    )
    solvent_sd = solvent_embedding.state_dict()

    def _patch(state_dict: dict, prefix: str) -> None:
        mix_key = prefix + "mix_csd.weight"
        w = state_dict[mix_key]
        state_dict[mix_key] = _expand_mix_csd_weight(w, sphere_channels)
        for k, v in solvent_sd.items():
            state_dict[prefix + "solvent_embedding." + k] = v.to(
                dtype=w.dtype, device=w.device
            )

    _patch(checkpoint.model_state_dict, "backbone.")
    if checkpoint.ema_state_dict is not None:
        _patch(checkpoint.ema_state_dict, "module.backbone.")

    backbone_cfg["use_solvent_embedding"] = True
    backbone_cfg["solvent_emb_grad"] = True
    backbone_cfg["solvent_emb_hidden"] = solvent_emb_hidden

    return checkpoint


def add_solvent_to_checkpoint_file(
    checkpoint_path: str,
    output_path: str,
    solvent_emb_hidden: int = DEFAULT_SOLVENT_EMB_HIDDEN,
    seed: int = 42,
) -> str:
    """Load a checkpoint, inject the solvent embedding, and save it.

    Args:
        checkpoint_path: Path to the solvent-free inference checkpoint.
        output_path: Where to write the upgraded checkpoint.
        solvent_emb_hidden: Hidden width of the solvent MLP.
        seed: Manual seed for the freshly-initialized solvent embedding.

    Returns:
        ``output_path``.
    """
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    checkpoint = add_solvent_embedding(
        checkpoint, solvent_emb_hidden=solvent_emb_hidden, seed=seed
    )
    torch.save(checkpoint, output_path)
    logging.info(f"Saved solvent-enabled checkpoint to {output_path}")
    return output_path


def _create_test_systems() -> dict:
    """Create PBC and non-PBC H2O test systems for the self-test."""
    from ase.build import molecule

    h2o_nopbc = molecule("H2O")
    h2o_pbc = molecule("H2O")
    h2o_pbc.set_cell([10.0, 10.0, 10.0])
    h2o_pbc.set_pbc(True)
    return {"nopbc": h2o_nopbc, "pbc": h2o_pbc}


def compare_checkpoints(
    original_path: str, upgraded_path: str, task_name: str = "omol"
) -> bool:
    """Check the upgraded checkpoint matches the original at init.

    Because the new mix_csd columns are zero, the solvent term contributes
    nothing before finetuning, so energies and forces should match to numerical
    precision on vacuum inputs.

    Args:
        original_path: Path to the solvent-free checkpoint.
        upgraded_path: Path to the solvent-enabled checkpoint.
        task_name: Task to evaluate both checkpoints under.

    Returns:
        ``True`` if all systems match within tolerance.
    """
    from fairchem.core.calculate.ase_calculator import FAIRChemCalculator

    calc_orig = FAIRChemCalculator.from_model_checkpoint(
        original_path, task_name=task_name
    )
    calc_new = FAIRChemCalculator.from_model_checkpoint(
        upgraded_path, task_name=task_name
    )

    all_match = True
    for sys_name, atoms in _create_test_systems().items():
        a0, a1 = atoms.copy(), atoms.copy()
        a0.calc, a1.calc = calc_orig, calc_new
        e_abs = abs(a0.get_potential_energy() - a1.get_potential_energy())
        f_abs = np.abs(a0.get_forces() - a1.get_forces()).max()
        match = e_abs < 1e-5 and f_abs < 1e-5
        all_match = all_match and match
        logging.info(
            f"[{task_name}/{sys_name}] dE={e_abs:.2e} dF={f_abs:.2e} "
            f"{'OK' if match else 'MISMATCH'}"
        )
    return all_match


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(
        description="Inject a solvent embedding into a solvent-free UMA "
        "inference checkpoint so it can be finetuned with solvent conditioning."
    )
    parser.add_argument("--checkpoint-in", type=str, required=True)
    parser.add_argument("--checkpoint-out", type=str, required=True)
    parser.add_argument(
        "--solvent-emb-hidden", type=int, default=DEFAULT_SOLVENT_EMB_HIDDEN
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--verify-task",
        type=str,
        default=None,
        help="If set, run a self-test comparing the upgraded checkpoint to the "
        "original under this task (e.g. 'omol').",
    )
    args = parser.parse_args()

    add_solvent_to_checkpoint_file(
        args.checkpoint_in,
        args.checkpoint_out,
        solvent_emb_hidden=args.solvent_emb_hidden,
        seed=args.seed,
    )

    if args.verify_task is not None:
        ok = compare_checkpoints(
            args.checkpoint_in, args.checkpoint_out, task_name=args.verify_task
        )
        print(f"Self-test match: {ok}")
