"""Chunk-relative end-effector processor steps for LeRobot pipelines.

These two steps give any LeRobot policy (ACT, diffusion, ...) chunk-relative
action semantics while keeping the *dataset* absolute and standard:

- The dataset stores absolute base-frame Cartesian state/actions
  (``{side}_ee.{x,y,z,rx,ry,rz}`` + ``{side}_gripper.pos``), the same schema
  for Mantis-rig and on-robot collection, so episodes from both sources mix in
  one dataset.
- :class:`MantisRelativeEEStep` runs in the *pre*-processor (before
  normalization). During training it rewrites every action chunk into the
  frame of the current end-effector pose (``T_rel = T_ref^-1 T_k`` per side)
  and zeroes the observation's EE pose dims (each state frame relative to
  itself), so nothing the policy sees or predicts is anchored to a world
  frame. At inference it does the same to the observation and caches the
  absolute reference pose. State pose dims are zeroed *per frame* — not
  relative to the newest frame — because LeRobot policies preprocess each
  observation once and queue it (a past frame's value cannot be revised as
  time advances), and only per-frame semantics are identical in training
  and inference. Proprioception therefore carries gripper (and torque)
  history but no pose; the motion cue comes from vision.
- :class:`MantisAbsoluteEEStep` runs in the *post*-processor (after
  unnormalization) and composes predicted relative actions with the cached
  reference (``T_abs = T_ref T_rel``), producing absolute Cartesian actions
  that ``run-policy`` executes through the robot's standard IK path.

Both steps are registered with LeRobot's ``ProcessorStepRegistry``, so a
checkpoint trained with them (see ``axol mantis.train``) reloads them
automatically in ``run-policy`` / the inference server — deployment is the
stock LeRobot path. They subclass LeRobot's generic relative/absolute steps
so the factory's post-load reconnection (``relative_step`` wiring) applies.

Rotations ride the same 3-element rotation-vector slots the dataset already
uses, so feature shapes and names never change. Chunk-relative rotations are
small (far from the ``pi`` singularity), which is exactly the regime where
rotation vectors are well behaved.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import torch
from lerobot.lerobot_types import EnvTransition, TransitionKey
from lerobot.processor import ProcessorStepRegistry
from lerobot.processor.relative_action_processor import (
    AbsoluteActionsProcessorStep,
    RelativeActionsProcessorStep,
)
from lerobot.utils.constants import OBS_STATE
from torch import Tensor

_EE_AXES = ("x", "y", "z", "rx", "ry", "rz")


def ee_pose_groups(names: list[str] | None) -> list[list[int]]:
    """Indices of each 6-axis EE pose group in a flat feature-name list.

    A group is a prefix ``P`` such that all of ``P.x P.y P.z P.rx P.ry P.rz``
    are present (e.g. ``left_ee`` / ``right_ee``). Returns one ``[ix, iy, iz,
    irx, iry, irz]`` list per group, ordered by position; every other index
    (grippers, torques) is untouched by the transform.
    """
    if not names:
        return []
    index = {str(n): i for i, n in enumerate(names)}
    prefixes: list[str] = []
    for n in index:
        if n.endswith(".x"):
            p = n[: -len(".x")]
            if all(f"{p}.{a}" in index for a in _EE_AXES):
                prefixes.append(p)
    prefixes.sort(key=lambda p: index[f"{p}.x"])
    return [[index[f"{p}.{a}"] for a in _EE_AXES] for p in prefixes]


# ----------------------------------------------------------------------
# Batched SE(3) in torch (float64 internally for stable log/exp maps)
# ----------------------------------------------------------------------


def _skew(v: Tensor) -> Tensor:
    """(..., 3) -> (..., 3, 3) cross-product matrix."""
    zero = torch.zeros_like(v[..., 0])
    return torch.stack(
        [
            torch.stack([zero, -v[..., 2], v[..., 1]], dim=-1),
            torch.stack([v[..., 2], zero, -v[..., 0]], dim=-1),
            torch.stack([-v[..., 1], v[..., 0], zero], dim=-1),
        ],
        dim=-2,
    )


def rotvec_to_matrix(r: Tensor) -> Tensor:
    """Rodrigues: (..., 3) rotation vector -> (..., 3, 3) rotation matrix."""
    theta = r.norm(dim=-1, keepdim=True)
    small = theta < 1e-8
    axis = r / theta.clamp(min=1e-12)
    k = _skew(axis)
    s = torch.sin(theta)[..., None]
    c = torch.cos(theta)[..., None]
    eye = torch.eye(3, dtype=r.dtype, device=r.device).expand(*r.shape[:-1], 3, 3)
    big = eye + s * k + (1.0 - c) * (k @ k)
    # Near zero, R ~= I + skew(r) (first-order; second-order term is O(theta^2)).
    tiny = eye + _skew(r)
    return torch.where(small[..., None], tiny, big)


def _matrix_to_quat_wxyz(m: Tensor) -> Tensor:
    """(..., 3, 3) rotation matrix -> (..., 4) wxyz quaternion (batched).

    Branch per element on the largest of the four squared components for
    numerical safety (the classic Shepperd selection, vectorized).
    """
    batch = m.shape[:-2]
    mf = m.reshape(-1, 3, 3)
    m00, m11, m22 = mf[:, 0, 0], mf[:, 1, 1], mf[:, 2, 2]
    # 4 * (component^2), each >= 0 up to rounding.
    comp_sq = torch.stack(
        [
            1.0 + m00 + m11 + m22,
            1.0 + m00 - m11 - m22,
            1.0 - m00 + m11 - m22,
            1.0 - m00 - m11 + m22,
        ],
        dim=-1,
    ).clamp(min=0.0)
    two_abs = comp_sq.sqrt()  # 2*|w|, 2*|x|, 2*|y|, 2*|z|
    # Candidate quaternions, one per pivot component (rows: pivot w/x/y/z).
    cand = torch.stack(
        [
            torch.stack(
                [
                    comp_sq[:, 0],
                    mf[:, 2, 1] - mf[:, 1, 2],
                    mf[:, 0, 2] - mf[:, 2, 0],
                    mf[:, 1, 0] - mf[:, 0, 1],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    mf[:, 2, 1] - mf[:, 1, 2],
                    comp_sq[:, 1],
                    mf[:, 1, 0] + mf[:, 0, 1],
                    mf[:, 0, 2] + mf[:, 2, 0],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    mf[:, 0, 2] - mf[:, 2, 0],
                    mf[:, 1, 0] + mf[:, 0, 1],
                    comp_sq[:, 2],
                    mf[:, 2, 1] + mf[:, 1, 2],
                ],
                dim=-1,
            ),
            torch.stack(
                [
                    mf[:, 1, 0] - mf[:, 0, 1],
                    mf[:, 0, 2] + mf[:, 2, 0],
                    mf[:, 2, 1] + mf[:, 1, 2],
                    comp_sq[:, 3],
                ],
                dim=-1,
            ),
        ],
        dim=-2,
    )  # (N, 4 pivots, 4 components)
    pivot = two_abs.argmax(dim=-1)  # (N,)
    rows = cand[torch.arange(mf.shape[0], device=m.device), pivot]
    denom = (2.0 * two_abs[torch.arange(mf.shape[0], device=m.device), pivot]).clamp(
        min=1e-12
    )
    q = rows / denom[:, None]
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    return q.reshape(*batch, 4)


def matrix_to_rotvec(m: Tensor) -> Tensor:
    """(..., 3, 3) rotation matrix -> (..., 3) rotation vector (angle in [0, pi])."""
    q = _matrix_to_quat_wxyz(m)
    q = torch.where(q[..., 0:1] < 0.0, -q, q)
    v = q[..., 1:]
    s = v.norm(dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(s, q[..., 0:1])
    # sin(theta/2) ~= theta/2 near zero: rotvec ~= 2 v.
    scale = torch.where(s > 1e-9, angle / s.clamp(min=1e-12), torch.full_like(s, 2.0))
    return v * scale


def pose6_to_matrix(pose6: Tensor) -> Tensor:
    """(..., 6) ``pos + rotvec`` -> homogeneous (..., 4, 4)."""
    out = torch.zeros(*pose6.shape[:-1], 4, 4, dtype=pose6.dtype, device=pose6.device)
    out[..., :3, :3] = rotvec_to_matrix(pose6[..., 3:6])
    out[..., :3, 3] = pose6[..., :3]
    out[..., 3, 3] = 1.0
    return out


def matrix_to_pose6(mat: Tensor) -> Tensor:
    """Homogeneous (..., 4, 4) -> (..., 6) ``pos + rotvec``."""
    return torch.cat([mat[..., :3, 3], matrix_to_rotvec(mat[..., :3, :3])], dim=-1)


def _se3_inv(mat: Tensor) -> Tensor:
    """Rigid-transform inverse of (..., 4, 4)."""
    rt = mat[..., :3, :3].transpose(-1, -2)
    out = torch.zeros_like(mat)
    out[..., :3, :3] = rt
    out[..., :3, 3] = -(rt @ mat[..., :3, 3:4])[..., 0]
    out[..., 3, 3] = 1.0
    return out


def _transform_pose_dims(
    values: Tensor,
    ref: Tensor,
    groups: list[list[int]],
    *,
    compose: bool,
) -> Tensor:
    """Rewrite the EE pose dims of ``values`` relative to / composed with ``ref``.

    ``values`` is (..., D) with any number of leading dims; ``ref`` is (B, D)
    (one reference per batch element, broadcast across time dims). With
    ``compose=False`` each pose group becomes ``T_ref^-1 T`` (relativize);
    with ``compose=True`` it becomes ``T_ref T`` (compose back to absolute).
    Non-pose dims (grippers, torques) pass through unchanged.
    """
    if not groups:
        return values
    if ref.device != values.device or ref.dtype != values.dtype:
        ref = ref.to(device=values.device, dtype=values.dtype)
    out = values.clone()
    # Broadcast the per-batch reference across any time dims of `values`.
    extra = values.ndim - ref.ndim
    for idx in groups:
        idx_t = torch.tensor(idx, device=values.device)
        pose = values.index_select(-1, idx_t).double()
        ref_pose = ref.index_select(-1, idx_t).double()
        ref_mat = pose6_to_matrix(ref_pose)
        if not compose:
            ref_mat = _se3_inv(ref_mat)
        ref_mat = ref_mat.reshape(
            *ref_pose.shape[:-1], *([1] * extra), 4, 4
        )  # (B, 1..., 4, 4)
        new_pose = matrix_to_pose6(ref_mat @ pose6_to_matrix(pose))
        out[..., idx] = new_pose.to(values.dtype)
    return out


def _current_frame(state: Tensor) -> Tensor:
    """Reference (current-frame) state: last observation step, shape (B, D)."""
    return state[..., -1, :] if state.ndim == 3 else state


@ProcessorStepRegistry.register("mantis_relative_ee_processor")
@dataclass
class MantisRelativeEEStep(RelativeActionsProcessorStep):
    """Relativize EE pose dims of actions; zero them in the proprio state.

    Runs before normalization. During training the batch carries both the
    observation and the action chunk, so the whole transform is stateless
    per batch: the chunk reference is the current-frame EE pose read from
    ``observation.state``. At inference only the observation is present; the
    absolute state is cached (base-class behavior) for the paired
    :class:`MantisAbsoluteEEStep` in the post-processor.

    State pose dims are zeroed per frame (see the module docstring for why
    per-frame is the only train/inference-consistent choice under LeRobot's
    observation queueing); grippers and torques pass through.

    ``action_names`` / ``state_names`` are the dataset feature names; pose
    groups (``*.x .y .z .rx .ry .rz``) are derived from them, everything else
    passes through untouched.
    """

    state_names: list[str] | None = None
    relativize_state: bool = True
    enabled: bool = True

    _action_groups: list[list[int]] | None = field(default=None, init=False, repr=False)
    _state_groups: list[list[int]] | None = field(default=None, init=False, repr=False)

    def _groups(self) -> tuple[list[list[int]], list[list[int]]]:
        if self._action_groups is None:
            self._action_groups = ee_pose_groups(self.action_names)
        if self._state_groups is None:
            names = self.state_names if self.state_names else self.action_names
            self._state_groups = ee_pose_groups(names)
        return self._action_groups, self._state_groups

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        observation = transition.get(TransitionKey.OBSERVATION) or {}
        state = observation.get(OBS_STATE) if observation else None

        # Cache the *absolute* state for the paired MantisAbsoluteEEStep before
        # any relativization below (never mutated in place).
        if state is not None:
            self._last_state = state

        if not self.enabled or state is None:
            return transition

        action_groups, state_groups = self._groups()
        ref = _current_frame(state)
        new_transition = transition.copy()

        action = new_transition.get(TransitionKey.ACTION)
        if action is not None and isinstance(action, Tensor):
            # The state and action feature layouts match (same EE prefixes),
            # so the reference pose is read with the *state* group indices and
            # written with the action ones. Guard against divergent layouts.
            if [g for g in action_groups] != [g for g in state_groups]:
                raise ValueError(
                    "mantis_relative_ee_processor: action and state EE pose "
                    f"layouts differ (action {action_groups} vs state "
                    f"{state_groups}); relative actions need matching layouts."
                )
            new_transition[TransitionKey.ACTION] = _transform_pose_dims(
                action, ref, action_groups, compose=False
            )

        if self.relativize_state and state_groups:
            new_state = state.clone()
            for idx in state_groups:
                new_state[..., idx] = 0.0
            new_obs = dict(observation)
            new_obs[OBS_STATE] = new_state
            new_transition[TransitionKey.OBSERVATION] = new_obs

        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "action_names": self.action_names,
            "state_names": self.state_names,
            "relativize_state": self.relativize_state,
        }


@ProcessorStepRegistry.register("mantis_absolute_ee_processor")
@dataclass
class MantisAbsoluteEEStep(AbsoluteActionsProcessorStep):
    """Compose predicted relative EE actions back to absolute poses.

    Runs after unnormalization in the post-processor. Reads the absolute
    reference pose cached by the paired :class:`MantisRelativeEEStep` during the
    preceding preprocessor call — in both the policy server and synchronous
    rollouts the preprocessor runs immediately before the chunk is predicted
    and postprocessed, so the cached state is the chunk's reference by
    construction. The ``relative_step`` reference is re-wired after
    checkpoint load by LeRobot's processor factory.
    """

    enabled: bool = True

    def __call__(self, transition: EnvTransition) -> EnvTransition:
        if not self.enabled:
            return transition
        if self.relative_step is None:
            raise RuntimeError(
                "mantis_absolute_ee_processor needs its paired "
                "mantis_relative_ee_processor (relative_step is None)."
            )
        cached = self.relative_step.get_cached_state()
        if cached is None:
            raise RuntimeError(
                "mantis_absolute_ee_processor: no cached state — the "
                "preprocessor must run before the postprocessor."
            )
        action = transition.get(TransitionKey.ACTION)
        if action is None:
            return transition

        groups = ee_pose_groups(self.relative_step.action_names)
        ref = _current_frame(cached)
        new_transition = transition.copy()
        new_transition[TransitionKey.ACTION] = _transform_pose_dims(
            action, ref, groups, compose=True
        )
        return new_transition

    def get_config(self) -> dict[str, Any]:
        return {"enabled": self.enabled}
