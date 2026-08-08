"""Inject the UMI relative-EE processor steps into LeRobot's train script.

``axol umi.train`` *is* ``lerobot-train`` — same config surface, same policies,
same checkpoints — with exactly one seam patched in: the processor factory.
:func:`install` wraps ``make_pre_post_processors`` inside the train module so

1. the preprocessor gains a :class:`~almond_axol.umi.processor.UMIRelativeEEStep`
   right before normalization and the postprocessor gains its paired
   :class:`~almond_axol.umi.processor.UMIAbsoluteEEStep` right after
   unnormalization, and
2. the normalization statistics for ``action`` / ``observation.state`` are
   recomputed over the *relativized* values (the dataset's on-disk stats
   describe absolute base-frame poses, which have the wrong scale and offset
   for chunk-relative offsets).

Both steps serialize into the checkpoint's processor configs, so the saved
policy runs on the completely stock deployment path (``axol run-policy``):
LeRobot's processor factory reconstructs them from the registry and re-wires
the relative/absolute pair after loading.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np
import torch

from lerobot.processor import NormalizerProcessorStep, UnnormalizerProcessorStep
from lerobot.types import TransitionKey
from lerobot.utils.constants import ACTION, OBS_STATE

from .processor import UMIAbsoluteEEStep, UMIRelativeEEStep, ee_pose_groups

_logger = logging.getLogger(__name__)

# Cap on relativized frames sampled for the normalization statistics. Windows
# are strided uniformly per episode to stay under it, so stats cover the whole
# dataset regardless of size while the computation stays a few seconds.
_STATS_MAX_WINDOWS = 20_000


def _feature_names(dataset, key: str) -> list[str] | None:
    names = dataset.meta.features.get(key, {}).get("names")
    return list(names) if names else None


def _episode_slices(episode_index: np.ndarray) -> list[tuple[int, int]]:
    """Contiguous [start, stop) frame ranges per episode, from the hf column."""
    if len(episode_index) == 0:
        return []
    change = np.flatnonzero(np.diff(episode_index)) + 1
    starts = np.concatenate([[0], change])
    stops = np.concatenate([change, [len(episode_index)]])
    return list(zip(starts.tolist(), stops.tolist()))


def _windowed(
    arr: np.ndarray, t: np.ndarray, offsets: list[int], stop: int
) -> np.ndarray:
    """Stack ``arr[t + k]`` for each offset ``k``, clipped to the episode.

    Matches LeRobot's delta-timestamp behavior at episode boundaries (edge
    frames repeat). ``arr`` is the episode's (T, D) block, ``t`` the sampled
    frame indices, returns (len(t), len(offsets), D).
    """
    idx = np.clip(t[:, None] + np.asarray(offsets)[None, :], 0, stop - 1)
    return arr[idx]


def compute_relative_stats(
    dataset, policy_cfg, action_names: list[str], state_names: list[str]
) -> dict[str, dict[str, np.ndarray]]:
    """Normalization stats for chunk-relativized ``action`` / ``observation.state``.

    Samples windows across every episode using the policy's own delta indices
    (the exact chunks it will train on), runs them through the same
    :class:`UMIRelativeEEStep` used in the pipeline, and computes stats over
    the relativized values. Keys mirror the dataset's on-disk stats entries so
    any normalization mode (mean/std, min/max, quantiles) keeps working.
    """
    hf = dataset.hf_dataset.select_columns([ACTION, OBS_STATE, "episode_index"])
    hf = hf.with_format("numpy")
    actions = np.asarray(hf[ACTION], dtype=np.float32)
    states = np.asarray(hf[OBS_STATE], dtype=np.float32)
    episode_index = np.asarray(hf["episode_index"])

    action_offsets = list(policy_cfg.action_delta_indices or [0])
    obs_offsets = list(policy_cfg.observation_delta_indices or [0])

    slices = _episode_slices(episode_index)
    total_frames = sum(stop - start for start, stop in slices)
    stride = max(1, total_frames // _STATS_MAX_WINDOWS)

    step = UMIRelativeEEStep(action_names=action_names, state_names=state_names)
    rel_actions: list[np.ndarray] = []
    rel_states: list[np.ndarray] = []
    for start, stop in slices:
        t = np.arange(0, stop - start, stride)
        a_win = _windowed(actions[start:stop], t, action_offsets, stop - start)
        s_win = _windowed(states[start:stop], t, obs_offsets, stop - start)
        state_t = torch.from_numpy(s_win)
        if len(obs_offsets) == 1:
            state_t = state_t[:, 0, :]  # matches the (B, D) training batch shape
        out = step(
            {
                TransitionKey.OBSERVATION: {OBS_STATE: state_t},
                TransitionKey.ACTION: torch.from_numpy(a_win),
            }
        )
        rel_actions.append(
            out[TransitionKey.ACTION].reshape(-1, a_win.shape[-1]).numpy()
        )
        rel_states.append(
            out[TransitionKey.OBSERVATION][OBS_STATE]
            .reshape(-1, s_win.shape[-1])
            .numpy()
        )

    def _stats(values: np.ndarray, template: dict[str, Any]) -> dict[str, np.ndarray]:
        out: dict[str, np.ndarray] = {}
        for key in template:
            if key == "mean":
                out[key] = values.mean(axis=0)
            elif key == "std":
                out[key] = values.std(axis=0)
            elif key == "min":
                out[key] = values.min(axis=0)
            elif key == "max":
                out[key] = values.max(axis=0)
            elif key == "count":
                out[key] = np.asarray([values.shape[0]])
            elif key.startswith("q") and key[1:].isdigit():
                out[key] = np.quantile(values, int(key[1:]) / 100.0, axis=0).astype(
                    np.float32
                )
            else:
                _logger.warning(
                    "relative stats: unknown stats key %r copied from absolute", key
                )
                out[key] = np.asarray(template[key])
        return {k: np.asarray(v, dtype=np.float32) for k, v in out.items()}

    base = dataset.meta.stats
    return {
        ACTION: _stats(np.concatenate(rel_actions), base[ACTION]),
        OBS_STATE: _stats(np.concatenate(rel_states), base[OBS_STATE]),
    }


def _swap_stats(stats: dict | None, relative: dict) -> dict | None:
    if stats is None:
        return None
    out = dict(stats)
    out.update(relative)
    return out


def _insert_steps(preprocessor, postprocessor, relative_step: UMIRelativeEEStep):
    """Place the relative step before normalize and its pair after unnormalize."""
    pre_steps = list(preprocessor.steps)
    if any(isinstance(s, UMIRelativeEEStep) for s in pre_steps):
        raise RuntimeError("preprocessor already contains a UMIRelativeEEStep")
    norm_idx = next(
        (i for i, s in enumerate(pre_steps) if isinstance(s, NormalizerProcessorStep)),
        None,
    )
    if norm_idx is None:
        raise RuntimeError(
            "umi.train: the policy's preprocessor has no NormalizerProcessorStep; "
            "cannot place the relative-EE step (unsupported policy type?)."
        )
    pre_steps.insert(norm_idx, relative_step)
    preprocessor.steps = pre_steps

    post_steps = list(postprocessor.steps)
    unnorm_idx = next(
        (
            i
            for i, s in enumerate(post_steps)
            if isinstance(s, UnnormalizerProcessorStep)
        ),
        None,
    )
    if unnorm_idx is None:
        raise RuntimeError(
            "umi.train: the policy's postprocessor has no UnnormalizerProcessorStep; "
            "cannot place the absolute-EE step (unsupported policy type?)."
        )
    post_steps.insert(unnorm_idx + 1, UMIAbsoluteEEStep(relative_step=relative_step))
    postprocessor.steps = post_steps


def install(train_module) -> None:
    """Patch ``make_dataset`` / ``make_pre_post_processors`` in ``train_module``.

    ``train_module`` is ``lerobot.scripts.lerobot_train``; its ``train()``
    creates the dataset first and the processors right after, so the wrapped
    dataset factory captures the dataset object the stats computation needs.
    """
    original_make_dataset = train_module.make_dataset
    original_make_processors = train_module.make_pre_post_processors
    captured: dict[str, Any] = {}

    def patched_make_dataset(cfg):
        if getattr(cfg.dataset, "streaming", False):
            raise ValueError(
                "umi.train computes relative-action stats from the on-disk "
                "dataset and does not support --dataset.streaming."
            )
        dataset = original_make_dataset(cfg)
        captured["dataset"] = dataset
        return dataset

    def patched_make_processors(policy_cfg, pretrained_path=None, **kwargs):
        dataset = captured.get("dataset")
        if dataset is None:
            raise RuntimeError(
                "umi.train: processors requested before the dataset was built"
            )

        action_names = _feature_names(dataset, ACTION)
        state_names = _feature_names(dataset, OBS_STATE)
        if not ee_pose_groups(action_names):
            raise ValueError(
                "umi.train needs a Cartesian dataset (actions named "
                "'*_ee.x/.y/.z/.rx/.ry/.rz'), but the action features are "
                f"{action_names}. Record with `axol collect-data --umi` or "
                "`--robot_config.observe_cartesian true`."
            )

        _logger.info("umi.train: computing chunk-relative normalization stats...")
        relative_stats = compute_relative_stats(
            dataset, policy_cfg, action_names, state_names
        )
        _logger.info(
            "umi.train: relative action std %s",
            np.array2string(relative_stats[ACTION]["std"], precision=4),
        )

        if "dataset_stats" in kwargs:
            kwargs["dataset_stats"] = _swap_stats(
                kwargs["dataset_stats"], relative_stats
            )
        # Fine-tune path: the train script overrides the (un)normalizer stats
        # from the checkpoint with the new dataset's absolute stats — swap in
        # the relative ones there too.
        pre_overrides = kwargs.get("preprocessor_overrides")
        if pre_overrides and "normalizer_processor" in pre_overrides:
            pre_overrides["normalizer_processor"]["stats"] = _swap_stats(
                pre_overrides["normalizer_processor"].get("stats"), relative_stats
            )
        post_overrides = kwargs.get("postprocessor_overrides")
        if post_overrides and "unnormalizer_processor" in post_overrides:
            post_overrides["unnormalizer_processor"]["stats"] = _swap_stats(
                post_overrides["unnormalizer_processor"].get("stats"), relative_stats
            )

        preprocessor, postprocessor = original_make_processors(
            policy_cfg, pretrained_path, **kwargs
        )

        has_relative = any(isinstance(s, UMIRelativeEEStep) for s in preprocessor.steps)
        if has_relative:
            # Fine-tuning a umi.train checkpoint: the steps deserialized from
            # it and the factory already re-wired the relative/absolute pair.
            _logger.info("umi.train: relative-EE steps loaded from checkpoint.")
        else:
            _insert_steps(
                preprocessor,
                postprocessor,
                UMIRelativeEEStep(action_names=action_names, state_names=state_names),
            )
            _logger.info("umi.train: injected relative-EE processor steps.")
        return preprocessor, postprocessor

    train_module.make_dataset = patched_make_dataset
    train_module.make_pre_post_processors = patched_make_processors
