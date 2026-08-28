"""
axol mantis.train

Train any LeRobot policy on a Cartesian Axol dataset with chunk-relative
end-effector actions. This command *is* ``lerobot-train`` —
identical CLI surface (draccus dotted overrides, ``--config_path``, wandb,
resume, accelerate), identical checkpoints — with one seam patched in: the
processor factory additionally installs the relative-EE step pair
(:mod:`almond_axol.mantis.processor`) and recomputes the action/state
normalization statistics over the relativized values.

The dataset stays completely standard: absolute base-frame EE poses as
recorded by ``axol collect-data --mantis`` (or on-robot with
``--robot_config.observe_cartesian true``), so rig-collected and on-robot
episodes mix freely in one dataset and vanilla ``lerobot-train`` still works
on it (just without the relative-action generalization). The relativization
lives in the *policy checkpoint's* processor pipeline; deployment is the
stock path — ``axol run-policy --policy.type act --policy_path <ckpt> ...``.

Example::

    axol mantis.train \\
        --dataset.repo_id almond/mantis_pick \\
        --policy.type act \\
        --output_dir outputs/mantis_pick_act \\
        --batch_size 32 --steps 100000

Note: with relative actions, chunks predicted from different observations are
anchored to different reference poses. Chunk-queue execution (the default) is
exact; ``run-policy --aggregate_fn temporal_ensemble`` blends near-identical
absolute actions from overlapping chunks and works well in practice, but the
blend is an approximation.
"""

from __future__ import annotations

import sys


def main(argv: list[str]) -> None:
    """Run LeRobot's trainer with the Mantis relative-EE processor injected."""
    from lerobot.scripts import lerobot_train

    from ..mantis.train_patch import install

    install(lerobot_train)
    sys.argv = ["lerobot-train", *argv]
    lerobot_train.main()
