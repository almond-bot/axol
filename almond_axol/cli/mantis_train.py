"""
axol mantis.train

Train any LeRobot policy on a Cartesian Axol dataset with chunk-relative
end-effector actions. This command uses ``lerobot-train``'s local CLI surface
(draccus dotted overrides, ``--config_path``, wandb, resume, accelerate) and
identical checkpoints, with one seam patched in: the processor factory
additionally installs the relative-EE step pair
(:mod:`almond_axol.mantis.processor`) and recomputes the action/state
normalization statistics over the relativized values.

Remote HF Jobs (``--job.target`` other than ``local``) are intentionally not
supported: their pod invokes plain ``lerobot-train`` and therefore cannot carry
this required process-local patch. Run this command on the training machine.

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


def _remote_job_exit(target: object) -> SystemExit:
    return SystemExit(
        "axol mantis.train does not support remote HF Jobs "
        f"(--job.target={target!s}): the remote pod runs plain lerobot-train "
        "and cannot carry Axol's required Mantis relative-EE processor patch. "
        "Run training locally with --job.target=local (or omit --job.target)."
    )


def _reject_explicit_remote_job(argv: list[str]) -> None:
    """Fail before LeRobot can dispatch an explicitly requested remote job."""
    for index, argument in enumerate(argv):
        if argument.startswith("--job.target="):
            target = argument.split("=", 1)[1]
        elif argument == "--job.target" and index + 1 < len(argv):
            target = argv[index + 1]
        else:
            continue
        if target != "local":
            raise _remote_job_exit(target)


def _reject_remote_submission(cfg) -> None:
    """Catch remote targets inherited from a config/checkpoint as well."""
    target = getattr(getattr(cfg, "job", None), "target", None)
    raise _remote_job_exit(target)


def main(argv: list[str]) -> None:
    """Run LeRobot's trainer with the Mantis relative-EE processor injected."""
    _reject_explicit_remote_job(argv)

    from lerobot.scripts import lerobot_train

    from ..mantis.train_patch import install

    install(lerobot_train)
    # A remote target can also be inherited from --config_path (especially a
    # resume checkpoint), so keep a second guard at LeRobot's dispatch seam.
    # Local training never calls submit_to_hf and retains the full upstream CLI.
    lerobot_train.submit_to_hf = _reject_remote_submission
    sys.argv = ["lerobot-train", *argv]
    lerobot_train.main()
