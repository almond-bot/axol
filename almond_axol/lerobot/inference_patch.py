"""LeRobot async-inference compatibility shims.

Isolated so both the auto-launched policy-server child process
(``run-policy``) and the standalone ``inference-server`` apply the exact
same patch through one guarded code path.
"""

from __future__ import annotations

import logging

_logger = logging.getLogger(__name__)


def disable_observation_similarity_filter() -> None:
    """Stop ``PolicyServer`` from dropping observations as "too similar".

    Upstream's ``observations_similar`` filter skips any observation whose
    joint-space L2 distance from the previous one is under a **hardcoded**
    1-rad tolerance (``lerobot.async_inference.helpers``). On Axol's 16-DOF
    arms at 60 Hz consecutive observations are almost always within that
    bound, so the filter drops nearly every observation and starves the
    action queue.

    LeRobot exposes no public knob for this — the tolerance is a function
    default that ``PolicyServer`` never threads through ``PolicyServerConfig``
    — so the only fix without an upstream change is to neutralize the module
    symbol before ``serve`` runs. This is a deliberate private-API
    dependency; it is guarded so a LeRobot upgrade that renames or removes
    the symbol fails loudly here instead of silently re-enabling the filter.

    (The clean long-term fix is to upstream a ``similarity_atol`` /
    ``skip_similar_observations`` field on ``PolicyServerConfig``.)
    """
    from lerobot.async_inference import policy_server as ps

    if not hasattr(ps, "observations_similar"):
        raise RuntimeError(
            "lerobot.async_inference.policy_server no longer defines "
            "'observations_similar'; the Axol observation-filter workaround "
            "needs review against the new LeRobot version (otherwise the "
            "policy server may silently drop observations and starve the "
            "action queue)."
        )

    ps.observations_similar = lambda *args, **kwargs: False
    _logger.debug("Disabled PolicyServer observation-similarity filter.")


def import_robot_client_preserving_logging() -> None:
    """Import lerobot's ``RobotClient`` without letting it hijack root logging.

    Importing ``lerobot.async_inference.robot_client`` runs ``get_logger`` at
    class scope, which calls ``init_logging``: the root logger is reset to
    ``NOTSET``, every installed handler is cleared, and lerobot's own console
    handler plus a ``logs/`` file handler at DEBUG level are installed. With
    the root effectively at DEBUG, python-can then emits two records per
    transmitted CAN frame — thousands of synchronous disk writes per second
    through the shared logging lock, sitting directly on the impedance-command
    path. Measured on the robot host, that throttled run-policy's 60 Hz
    control loop to an irregular 35-45 Hz per arm (visible arm jitter) and
    grew a multi-hundred-MB log file per session.

    Snapshot the root logger's level and handlers, trigger the import, then
    restore both, so the process keeps exactly the logging its entry point
    (CLI ``main`` or the serve runner's capture) configured. The ``can``
    logger is additionally pinned to INFO so even a deliberate DEBUG session
    can't re-enable per-frame TX logging on the control path.
    """
    root = logging.getLogger()
    prior_level = root.level
    prior_handlers = root.handlers[:]

    import lerobot.async_inference.robot_client  # noqa: F401

    root.setLevel(prior_level)
    root.handlers[:] = prior_handlers
    logging.getLogger("can").setLevel(logging.INFO)
    _logger.debug("Restored root logging after lerobot robot_client import.")
