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
