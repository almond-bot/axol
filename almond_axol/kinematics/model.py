"""Process-wide shared robot model for the kinematics entry points.

The bundled URDF never changes at runtime, but ``pyroki.Robot.from_urdf``
mints a fresh ``JointVar`` class on every call — and that class is a
*static* field of the ``Robot`` pytree. Two ``Robot`` instances built from
the same URDF therefore never compare equal for JIT-cache purposes: every
jitted function that takes a ``Robot`` (the IK solve, forward kinematics)
re-traces and re-runs jaxls problem analysis per instance, costing seconds
of pure Python work each time even though the compiled XLA executable is
identical (and served from the persistent cache, see :mod:`.jax_cache`).

Building the robot once per process and handing the same instance to every
:class:`~almond_axol.kinematics.solver.KinematicsSolver` /
:class:`~almond_axol.kinematics.fk.AxolForwardKinematics` makes each
instance after the first hit JAX's in-memory jit cache, so constructing
additional objects is close to free.

The collision model is built lazily and separately: forward-kinematics-only
users (observation recording) never pay for capsule fitting.

Each :class:`~almond_axol.constants.AxolModel` (hardware version) has its own
URDF and therefore its own robot and collision model; both are cached per
version. A ``model`` of ``None`` is inferred — mobile when Jelly is enabled — by
:func:`almond_axol.settings.resolve_robot_model`.
"""

from __future__ import annotations

import logging
import threading

import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

from ..constants import AxolModel, torso_links, urdf_path
from ..settings import resolve_robot_model

_logger = logging.getLogger(__name__)


def _closest_segment_to_segment_points(
    a1: jnp.ndarray, b1: jnp.ndarray, a2: jnp.ndarray, b2: jnp.ndarray
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Closest points between segments ``[a1, b1]`` and ``[a2, b2]``.

    Replaces pyroki's version (see :func:`_patch_pyroki`), which is wrong
    for (anti)parallel segments: it projects ``a2`` onto the first segment
    and ``a1`` onto the second, so the two "closest" points do not belong to
    each other. Two antiparallel vertical capsules side by side then read
    their vertical offset as extra clearance — on Axol Mobile the right
    upper arm hanging beside the lift column read 249 mm instead of 84 mm,
    and the upper-arm guard, calibrated to that bogus home clearance, shoved
    the right arm up and out of its rest pose. This is Ericson's routine
    (Real-Time Collision Detection, 5.1.9): pick ``s``, derive ``t`` from it,
    and re-derive ``s`` whenever ``t`` clamps, so the pair is always
    consistent. Divisions are guarded so gradients stay finite.
    """
    eps = 1e-9
    d1 = b1 - a1
    d2 = b2 - a2
    r = a1 - a2
    a = jnp.sum(d1 * d1, axis=-1)
    e = jnp.sum(d2 * d2, axis=-1)
    f = jnp.sum(d2 * r, axis=-1)
    c = jnp.sum(d1 * r, axis=-1)
    b = jnp.sum(d1 * d2, axis=-1)
    denom = a * e - b * b
    general = denom > 1e-6 * jnp.maximum(a * e, eps)
    s = jnp.where(
        general, jnp.clip((b * f - c * e) / jnp.where(general, denom, 1.0), 0, 1), 0.0
    )
    t = (b * s + f) / jnp.maximum(e, eps)
    s = jnp.where(
        t < 0.0,
        jnp.clip(-c / jnp.maximum(a, eps), 0.0, 1.0),
        jnp.where(t > 1.0, jnp.clip((b - c) / jnp.maximum(a, eps), 0.0, 1.0), s),
    )
    t = jnp.clip(t, 0.0, 1.0)
    return a1 + d1 * s[..., None], a2 + d2 * t[..., None]


def _patch_pyroki() -> None:
    """Swap in the parallel-safe segment routine for pyroki's capsule pairs.

    ``capsule_capsule`` looks the helper up on the module at call (trace)
    time, so replacing the attribute is enough. Remove once almond-pyroki
    ships the fix (``tests/test_robot_model.py`` asserts the stock routine
    is still wrong, and fails when it is not).
    """
    from pyroki.collision import _utils

    _utils.closest_segment_to_segment_points = _closest_segment_to_segment_points


_patch_pyroki()

# The two shoulder links mount directly onto the torso. Their conservative
# capsules overlap the base by construction and cannot be used as collision
# constraints. Distal links must remain protected even when their fitted
# capsule overlaps at the straight-down home pose: their clearance relative
# to that known-safe pose still detects an arm folding into the base.
_SHOULDER_MOUNT_SUFFIXES = ("_s2", "_s3")

# Per-pair collision-cost activation distances are derived from each pair's
# clearance at the home pose: ``min(home_clearance - _MARGIN_REST_BUFFER,
# default_margin)``. Pairs that *live* near their activation shell (the
# wrists and grippers pass within 10-17 mm of the base during ordinary
# close-over-table work, and the elbow capsules graze at rest) would
# otherwise keep the cost hinge active across the entire work envelope —
# replaying recorded deburring sessions showed base<->wrist pairs inside a
# uniform 25 mm activation for up to 59% of all frames, and every crossing of
# the shell pulses the collision gradient into the arm, which reads as
# path-specific jitter. Deriving the margin from the home clearance keeps
# such pairs silent in their normal envelope while pairs with generous
# clearance keep the full early-warning distance.
#
# The activation may be *negative*: ``base <-> e2`` (the elbow) reads
# -1.4 mm at home because pyroki's conservative capsule fits already
# interpenetrate there while the physical parts hang in free air. pyroki's
# stock cost cannot activate below zero (its smoothing ramp spans [0,
# margin]), so this pair used to be clamped to a +8 mm activation — leaving
# its gradient permanently active at rest and across the whole
# gripper-in-front-of-torso envelope, where it measurably tripled the
# solver's per-tick output acceleration and kicked the elbow at every shell
# crossing. The custom residual in .solver decouples the activation start
# from the ramp width, so such pairs activate only when the pose actually
# gets closer than home, while keeping the full linear pushback deeper in.
_RAMP_WIDTH_MIN = 0.008
_MARGIN_REST_BUFFER = 0.002

_lock = threading.RLock()
_urdf: dict[AxolModel, yourdfpy.URDF] = {}
_robot: dict[AxolModel, pk.Robot] = {}
_robot_coll: dict[AxolModel, pk.collision.RobotCollision] = {}


def _load_urdf(model: AxolModel) -> yourdfpy.URDF:
    if model not in _urdf:
        path = urdf_path(model)
        _logger.info("Loading Axol URDF (%s)...", path.name)
        _urdf[model] = yourdfpy.URDF.load(str(path), mesh_dir=str(path.parent))
    return _urdf[model]


def shared_robot(model: AxolModel | str | None = None) -> pk.Robot:
    """The pyroki robot for ``model``'s bundled URDF, built once per process."""
    resolved = resolve_robot_model(model)
    with _lock:
        if resolved not in _robot:
            _robot[resolved] = pk.Robot.from_urdf(_load_urdf(resolved))
        return _robot[resolved]


def shared_robot_collision(
    model: AxolModel | str | None = None,
) -> pk.collision.RobotCollision:
    """``model``'s torso<->arm collision model, built once per process."""
    resolved = resolve_robot_model(model)
    with _lock:
        if resolved not in _robot_coll:
            _robot_coll[resolved] = _build_robot_collision(
                _load_urdf(resolved), torso_links(resolved)
            )
        return _robot_coll[resolved]


def _build_robot_collision(
    urdf: yourdfpy.URDF, torso: tuple[str, ...] = torso_links()
) -> pk.collision.RobotCollision:
    """Build ``RobotCollision`` with self-collision restricted to torso<->arm pairs.

    Each Axol arm is a serial chain attached to a static torso (``torso``:
    ``base`` + ``s1`` on the classic Axol, plus the lift column's top plate
    and the ``head`` camera mount on the mobile one — see
    :func:`almond_axol.constants.torso_links`). pyroki's PCA capsule fit
    produces conservative single-capsule-per-link shapes that always overlap at adjacent-link joint interfaces,
    so blanket self-collision causes persistent jitter the IK cannot
    resolve. We restrict the active pair set to the only collisions that
    actually matter: any link pair where exactly one side is the torso
    and the other is an arm link. Within-arm, cross-arm, and torso<->torso
    pairs are filtered out (cross-arm contacts are unreachable, within-arm
    is constrained by joint limits, and torso<->torso is rigidly fixed).

    Only the shoulder-mount links (``s2``/``s3``) are excluded. The old model
    also discarded every distal pair whose conservative capsule happened to
    overlap at the safe straight-down home pose. That removed ``base/s1 <->
    e1`` — the upper-arm pair that physically contacted the base during a
    cross-body reach — as well as ``base <-> w2``. The custom residual uses a
    per-pair activation relative to home clearance, so those negative home
    distances are usable: the cost is off at home and activates only when the
    arm moves closer to the base than that known-safe reference.
    """
    link_names = [link.name for link in urdf.robot.links]

    def is_arm(n: str) -> bool:
        return n.startswith("left_") or n.startswith("right_")

    def is_torso(n: str) -> bool:
        return n in torso

    ignore: set[tuple[str, str]] = set()
    for i, a in enumerate(link_names):
        for b in link_names[i + 1 :]:
            keep = (is_torso(a) and is_arm(b)) or (is_torso(b) and is_arm(a))
            arm_link = b if is_torso(a) else a if is_torso(b) else ""
            if arm_link.endswith(_SHOULDER_MOUNT_SUFFIXES):
                keep = False
            if not keep:
                ignore.add((a, b))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))
    _logger.info(
        "RobotCollision: restricted to %d torso<->arm pairs.",
        len(rc.active_idx_i),
    )
    return rc


_UPPER_ARM_GUARD_SLACK = 0.020


def upper_arm_guard_floor(home_clearance: float) -> float:
    """Hard-stop clearance for an upper arm (e1) against the body.

    On the classic Axol the fitted capsules already overlap at the safe
    straight-down pose, so the threshold is relative to it: at most 20 mm
    closer than home. The recorded cross-body contact was 23-31 mm closer,
    leaving about 10 mm of model-space headroom. Where the body is genuinely
    clear at home — the Jelly lift column sits behind the mobile arms, 84 mm
    from the upper arm — "20 mm closer than home" would forbid ordinary
    poses (the default rest pose comes 26 mm closer), so the floor is capped
    at zero: there the guard stops actual capsule contact. Classic floors
    (home clearance below 20 mm) are unchanged.
    """
    return min(float(home_clearance) - _UPPER_ARM_GUARD_SLACK, 0.0)


def collision_cost_params(
    robot: pk.Robot, rc: pk.collision.RobotCollision, default_margin: float
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair ``(activation_start, ramp_width)`` for the smoothed collision cost.

    ``activation_start = min(home_clearance - _MARGIN_REST_BUFFER,
    default_margin)`` for each active pair — possibly negative for pairs
    whose conservative capsules already interpenetrate at home (see the
    constants above). ``ramp_width`` is the distance over which the residual
    ramps smoothly from zero to its full linear slope: equal to the activation
    start where that start is generous, floored at ``_RAMP_WIDTH_MIN`` so
    near-shell pairs keep a protective gradient onset.

    Cheap (one collision-distance evaluation), so callers compute it per
    solver instance rather than caching it here: the result depends on the
    configured ``default_margin``.
    """
    q0 = jnp.zeros(robot.joints.num_actuated_joints)
    d = np.asarray(rc.compute_self_collision_distance(robot, q0))
    starts = np.minimum(d - _MARGIN_REST_BUFFER, default_margin).astype(np.float32)
    widths = np.maximum(starts, _RAMP_WIDTH_MIN).astype(np.float32)
    _logger.info(
        "Collision activation: %d of %d pairs below the default %.0f mm "
        "(%d activate only inside their home-pose clearance).",
        int((starts < default_margin).sum()),
        len(starts),
        1e3 * default_margin,
        int((starts <= 0.0).sum()),
    )
    return starts, widths
