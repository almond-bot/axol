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

The collision model (:class:`~.collision.ArmBodyCollision`) is built lazily
and separately: forward-kinematics-only users (observation recording) never
pay for building it.

Each :class:`~almond_axol.constants.AxolModel` (hardware version) has its own
URDF and therefore its own robot and collision model; both are cached per
version (and, on the mobile Axol, per gripper). A ``model`` of ``None`` is
inferred — mobile when Jelly is enabled — by
:func:`almond_axol.settings.resolve_robot_model`, and a ``gripper`` of
``None`` by :func:`almond_axol.settings.resolve_gripper`.
"""

from __future__ import annotations

import io
import logging
import threading
from pathlib import Path
from xml.etree import ElementTree

import jax
import jax.numpy as jnp
import numpy as np
import pyroki as pk
import yourdfpy

from ..constants import (
    AxolModel,
    torso_links,
    urdf_arm_joint_names,
    urdf_path,
)
from ..settings import resolve_gripper, resolve_robot_model
from .collision import ArmBodyCollision

_logger = logging.getLogger(__name__)


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
# Keyed by (version, gripper): the parcel gripper has its own URDF.
_Key = tuple[AxolModel, str]
_urdf: dict[_Key, yourdfpy.URDF] = {}
_robot: dict[_Key, pk.Robot] = {}
_robot_coll: dict[_Key, ArmBodyCollision] = {}
_lift_shift: dict[_Key, tuple[np.ndarray, float]] = {}
_finger_slide: dict[_Key, tuple[np.ndarray, np.ndarray, float]] = {}

# The lift's driving joint in the generated URDFs (0 = fully raised,
# lowering the arms); any further stages mimic it.
_LIFT_JOINT = "lift"


def _key(model: AxolModel | str | None, gripper: str | None) -> _Key:
    resolved = resolve_robot_model(model)
    # Only the mobile Axol has a URDF per gripper: skip the settings lookup
    # for the others.
    if resolved is AxolModel.MOBILE:
        gripper = resolve_gripper(gripper)
    else:
        gripper = "parallel"
    return resolved, gripper


def ik_urdf_xml(path: Path) -> str:
    """The bundled URDF at ``path`` as IK sees it.

    The generated models are full robots (see
    :func:`~almond_axol.constants.urdf_path`). For IK every joint but the
    arms' is frozen at zero — the body, lift, fingers and wheels — since
    pyroki treats every movable joint as a variable; and each link's
    ``<ik_collision>`` shapes replace its per-part ``<collision>`` hulls (see
    :mod:`.collision`). A URDF without ``<ik_collision>`` (the classic one)
    keeps its collision geometry.
    """
    root = ElementTree.parse(path).getroot()
    solved = set(
        urdf_arm_joint_names(is_left=True) + urdf_arm_joint_names(is_left=False)
    )
    for joint in root.findall("joint"):
        if joint.get("type") == "fixed" or joint.get("name") in solved:
            continue
        joint.set("type", "fixed")
        for tag in ("axis", "limit", "mimic", "dynamics", "drive"):
            for el in joint.findall(tag):
                joint.remove(el)
    if root.find("link/ik_collision") is not None:
        for link in root.findall("link"):
            for el in link.findall("collision"):
                link.remove(el)
            for el in link.findall("ik_collision"):
                el.tag = "collision"
    return ElementTree.tostring(root, encoding="unicode")


def ik_urdf(path: Path) -> yourdfpy.URDF:
    """The bundled URDF at ``path`` as IK sees it (:func:`ik_urdf_xml`), with
    meshes resolved next to it."""
    return yourdfpy.URDF.load(io.StringIO(ik_urdf_xml(path)), mesh_dir=str(path.parent))


def _load_urdf(key: _Key) -> yourdfpy.URDF:
    if key not in _urdf:
        path = urdf_path(key[0], gripper=key[1])
        _logger.info("Loading Axol URDF (%s)...", path.name)
        _urdf[key] = ik_urdf(path)
    return _urdf[key]


def shared_urdf(
    model: AxolModel | str | None = None, *, gripper: str | None = None
) -> yourdfpy.URDF:
    """The parsed bundled URDF for ``model``, as IK sees it, loaded once per
    process."""
    with _lock:
        return _load_urdf(_key(model, gripper))


def shared_robot(
    model: AxolModel | str | None = None, *, gripper: str | None = None
) -> pk.Robot:
    """The pyroki robot for ``model``'s bundled URDF, built once per process.

    ``gripper`` is the fitted gripper (``AxolConfig.gripper``; ``None`` reads
    it from the shared settings): the mobile Axol has a URDF per gripper.
    """
    with _lock:
        key = _key(model, gripper)
        if key not in _robot:
            _robot[key] = pk.Robot.from_urdf(_load_urdf(key))
        return _robot[key]


def shared_robot_collision(
    model: AxolModel | str | None = None, *, gripper: str | None = None
) -> ArmBodyCollision:
    """``model``'s torso<->arm collision model, built once per process."""
    with _lock:
        key = _key(model, gripper)
        if key not in _robot_coll:
            _robot_coll[key] = _build_robot_collision(
                _load_urdf(key),
                torso_links(key[0]),
                tuple(shared_robot(key[0], gripper=key[1]).links.names),
            )
        return _robot_coll[key]


def lift_link_shift(
    model: AxolModel | str | None = None, *, gripper: str | None = None
) -> tuple[np.ndarray, float]:
    """How the lift moves each IK link relative to the arms, and its travel.

    IK freezes the lift fully raised (:func:`ik_urdf_xml`) and solves in the
    torso's frame, which the lift carries with the arms. Lowering it leaves
    the parts below behind — the Jelly deck and the lower column sections —
    so relative to the arms they rise. Returns ``(shift, travel)``:
    ``shift`` is ``(num_links, 3)`` in the IK robot's link order, each link's
    displacement in the torso frame per metre the lift is lowered (zero for
    the links that ride with the arms), and ``travel`` the lift's full stroke
    (m). A version without a lift has zero shift and travel.
    """
    with _lock:
        key = _key(model, gripper)
        if key not in _lift_shift:
            names = list(shared_robot(key[0], gripper=key[1]).links.names)
            _lift_shift[key] = _measure_lift_shift(
                urdf_path(key[0], gripper=key[1]), names
            )
        return _lift_shift[key]


def finger_slide(
    model: AxolModel | str | None = None, *, gripper: str | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """How the gripper opening moves each finger link, and the fingers' travel.

    IK freezes the fingers closed (:func:`ik_urdf_xml`). Each finger slides
    along its joint axis, in its own frame, as the gripper opens. Returns
    ``(axis, side, travel)``: ``axis`` is ``(num_links, 3)`` in the IK
    robot's link order, each finger link's displacement in its own frame per
    metre of opening (zero for every other link); ``side`` is ``(num_links,)``,
    0 for a left finger, 1 for a right one and -1 otherwise; ``travel`` is the
    fingers' full stroke (m). A gripper without sliding fingers (the parcel
    gripper, the classic URDF) has zero axes and travel.
    """
    with _lock:
        key = _key(model, gripper)
        if key not in _finger_slide:
            names = list(shared_robot(key[0], gripper=key[1]).links.names)
            _finger_slide[key] = _measure_finger_slide(
                urdf_path(key[0], gripper=key[1]), names
            )
        return _finger_slide[key]


def _measure_finger_slide(
    path: Path, names: list[str]
) -> tuple[np.ndarray, np.ndarray, float]:
    axis = np.zeros((len(names), 3), dtype=np.float32)
    side = np.full(len(names), -1, dtype=np.int32)
    travel = 0.0
    full = yourdfpy.URDF.load(
        str(path), build_collision_scene_graph=False, load_meshes=False
    )
    for joint in full.joint_map.values():
        child = joint.child
        if (
            joint.type != "prismatic"
            or "_finger_" not in child
            or child not in names
            or joint.axis is None
        ):
            continue
        k = names.index(child)
        multiplier = 1.0 if joint.mimic is None else float(joint.mimic.multiplier)
        axis[k] = np.asarray(joint.axis, dtype=np.float32) * multiplier
        side[k] = 0 if child.startswith("left_") else 1
        driver = joint if joint.mimic is None else full.joint_map[joint.mimic.joint]
        travel = max(travel, float(driver.limit.upper))
    return axis, side, travel


def _measure_lift_shift(path: Path, names: list[str]) -> tuple[np.ndarray, float]:
    shift = np.zeros((len(names), 3), dtype=np.float32)
    full = yourdfpy.URDF.load(
        str(path), build_collision_scene_graph=False, load_meshes=False
    )
    joint = full.joint_map.get(_LIFT_JOINT)
    if joint is None or joint.limit is None:
        return shift, 0.0
    travel = float(joint.limit.upper)
    root = full.base_link
    # The torso root: the link the arms' first joints hang from (s1's
    # ancestor chain ends there), whose frame IK solves in.
    torso = "root" if "root" in full.link_map else "s1"

    def relative(q: float) -> np.ndarray:
        cfg = {n: 0.0 for n in full.actuated_joint_names}
        cfg[_LIFT_JOINT] = q
        full.update_cfg(cfg)
        world_torso = full.get_transform(torso, root)
        inv = np.linalg.inv(world_torso)
        out = np.zeros((len(names), 3))
        for k, name in enumerate(names):
            if name in full.link_map:
                out[k] = (inv @ full.get_transform(name, root))[:3, 3]
        return out

    shift[:] = (relative(travel) - relative(0.0)) / travel
    full.update_cfg({n: 0.0 for n in full.actuated_joint_names})
    return shift, travel


def _uses_primitives(urdf: yourdfpy.URDF) -> bool:
    """Whether the IK collision is fitted boxes and capsules (no meshes)."""
    geoms = [c.geometry for link in urdf.robot.links for c in link.collisions]
    return bool(geoms) and all(g.mesh is None for g in geoms)


def _build_robot_collision(
    urdf: yourdfpy.URDF,
    torso: tuple[str, ...] = torso_links(),
    link_names: tuple[str, ...] | None = None,
) -> ArmBodyCollision:
    """Build the collision model, restricted to torso<->arm pairs.

    Each Axol arm is a serial chain attached to a static torso (``torso``:
    ``base`` + ``s1`` on the classic Axol, plus the lift column, the deck and
    the ``head`` camera mount on the mobile one — see
    :func:`almond_axol.constants.torso_links`). The only collisions that
    matter are an arm link against the torso: within-arm, cross-arm, and
    torso<->torso pairs are left out (cross-arm contacts are unreachable,
    within-arm is constrained by joint limits, and torso<->torso is rigidly
    fixed).

    Only the shoulder-mount links (``s2``/``s3``) are excluded. They mount
    directly onto the torso, so their shapes overlap it by construction.
    Distal pairs whose shapes overlap at the safe straight-down home pose
    are kept: the custom residual activates each pair relative to its home
    clearance (:func:`collision_cost_params`), so the cost is off at home
    and comes on only when the arm moves closer to the body than that.

    A URDF of fitted boxes and capsules (:func:`_uses_primitives`) collides
    those, against ``link_names`` (the pyroki robot's link order); any other
    collides pyroki's one capsule per link, fitted to its meshes.
    """
    # Frame-only links (camera optical frames, the fingers' own frames) would
    # otherwise enter as zero-radius capsules at their origins.
    has_geometry = {link.name for link in urdf.robot.links if link.collisions}

    def is_arm(n: str) -> bool:
        return (
            n.startswith(("left_", "right_"))
            and n in has_geometry
            and not n.endswith(_SHOULDER_MOUNT_SUFFIXES)
        )

    def is_torso(n: str) -> bool:
        return n in torso

    if _uses_primitives(urdf):
        if link_names is None:
            link_names = tuple(pk.Robot.from_urdf(urdf).links.names)
        rc = ArmBodyCollision.from_primitives(urdf, link_names, is_arm, is_torso)
        _logger.info(
            "Arm/body collision: %d capsules and %d boxes, %d torso<->arm pairs.",
            len(rc.capsule_link),
            len(rc.box_link),
            len(rc.active_idx_i),
        )
        return rc

    names = [link.name for link in urdf.robot.links]
    ignore: set[tuple[str, str]] = set()
    for i, a in enumerate(names):
        for b in names[i + 1 :]:
            if not ((is_torso(a) and is_arm(b)) or (is_torso(b) and is_arm(a))):
                ignore.add((a, b))

    rc = pk.collision.RobotCollision.from_urdf(urdf, user_ignore_pairs=tuple(ignore))
    _logger.info(
        "RobotCollision: restricted to %d torso<->arm pairs.",
        len(rc.active_idx_i),
    )
    return ArmBodyCollision.from_pyroki(rc)


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
    robot: pk.Robot,
    rc: ArmBodyCollision,
    default_margin: float,
    home_shifts: tuple[jax.Array, ...] = (),
) -> tuple[np.ndarray, np.ndarray]:
    """Per-pair ``(activation_start, ramp_width)`` for the smoothed collision cost.

    ``activation_start = min(home_clearance - _MARGIN_REST_BUFFER,
    default_margin)`` for each active pair — possibly negative for pairs
    whose conservative capsules already interpenetrate at home (see the
    constants above). ``ramp_width`` is the distance over which the residual
    ramps smoothly from zero to its full linear slope: equal to the activation
    start where that start is generous, floored at ``_RAMP_WIDTH_MIN`` so
    near-shell pairs keep a protective gradient onset.

    ``home_shifts``: further link shifts (see
    :func:`~.solver.body_collision_distance`) the home pose is measured with,
    the closest of all counting (the gripper open as well as closed), so no
    pair is active at home in any of them.

    Cheap (a few collision-distance evaluations), so callers compute it per
    solver instance rather than caching it here: the result depends on the
    configured ``default_margin``.
    """
    from .solver import body_collision_distance

    q0 = jnp.zeros(robot.joints.num_actuated_joints)
    d = np.asarray(rc.compute_self_collision_distance(robot, q0))
    for shift in home_shifts:
        d = np.minimum(d, np.asarray(body_collision_distance(robot, rc, q0, shift)))
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
