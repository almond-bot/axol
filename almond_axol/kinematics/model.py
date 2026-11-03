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
import trimesh
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


# The two shoulder links mount directly onto the torso. Their capsules
# overlap the shoulder bar by construction and cannot be used as collision
# constraints. Distal links must remain protected even when their fitted
# capsule comes close to (or overlaps) the body at the straight-down home
# pose: their clearance relative to that known-safe pose still detects an
# arm folding into the body.
_SHOULDER_MOUNT_SUFFIXES = ("_s2", "_s3")

# Per-pair collision-cost activation distances are derived from each pair's
# clearance at the home pose: ``min(home_clearance - _MARGIN_REST_BUFFER,
# default_margin)``. Pairs that *live* near their activation shell (the
# wrists and grippers pass within 10-17 mm of the base during ordinary
# close-over-table work, and the classic elbow capsules graze its column at
# rest) would
# otherwise keep the cost hinge active across the entire work envelope —
# replaying recorded deburring sessions showed base<->wrist pairs inside a
# uniform 25 mm activation for up to 59% of all frames, and every crossing of
# the shell pulses the collision gradient into the arm, which reads as
# path-specific jitter. Deriving the margin from the home clearance keeps
# such pairs silent in their normal envelope while pairs with generous
# clearance keep the full early-warning distance.
#
# The activation may be *negative*: a fitted capsule can overlap a body part
# at home while the physical parts hang in free air (a capsule over a boxy
# arm part bulges past its corners: the classic elbow's reads 0.1 mm from its
# column where the part is 27 mm clear). pyroki's stock cost cannot activate
# below zero (its smoothing ramp spans [0, margin]); clamping such a pair to
# a positive activation leaves its gradient permanently active at rest and
# across the whole gripper-in-front-of-torso envelope, where it measurably
# tripled the solver's per-tick output acceleration and kicked the elbow at
# every shell crossing. The custom residual in .solver decouples the
# activation start from the ramp width, so such pairs activate only when the
# pose actually gets closer than home, while keeping the full linear
# pushback deeper in.
_RAMP_WIDTH_MIN = 0.008
_MARGIN_REST_BUFFER = 0.002

_lock = threading.RLock()
# Keyed by (version, gripper): the parcel gripper has its own URDF.
_Key = tuple[AxolModel, str]
_urdf: dict[_Key, yourdfpy.URDF] = {}
_robot: dict[_Key, pk.Robot] = {}
_robot_coll: dict[_Key, ArmBodyCollision] = {}
_lift_shift: dict[_Key, tuple[np.ndarray, float]] = {}
_finger_slide: dict[_Key, tuple[np.ndarray, np.ndarray, float, float]] = {}

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
    :mod:`.collision`).
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
) -> tuple[np.ndarray, np.ndarray, float, float]:
    """How the gripper opening moves each finger link.

    IK freezes the finger joints at zero (:func:`ik_urdf_xml`), which is
    closed on Jelly and part of the way open on some other exports. Each
    finger slides along its joint axis, in its own frame. Returns ``(axis,
    side, closed, stroke)``: at opening ``g`` (0 closed, 1 fully open) the
    driving finger joint is at ``closed + g * stroke`` (m), and each finger
    link is displaced by ``axis`` times that, in its own frame. ``axis`` is
    ``(num_links, 3)`` in the IK robot's link order (zero for every other
    link, the mimic multiplier folded in); ``side`` is ``(num_links,)``, 0
    for a left finger, 1 for a right one and -1 otherwise. ``closed`` is the
    joint limit where the fingers meet, ``stroke`` the signed distance to the
    other. A gripper without sliding fingers (the parcel gripper, the
    classic URDF) has zero axes and stroke.
    """
    with _lock:
        key = _key(model, gripper)
        if key not in _finger_slide:
            names = list(shared_robot(key[0], gripper=key[1]).links.names)
            _finger_slide[key] = _measure_finger_slide(
                urdf_path(key[0], gripper=key[1]), names
            )
        return _finger_slide[key]


def _finger_gap(full: yourdfpy.URDF, root: Path, driver: str, q: float) -> float:
    """How far apart the left gripper's fingers' part hulls are (centroid to
    centroid, m) with the finger joint ``driver`` at ``q``."""
    cfg = {n: 0.0 for n in full.actuated_joint_names}
    cfg[driver] = q
    full.update_cfg(cfg)
    centres = []
    for link in ("left_finger_1", "left_finger_2"):
        pts = []
        for col in full.link_map[link].collisions:
            mesh = trimesh.load(root / col.geometry.mesh.filename, force="mesh")
            pose = np.eye(4) if col.origin is None else np.asarray(col.origin)
            pts.append(
                trimesh.transform_points(mesh.vertices, full.get_transform(link) @ pose)
            )
        centres.append(np.vstack(pts).mean(axis=0))
    full.update_cfg({n: 0.0 for n in full.actuated_joint_names})
    return float(np.linalg.norm(centres[0] - centres[1]))


def _measure_finger_slide(
    path: Path, names: list[str]
) -> tuple[np.ndarray, np.ndarray, float, float]:
    axis = np.zeros((len(names), 3), dtype=np.float32)
    side = np.full(len(names), -1, dtype=np.int32)
    full = yourdfpy.URDF.load(
        str(path), build_collision_scene_graph=False, load_meshes=False
    )
    driver = None
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
        if joint.mimic is None and child == "left_finger_1":
            driver = joint
    if driver is None:
        return axis, side, 0.0, 0.0
    # The closed end: the limit where the fingers' hulls come together.
    ends = (float(driver.limit.lower), float(driver.limit.upper))
    closed, opened = sorted(
        ends, key=lambda q: _finger_gap(full, path.parent, driver.name, q)
    )
    return axis, side, closed, opened - closed


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


def _build_robot_collision(
    urdf: yourdfpy.URDF,
    torso: tuple[str, ...] = torso_links(),
    link_names: tuple[str, ...] | None = None,
) -> ArmBodyCollision:
    """Build the collision model, restricted to torso<->arm pairs.

    Each Axol arm is a serial chain attached to a static torso (``torso``:
    ``base`` + ``s1`` on the classic Axol; the shoulder bar, the ``head``
    camera mount and the mount, lift column or deck below them on the others
    — see
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

    The shapes are the URDF's fitted boxes and capsules (its
    ``<ik_collision>``, see :func:`ik_urdf_xml`); ``link_names`` is the
    pyroki robot's link order.
    """
    # Frame-only links (camera optical frames) have no shapes.
    has_geometry = {link.name for link in urdf.robot.links if link.collisions}

    def is_arm(n: str) -> bool:
        return (
            n.startswith(("left_", "right_"))
            and n in has_geometry
            and not n.endswith(_SHOULDER_MOUNT_SUFFIXES)
        )

    def is_torso(n: str) -> bool:
        return n in torso

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


_UPPER_ARM_GUARD_SLACK = 0.020


def upper_arm_guard_floor(home_clearance: float) -> float:
    """Hard-stop clearance for an upper arm (e1) against the body.

    Where the upper arm hangs close to the body at the safe straight-down
    pose (every version's shoulder bar, 13-18 mm; the classic column, 5 mm),
    the threshold is relative to it: at most 20 mm closer than home. The
    recorded cross-body contact was 23-31 mm closer, leaving about 10 mm of
    model-space headroom. Where the body is clear at home (the lift columns
    and mount columns sit 53-90 mm from the upper arm), "20 mm closer than
    home" would forbid ordinary poses (the default rest pose comes 26 mm
    closer), so the floor is capped at zero: there the guard stops actual
    shape contact.
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
    whose fitted shapes already interpenetrate at home (see the constants
    above). ``ramp_width`` is the distance over which the residual
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
