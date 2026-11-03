"""The arm/body collision model IK keeps the arms off the body with.

Every Axol arm is a serial chain on a static body (the shoulder bar, head
mount, lift column, deck, ...), and the only contacts IK guards against are
an arm part against a body part (see :mod:`.model`). :class:`ArmBodyCollision`
holds the shapes of both sides in their link frames and the pairs to check:
each URDF's ``<ik_collision>`` boxes and cylinders, fitted to each part by
``scripts/build_urdfs.py``, any number per link. An ik_collision cylinder is
a capsule: its radius, and the straight section's length along the origin's
z. Arm parts are capsules; body parts are boxes or capsules.

Distances are signed (negative: overlap). Capsule-capsule pairs use pyroki's
distance. Capsule-box pairs use :func:`capsule_box_distance`, which is exact:
pyroki's own refines a closest point twice and can read a capsule near a box
edge as further away than it is.
"""

from __future__ import annotations

from collections.abc import Callable

import jax
import jax.numpy as jnp
import jax_dataclasses as jdc
import jaxlie
import numpy as np
import pyroki as pk
import yourdfpy
from pyroki.collision import Box, Capsule

# Golden-section steps along a capsule's segment for its closest approach to
# a box: the bracket shrinks to 0.618^n of the segment (under 0.1 mm on a
# 0.35 m upper arm).
_SECTION_STEPS = 18
_GOLDEN = (np.sqrt(5.0) - 1.0) / 2.0


def _box_sdf(p: jax.Array, half: jax.Array) -> jax.Array:
    """Signed distance from points ``p`` (box frame) to a box of half-extents
    ``half`` centred at the origin."""
    q = jnp.abs(p) - half
    out = jnp.maximum(q, 0.0)
    sq = jnp.sum(out**2, axis=-1)
    # sqrt's gradient is infinite at 0: take it only where it is positive.
    safe = jnp.sqrt(jnp.where(sq > 0.0, sq, 1.0))
    outside = jnp.where(sq > 0.0, safe, 0.0)
    inside = jnp.minimum(jnp.max(q, axis=-1), 0.0)
    return outside + inside


def capsule_box_distance(capsule: Capsule, box: Box) -> jax.Array:
    """Signed distance between capsules and boxes (matching batch shapes).

    A box's signed distance field is convex, so along the capsule's segment
    it has one minimum: a golden-section search finds it, and the distance is
    that minimum less the capsule's radius. The search position is held
    constant for the gradient, which at a minimum is the exact one (the
    derivative along the segment is zero there, or the search sits at an
    end).
    """
    axis = capsule.axis
    half_len = capsule.height[..., None] / 2.0
    centre = capsule.pose.translation()
    to_box = box.pose.inverse()
    p0 = to_box.apply(centre - axis * half_len)
    p1 = to_box.apply(centre + axis * half_len)
    half = box.half_extent

    def f(t: jax.Array) -> jax.Array:
        return _box_sdf(p0 + (p1 - p0) * t[..., None], half)

    lo = jnp.zeros(p0.shape[:-1], dtype=p0.dtype)
    hi = jnp.ones_like(lo)

    def step(_: int, bracket: tuple[jax.Array, jax.Array]):
        lo, hi = bracket
        a = hi - _GOLDEN * (hi - lo)
        b = lo + _GOLDEN * (hi - lo)
        left = f(a) <= f(b)
        return jnp.where(left, lo, a), jnp.where(left, b, hi)

    lo, hi = jax.lax.fori_loop(
        0, _SECTION_STEPS, step, (jax.lax.stop_gradient(lo), jax.lax.stop_gradient(hi))
    )
    t = jax.lax.stop_gradient((lo + hi) / 2.0)
    # The ends too: the search brackets the minimum but never evaluates them.
    best = jnp.minimum(f(t), jnp.minimum(f(jnp.zeros_like(t)), f(jnp.ones_like(t))))
    return best - capsule.radius


@jdc.pytree_dataclass
class ArmBodyCollision:
    """Arm and body shapes in their link frames, and the arm/body pairs.

    ``link_names`` is the pyroki robot's link order. Pair ``k`` is checked
    between the shapes ``pairs`` lists; ``active_idx_i[k]`` /
    ``active_idx_j[k]`` are the *links* those shapes belong to, so a caller
    can name a pair (several pairs can share a link pair).
    """

    link_names: jdc.Static[tuple[str, ...]]
    capsules: Capsule
    capsule_link: jdc.Static[tuple[int, ...]]
    boxes: Box | None
    box_link: jdc.Static[tuple[int, ...]]
    # (capsule, capsule) and (capsule, box) shape pairs; distances come in
    # that order.
    capsule_pairs: jdc.Static[tuple[tuple[int, int], ...]]
    box_pairs: jdc.Static[tuple[tuple[int, int], ...]]

    @property
    def active_idx_i(self) -> tuple[int, ...]:
        return tuple(self.capsule_link[i] for i, _ in self.capsule_pairs) + tuple(
            self.capsule_link[i] for i, _ in self.box_pairs
        )

    @property
    def active_idx_j(self) -> tuple[int, ...]:
        return tuple(self.capsule_link[j] for _, j in self.capsule_pairs) + tuple(
            self.box_link[j] for _, j in self.box_pairs
        )

    @property
    def solid_links(self) -> frozenset[str]:
        """Links with any shape."""
        return frozenset(
            self.link_names[k] for k in (*self.capsule_link, *self.box_link)
        )

    def shapes_at(self, link_poses: jax.Array) -> tuple[Capsule, Box | None]:
        """The shapes placed by ``link_poses`` (``(*batch, num_links, 7)``,
        wxyz_xyz, as pyroki's forward kinematics returns them)."""
        caps = self.capsules.transform(
            jaxlie.SE3(link_poses[..., jnp.asarray(self.capsule_link), :])
        )
        boxes = None
        if self.boxes is not None:
            boxes = self.boxes.transform(
                jaxlie.SE3(link_poses[..., jnp.asarray(self.box_link), :])
            )
        return caps, boxes

    def distances(self, link_poses: jax.Array) -> jax.Array:
        """Signed distance of every pair (``(*batch, num_pairs)``) with the
        links at ``link_poses`` (see :meth:`shapes_at`)."""
        caps, boxes = self.shapes_at(link_poses)

        def pick(geom, idx):
            index = jnp.asarray(idx, dtype=jnp.int32)
            return jax.tree.map(lambda x: x[..., index, :], geom)

        out = []
        if self.capsule_pairs:
            i, j = zip(*self.capsule_pairs)
            out.append(pk.collision.collide(pick(caps, i), pick(caps, j)))
        if self.box_pairs:
            assert boxes is not None
            i, j = zip(*self.box_pairs)
            out.append(capsule_box_distance(pick(caps, i), pick(boxes, j)))
        return jnp.concatenate(out, axis=-1)

    def compute_self_collision_distance(
        self, robot: pk.Robot, cfg: jax.Array
    ) -> jax.Array:
        """Signed distance of every pair at joint configuration ``cfg``."""
        return self.distances(robot.forward_kinematics(cfg))

    # -- builders ------------------------------------------------------------

    @staticmethod
    def from_primitives(
        urdf: yourdfpy.URDF,
        link_names: tuple[str, ...],
        is_arm: Callable[[str], bool],
        is_body: Callable[[str], bool],
    ) -> ArmBodyCollision:
        """Every link's ``<collision>`` boxes and capsules (cylinders), with a
        pair for each arm shape against each body shape. ``link_names``: the
        pyroki robot's link order."""
        caps: list[tuple[int, float, float, np.ndarray]] = []
        boxes: list[tuple[int, np.ndarray, np.ndarray]] = []
        for k, name in enumerate(link_names):
            link = urdf.link_map.get(name)
            for col in [] if link is None else link.collisions:
                pose = np.eye(4) if col.origin is None else np.asarray(col.origin)
                geom = col.geometry
                if geom.cylinder is not None:
                    caps.append((k, geom.cylinder.radius, geom.cylinder.length, pose))
                elif geom.box is not None:
                    if is_arm(name):
                        raise ValueError(f"{name}: arm parts collide as capsules")
                    boxes.append((k, np.asarray(geom.box.size, dtype=float), pose))
                else:
                    raise ValueError(f"{name}: IK collision must be a box or cylinder")

        def wxyz_xyz(pose: np.ndarray) -> np.ndarray:
            return np.asarray(jaxlie.SE3.from_matrix(pose).wxyz_xyz)

        capsules = Capsule.from_radius_height(
            radius=jnp.asarray([c[1] for c in caps]),
            height=jnp.asarray([c[2] for c in caps]),
            position=jnp.asarray([wxyz_xyz(c[3])[4:] for c in caps]),
            wxyz=jnp.asarray([wxyz_xyz(c[3])[:4] for c in caps]),
        )
        box_geom = None
        if boxes:
            box_geom = Box.from_extent(
                extent=jnp.asarray([b[1] for b in boxes]),
                position=jnp.asarray([wxyz_xyz(b[2])[4:] for b in boxes]),
                wxyz=jnp.asarray([wxyz_xyz(b[2])[:4] for b in boxes]),
            )
        cap_link = tuple(c[0] for c in caps)
        box_link = tuple(b[0] for b in boxes)
        arm = [i for i, k in enumerate(cap_link) if is_arm(link_names[k])]
        body_caps = [i for i, k in enumerate(cap_link) if is_body(link_names[k])]
        body_boxes = [i for i, k in enumerate(box_link) if is_body(link_names[k])]
        return ArmBodyCollision(
            link_names=tuple(link_names),
            capsules=capsules,
            capsule_link=cap_link,
            boxes=box_geom,
            box_link=box_link,
            capsule_pairs=tuple((a, b) for a in arm for b in body_caps),
            box_pairs=tuple((a, b) for a in arm for b in body_boxes),
        )
