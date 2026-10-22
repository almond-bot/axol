"""Exported IK programs: skip re-tracing the solver on every start.

JAX's persistent compilation cache (:mod:`.jax_cache`) keeps the compiled XLA
program, but every new process still *traces* each jitted function in Python
and lowers it to StableHLO before it can even look the program up — for the
IK solve (jaxls' Levenberg-Marquardt loop over pyroki's kinematics and
collision costs) that is ~5 s per graph shape on a desktop and 9-15 s on a
Jetson, twice at start-up (with and without elbow hints).

:class:`ExportedJit` wraps such a function: the first time a graph shape is
used it is exported (:mod:`jax.export`) to ``~/.almond/jax-cache/exports``;
every later process deserializes that export and only compiles it, which the
persistent cache then serves. The export's file name digests everything that
shapes the program — the function's static arguments, its inputs' structure,
shapes and dtypes, the solver's source, the robot model it was traced for and
the jax / jaxlib / pyroki / jaxls versions — so a stale program is never
loaded: a change exports anew. Anything unexpected (``flatbuffers`` missing,
an unreadable export, an export that fails) falls back to the plain jitted
function, as before.

Set ``ALMOND_IK_EXPORTS=0`` to always trace.
"""

from __future__ import annotations

import hashlib
import importlib.metadata
import inspect
import logging
import os
import re
import threading
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

_logger = logging.getLogger(__name__)

EXPORT_DIR = Path.home() / ".almond" / "jax-cache" / "exports"

# The code an export embeds: these modules' source goes into every digest.
_SOURCES = ("solver.py", "model.py", "aot.py")
_PACKAGES = ("jax", "jaxlib", "pyroki", "jaxls")
_ADDRESS = re.compile(r"0x[0-9a-fA-F]+")

_lapack_ready = False
_lapack_lock = threading.Lock()


def _ensure_lapack() -> None:
    """Register jaxlib's LAPACK kernels before running a deserialized export.

    jaxlib registers them lazily, the first time the process traces linear
    algebra. A deserialized export skips that trace, so its Cholesky would
    call an unregistered kernel and crash the process.
    """
    global _lapack_ready
    with _lapack_lock:
        if not _lapack_ready:
            jnp.linalg.cholesky(jnp.eye(2, dtype=jnp.float32)).block_until_ready()
            _lapack_ready = True


def _environment_digest() -> str:
    h = hashlib.sha256()
    here = Path(__file__).resolve().parent
    for name in _SOURCES:
        h.update((here / name).read_bytes())
    for pkg in _PACKAGES:
        try:
            h.update(f"{pkg}={importlib.metadata.version(pkg)}".encode())
        except importlib.metadata.PackageNotFoundError:
            h.update(f"{pkg}=?".encode())
    return h.hexdigest()


def _enabled() -> bool:
    if os.environ.get("ALMOND_IK_EXPORTS", "1") == "0":
        return False
    try:
        import flatbuffers  # noqa: F401  (jax.export serialization needs it)
    except ImportError:
        return False
    return True


def _as_leaf(x: Any) -> Any:
    # Python scalars become f32/i32 arrays, so every call (and the export)
    # sees the same input types.
    if isinstance(x, bool):
        return np.asarray(x)
    if isinstance(x, float):
        return np.float32(x)
    if isinstance(x, int):
        return np.int32(x)
    return x


class ExportedJit:
    """A jitted function whose traced program is exported and reused.

    Called like the function itself (``static_argnames`` as for
    :func:`jax.jit`), plus ``_fingerprint=``: a string describing what the
    inputs encode that their shapes do not (the robot model), folded into the
    export's digest.
    """

    def __init__(
        self, fn: Callable[..., Any], name: str, static_argnames: tuple[str, ...] = ()
    ) -> None:
        self._fn = fn
        self._name = name
        self._static = static_argnames
        self._signature = inspect.signature(inspect.unwrap(fn))
        self._lock = threading.Lock()
        self._compiled: dict[Any, Callable[..., Any]] = {}
        self._env: str | None = None

    def __call__(self, *args: Any, _fingerprint: str = "", **kwargs: Any) -> Any:
        if not _enabled():
            return self._fn(*args, **kwargs)
        bound = self._signature.bind(*args, **kwargs)
        statics = {k: v for k, v in bound.arguments.items() if k in self._static}
        dynamic = {k: v for k, v in bound.arguments.items() if k not in self._static}
        leaves, treedef = jax.tree.flatten(dynamic)
        leaves = [_as_leaf(x) for x in leaves]
        avals = tuple((np.shape(x), str(jnp.asarray(x).dtype)) for x in leaves)
        key = (treedef, tuple(sorted(statics.items())), avals, _fingerprint)
        fn = self._compiled.get(key)
        if fn is None:
            with self._lock:
                fn = self._compiled.get(key)
                if fn is None:
                    fn = self._load_or_export(treedef, statics, avals, _fingerprint)
                    self._compiled[key] = fn
        return fn(*leaves)

    def _plain(self, treedef: Any, statics: dict[str, Any]) -> Callable[..., Any]:
        def call(*leaves: Any) -> Any:
            return self._fn(**jax.tree.unflatten(treedef, leaves), **statics)

        return call

    def _load_or_export(
        self,
        treedef: Any,
        statics: dict[str, Any],
        avals: tuple[tuple[tuple[int, ...], str], ...],
        fingerprint: str,
    ) -> Callable[..., Any]:
        from jax import export

        if self._env is None:
            self._env = _environment_digest()
        digest = hashlib.sha256(
            repr(
                (
                    self._name,
                    self._env,
                    _ADDRESS.sub("0x", str(treedef)),
                    sorted(statics.items()),
                    avals,
                    fingerprint,
                )
            ).encode()
        ).hexdigest()[:32]
        path = EXPORT_DIR / f"{self._name}-{digest}.jaxexport"
        _ensure_lapack()
        if path.is_file():
            try:
                exported = export.deserialize(bytearray(path.read_bytes()))
                _logger.info("Loaded exported %s (%s)", self._name, path.name)
                return jax.jit(exported.call)
            except Exception as e:  # noqa: BLE001 - any failure: trace instead
                _logger.warning("Could not load %s (%s); re-exporting", path.name, e)
        plain = self._plain(treedef, statics)
        try:
            exported = export.export(jax.jit(plain))(
                *(jax.ShapeDtypeStruct(shape, dtype) for shape, dtype in avals)
            )
            EXPORT_DIR.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(f".{os.getpid()}.tmp")
            tmp.write_bytes(exported.serialize())
            os.replace(tmp, path)
            _logger.info("Exported %s (%s)", self._name, path.name)
            return jax.jit(exported.call)
        except Exception as e:  # noqa: BLE001 - export is an optimisation only
            _logger.warning("Could not export %s (%s); tracing as usual", self._name, e)
            return jax.jit(plain)
