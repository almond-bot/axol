"""Shared JAX runtime setup for the FK and IK entry points.

JAX JIT-compiles per process: every ``jax.jit`` graph is lowered and compiled
by XLA the first time it runs, and by default the resulting executable lives
only in memory. That makes the IK solver pay the full XLA compile (~tens of
seconds on CPU) on every boot even though nothing changed between sessions.

Pointing ``jax_compilation_cache_dir`` at a stable on-disk location lets XLA
reuse compiled executables across processes. The cache key covers the computed
graph, jax/jaxlib versions, backend, and compile flags, so stale entries are
never reused — a jax upgrade or a change to the solver costs simply recompiles
and rewrites the cache. Note the cache only skips the XLA backend compile;
Python-side tracing and lowering (e.g. ``jaxls`` problem analysis) still runs
each session.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

_logger = logging.getLogger(__name__)


def disable_gpu_preallocation() -> None:
    """Stop XLA from reserving 75% of GPU memory for the tiny FK/IK graphs.

    By default the JAX CUDA backend preallocates 75% of the device memory at
    init. On a Jetson the "device memory" is the unified system RAM, so a
    single JAX process tries to reserve most of the 16 GB the cameras, NVENC,
    and control loops also live in — and serve runs several JAX processes
    (the IK worker subprocess plus in-process FK). On discrete GPUs it would
    likewise starve the policy server of VRAM. The kinematics graphs need a
    few MB, so let them allocate on demand instead.

    Reads at backend init, so this must run before the first JAX op — the
    same contract as :func:`enable_persistent_compilation_cache`. An explicit
    operator setting is honored.
    """
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def enable_persistent_compilation_cache() -> None:
    """Point JAX at an on-disk compilation cache so XLA compiles survive restarts.

    Must be called before the first JIT compilation to benefit that compile.
    A cache dir already configured by the operator (via the
    ``JAX_COMPILATION_CACHE_DIR`` env var, which JAX latches into ``jax.config``
    at import time, or via code) is honored.
    """
    import jax

    if jax.config.jax_compilation_cache_dir:
        return

    cache_dir = Path.home() / ".almond" / "jax-cache"
    try:
        cache_dir.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        _logger.warning(
            "Could not create JAX compilation cache dir %s: %s", cache_dir, e
        )
        return

    jax.config.update("jax_compilation_cache_dir", str(cache_dir))
    # Cache every executable regardless of size or compile time — the solver
    # graph is the dominant cost, but the small FK kernels are free to keep too.
    jax.config.update("jax_persistent_cache_min_entry_size_bytes", -1)
    jax.config.update("jax_persistent_cache_min_compile_time_secs", 0.0)
    _logger.info("JAX persistent compilation cache: %s", cache_dir)
