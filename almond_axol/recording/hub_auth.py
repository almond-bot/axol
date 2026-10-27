"""HuggingFace Hub login for the dataset upload at session end.

The installer registers ``axol.service`` as root and points ``HF_LEROBOT_HOME``
at the installing user's ``~/.cache/huggingface/lerobot``, but nothing pointed
huggingface_hub at that user's login: the root service looked for its token in
root's own ``~/.cache/huggingface/token``, found none, and every
``push_to_hub`` failed with 401 Unauthorized after the last episode was saved.

:func:`use_operator_hub_login` sets ``HF_TOKEN_PATH`` to the token beside the
operator's lerobot home, so ``hf auth login`` run as the operator is the login
the service uses. It runs from ``axol serve`` startup, which covers units
installed before this existed (the self-updater never rewrites the unit).

:func:`require_hub_login` checks the login before a session that will upload,
so a missing, rejected or read-only token fails at start rather than after the
operator has recorded the whole session.
"""

from __future__ import annotations

import logging
import os
import shutil
import sys
from pathlib import Path

_logger = logging.getLogger(__name__)

_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HF_TOKEN_PATH")
_TURN_OFF = "or turn off Push to HuggingFace Hub (`--push_to_hub false`)."


def _default_hf_home() -> Path:
    """huggingface_hub's ``HF_HOME`` default, without importing it.

    Importing ``huggingface_hub.constants`` freezes ``HF_TOKEN_PATH`` from the
    environment, so this must not import it before the redirect is set.
    """
    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        return Path(hf_home).expanduser()
    cache = os.environ.get("XDG_CACHE_HOME") or os.path.join("~", ".cache")
    return Path(cache).expanduser() / "huggingface"


def use_operator_hub_login() -> None:
    """Point a root service at the operator's HuggingFace token.

    Only when running as root with ``HF_LEROBOT_HOME`` set (the installer's
    redirect), no token configured through the environment, and no login of
    root's own, so an explicit ``sudo hf auth login`` keeps working. Must run
    before anything imports huggingface_hub.
    """
    if os.geteuid() != 0:
        return
    if any(os.environ.get(name) for name in _TOKEN_ENV_VARS):
        return
    lerobot_home = os.environ.get("HF_LEROBOT_HOME")
    if not lerobot_home:
        return
    if (_default_hf_home() / "token").exists():
        return
    token_path = Path(lerobot_home).expanduser().parent / "token"
    os.environ["HF_TOKEN_PATH"] = str(token_path)
    _logger.info("Using the HuggingFace login at %s for dataset uploads.", token_path)


def _hf_command() -> str:
    """How to run the ``hf`` CLI on this machine.

    ``uv tool install`` exposes only ``axol``, so on an installed robot ``hf``
    is usually in the tool's environment but not on PATH.
    """
    if shutil.which("hf"):
        return "hf"
    beside = Path(sys.executable).parent / "hf"
    return str(beside) if beside.is_file() else "hf"


def _token_source(token: str) -> str:
    from huggingface_hub import constants

    if token in (os.environ.get("HF_TOKEN"), os.environ.get("HUGGING_FACE_HUB_TOKEN")):
        return "the HF_TOKEN environment variable"
    return str(constants.HF_TOKEN_PATH)


def require_hub_login() -> None:
    """Fail before recording if the end-of-session upload can't authenticate.

    Raises ``RuntimeError`` with what to do when there is no token, the Hub
    rejects it, or it is read-only. A network failure only warns: the robot
    may be offline now and online when the session ends.
    """
    from huggingface_hub import HfApi, get_token
    from huggingface_hub.errors import HfHubHTTPError

    hf = _hf_command()
    token = get_token()
    if token is None:
        raise RuntimeError(
            "Not logged in to HuggingFace, so the dataset can't upload when the "
            f"session ends. Run `{hf} auth login` on the robot with a write "
            f"token, {_TURN_OFF}"
        )
    try:
        info = HfApi().whoami(token=token)
    except HfHubHTTPError as exc:
        if exc.response is not None and exc.response.status_code == 401:
            raise RuntimeError(
                f"HuggingFace rejected the login token in {_token_source(token)}. "
                f"Run `{hf} auth login --force` on the robot with a write token, "
                f"{_TURN_OFF}"
            ) from exc
        _logger.warning("Could not check the HuggingFace login: %s", exc)
        return
    except Exception as exc:  # noqa: BLE001 - offline now, maybe not at the end
        _logger.warning(
            "Could not reach HuggingFace to check the login (%s). Recording "
            "continues, but the upload at the end may fail.",
            exc,
        )
        return
    role = (info.get("auth") or {}).get("accessToken", {}).get("role")
    if role == "read":
        raise RuntimeError(
            f"The HuggingFace token in {_token_source(token)} is read-only, so the "
            f"dataset can't upload. Run `{hf} auth login --force` on the robot "
            f"with a write token, {_TURN_OFF}"
        )
    _logger.info(
        "The dataset uploads to HuggingFace as %s when the session ends.",
        info.get("name", "the logged-in user"),
    )
