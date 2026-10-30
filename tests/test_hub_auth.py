"""HuggingFace login for the dataset upload at session end.

The root ``axol.service`` must use the operator's ``hf auth login`` (it used
root's empty token store, so every upload failed with 401 after the session),
and a session that will upload must fail at start when the login can't work.
"""

from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import httpx
from huggingface_hub.errors import HfHubHTTPError

from almond_axol.recording import hub_auth


def _http_error(status: int) -> HfHubHTTPError:
    request = httpx.Request("GET", "https://huggingface.co/api/whoami-v2")
    return HfHubHTTPError("error", response=httpx.Response(status, request=request))


class UseOperatorHubLoginTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        tmp = Path(self._tmp.name)
        self.root_hf_home = tmp / "root-hf"
        self.lerobot_home = tmp / "operator" / ".cache" / "huggingface" / "lerobot"
        self.env = {
            "HF_HOME": str(self.root_hf_home),
            "HF_LEROBOT_HOME": str(self.lerobot_home),
        }

    def _run(self, *, euid: int = 0, env: dict[str, str] | None = None) -> dict:
        environ = dict(self.env if env is None else env)
        with (
            mock.patch.dict(os.environ, environ, clear=True),
            mock.patch.object(hub_auth.os, "geteuid", return_value=euid),
        ):
            hub_auth.use_operator_hub_login()
            return dict(os.environ)

    def test_root_service_uses_the_operator_token(self) -> None:
        env = self._run()
        self.assertEqual(
            env["HF_TOKEN_PATH"],
            str(self.lerobot_home.parent / "token"),
        )

    def test_non_root_is_untouched(self) -> None:
        self.assertNotIn("HF_TOKEN_PATH", self._run(euid=1000))

    def test_without_lerobot_redirect_is_untouched(self) -> None:
        env = self._run(env={"HF_HOME": str(self.root_hf_home)})
        self.assertNotIn("HF_TOKEN_PATH", env)

    def test_roots_own_login_wins(self) -> None:
        self.root_hf_home.mkdir(parents=True)
        (self.root_hf_home / "token").write_text("hf_root")
        self.assertNotIn("HF_TOKEN_PATH", self._run())

    def test_explicit_environment_wins(self) -> None:
        for name, value in (
            ("HF_TOKEN", "hf_env"),
            ("HF_TOKEN_PATH", "/etc/hf/token"),
        ):
            with self.subTest(name=name):
                env = self._run(env={**self.env, name: value})
                self.assertEqual(env[name], value)
                if name != "HF_TOKEN_PATH":
                    self.assertNotIn("HF_TOKEN_PATH", env)


class RequireHubLoginTest(unittest.TestCase):
    def _check(self, *, token: str | None = "hf_x", whoami=None) -> None:
        api = mock.Mock()
        if isinstance(whoami, BaseException):
            api.whoami.side_effect = whoami
        else:
            api.whoami.return_value = whoami or {}
        with (
            mock.patch("huggingface_hub.get_token", return_value=token),
            mock.patch("huggingface_hub.HfApi", return_value=api),
            mock.patch.object(hub_auth, "_hf_command", return_value="hf"),
        ):
            hub_auth.require_hub_login()

    def test_missing_login_fails_before_recording(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"Not logged in.*hf auth login"):
            self._check(token=None)

    def test_rejected_token_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, r"rejected.*auth login --force"):
            self._check(whoami=_http_error(401))

    def test_read_only_token_fails(self) -> None:
        with self.assertRaisesRegex(RuntimeError, "read-only"):
            self._check(
                whoami={"name": "op", "auth": {"accessToken": {"role": "read"}}}
            )

    def test_write_token_passes(self) -> None:
        self._check(whoami={"name": "op", "auth": {"accessToken": {"role": "write"}}})

    def test_offline_only_warns(self) -> None:
        with self.assertLogs(hub_auth._logger, "WARNING"):  # noqa: SLF001
            self._check(whoami=httpx.ConnectError("no route to host"))

    def test_other_http_errors_only_warn(self) -> None:
        with self.assertLogs(hub_auth._logger, "WARNING"):  # noqa: SLF001
            self._check(whoami=_http_error(503))


if __name__ == "__main__":
    unittest.main()
