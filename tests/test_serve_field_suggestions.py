"""Per-run field suggestions: ``CommandDef.field_suggestions`` → the panel's pick list."""

from __future__ import annotations

import unittest
from typing import Any

import httpx

from almond_axol.serve import commands
from almond_axol.serve.commands import CommandDef, command_specs, field_suggestions

try:
    from .test_serve_session_reservation import _Manager, _Runner, _test_app
except ImportError:
    # ``unittest discover -s tests`` imports test modules top-level (no parent
    # package), so the relative import above fails there; fall back to the
    # absolute name, which resolves because the start dir is on sys.path.
    from test_serve_session_reservation import _Manager, _Runner, _test_app


def _command(provider: Any = None, **kwargs: Any) -> CommandDef:
    return CommandDef(
        "suggest-op",
        "suggest-op",
        "Suggest op",
        "An op with a suggested per-run field.",
        "Operate",
        "draccus",
        lambda: None,
        per_run_fields=("model_id", "task"),
        field_suggestions={"model_id": provider} if provider is not None else None,
        **kwargs,
    )


class FieldSuggestionsRegistryTest(unittest.TestCase):
    def setUp(self) -> None:
        self._saved = dict(commands.COMMANDS)

    def tearDown(self) -> None:
        commands.COMMANDS.clear()
        commands.COMMANDS.update(self._saved)
        commands._schema_cache.pop("suggest-op", None)

    def test_specs_list_the_suggested_fields(self) -> None:
        commands.register(_command(lambda: []))
        spec = next(s for s in command_specs() if s["id"] == "suggest-op")
        self.assertEqual(spec["perRunFields"], ["model_id", "task"])
        self.assertEqual(spec["suggestedFields"], ["model_id"])
        plain = next(s for s in command_specs() if s["id"] == "teleop")
        self.assertEqual(plain["suggestedFields"], [])

    def test_rows_are_normalized(self) -> None:
        commands.register(
            _command(
                lambda: [
                    {"value": " ckpt-24999-debur ", "label": "debur-pi07 · step 24999"},
                    {"value": "", "label": "dropped: blank value"},
                    {"value": "ckpt-5000-debur", "label": ""},
                    {"value": "ckpt-1", "label": 7},
                ]
            )
        )
        self.assertEqual(
            field_suggestions("suggest-op", "model_id"),
            [
                {"value": "ckpt-24999-debur", "label": "debur-pi07 · step 24999"},
                {"value": "ckpt-5000-debur", "label": None},
                {"value": "ckpt-1", "label": "7"},
            ],
        )

    def test_undeclared_command_or_field_is_a_key_error(self) -> None:
        commands.register(_command(lambda: []))
        with self.assertRaises(KeyError):
            field_suggestions("suggest-op", "task")
        with self.assertRaises(KeyError):
            field_suggestions("no-such-op", "model_id")
        commands.register(_command())
        with self.assertRaises(KeyError):
            field_suggestions("suggest-op", "model_id")


class FieldSuggestionsApiTest(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self._saved = dict(commands.COMMANDS)

    def tearDown(self) -> None:
        commands.COMMANDS.clear()
        commands.COMMANDS.update(self._saved)
        commands._schema_cache.pop("suggest-op", None)

    def _client(self) -> httpx.AsyncClient:
        transport = httpx.ASGITransport(app=_test_app(_Manager(), _Runner()))
        return httpx.AsyncClient(transport=transport, base_url="http://test")

    async def test_returns_the_provider_rows(self) -> None:
        commands.register(_command(lambda: [{"value": "ckpt-1", "label": "run a"}]))
        async with self._client() as client:
            response = await client.get("/api/commands/suggest-op/suggestions/model_id")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"suggestions": [{"value": "ckpt-1", "label": "run a"}], "error": None},
        )

    async def test_provider_failure_is_reported_next_to_an_empty_list(self) -> None:
        def boom() -> list[dict[str, Any]]:
            raise RuntimeError("registry unreachable")

        commands.register(_command(boom))
        async with self._client() as client:
            response = await client.get("/api/commands/suggest-op/suggestions/model_id")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(
            response.json(),
            {"suggestions": [], "error": "RuntimeError: registry unreachable"},
        )

    async def test_provider_key_error_is_a_failure_not_a_404(self) -> None:
        def lookup() -> list[dict[str, Any]]:
            return {}["missing-registry-entry"]

        commands.register(_command(lookup))
        async with self._client() as client:
            response = await client.get("/api/commands/suggest-op/suggestions/model_id")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json()["suggestions"], [])
        self.assertIn("KeyError", response.json()["error"])

    async def test_undeclared_field_is_not_found(self) -> None:
        commands.register(_command(lambda: []))
        async with self._client() as client:
            missing = await client.get("/api/commands/suggest-op/suggestions/task")
            unknown = await client.get("/api/commands/nope/suggestions/model_id")
        self.assertEqual(missing.status_code, 404)
        self.assertEqual(unknown.status_code, 404)
