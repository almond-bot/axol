"""Per-run field suggestions: ``CommandDef.field_suggestions`` → the panel's pick list."""

from __future__ import annotations

import dataclasses
import unittest
from typing import Any

import httpx

from almond_axol.serve import commands
from almond_axol.serve.commands import (
    CommandDef,
    check_strict_fields,
    command_specs,
    field_suggestions,
)

try:
    from .test_serve_session_reservation import _Manager, _Runner, _test_app
except ImportError:
    # ``unittest discover -s tests`` imports test modules top-level (no parent
    # package), so the relative import above fails there; fall back to the
    # absolute name, which resolves because the start dir is on sys.path.
    from test_serve_session_reservation import _Manager, _Runner, _test_app


@dataclasses.dataclass
class _SuggestConfig:
    model_id: str = ""
    task: str = ""


def _command(provider: Any = None, **kwargs: Any) -> CommandDef:
    return CommandDef(
        "suggest-op",
        "suggest-op",
        "Suggest op",
        "An op with a suggested per-run field.",
        "Operate",
        "draccus",
        lambda: _SuggestConfig,
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


def _catalog() -> list[dict[str, Any]]:
    return [{"value": "debur", "label": "Debur"}, {"value": "sanding", "label": None}]


class StrictFieldsTest(unittest.TestCase):
    """``strict_fields``: the pick list is a catalog, not a convenience."""

    def setUp(self) -> None:
        self._saved = dict(commands.COMMANDS)

    def tearDown(self) -> None:
        commands.COMMANDS.clear()
        commands.COMMANDS.update(self._saved)
        commands._schema_cache.pop("suggest-op", None)

    def test_specs_list_the_strict_fields(self) -> None:
        commands.register(_command(_catalog, strict_fields=("model_id",)))
        spec = next(s for s in command_specs() if s["id"] == "suggest-op")
        self.assertEqual(spec["suggestedFields"], ["model_id"])
        self.assertEqual(spec["strictFields"], ["model_id"])
        plain = next(s for s in command_specs() if s["id"] == "teleop")
        self.assertEqual(plain["strictFields"], [])

    def test_a_strict_field_needs_a_provider(self) -> None:
        with self.assertRaises(ValueError) as ctx:
            _command(strict_fields=("model_id",))
        self.assertIn("model_id", str(ctx.exception))
        self.assertIn("field_suggestions", str(ctx.exception))

    def test_a_listed_value_passes_and_an_unlisted_one_is_refused(self) -> None:
        commands.register(_command(_catalog, strict_fields=("model_id",)))
        check_strict_fields("suggest-op", {"model_id": "debur", "task": "anything"})
        check_strict_fields("suggest-op", {"model_id": " sanding "})
        with self.assertRaises(ValueError) as ctx:
            check_strict_fields("suggest-op", {"model_id": "deburr"})
        message = str(ctx.exception)
        self.assertIn("model_id", message)
        self.assertIn("'deburr'", message)
        self.assertIn("debur, sanding", message)

    def test_blank_and_absent_values_are_the_schemas_business(self) -> None:
        commands.register(_command(_catalog, strict_fields=("model_id",)))
        check_strict_fields("suggest-op", {})
        check_strict_fields("suggest-op", {"model_id": ""})
        check_strict_fields("suggest-op", {"model_id": None})

    def test_a_failing_provider_refuses_rather_than_waves_through(self) -> None:
        def boom() -> list[dict[str, Any]]:
            raise RuntimeError("catalog unreadable")

        commands.register(_command(boom, strict_fields=("model_id",)))
        with self.assertRaises(ValueError) as ctx:
            check_strict_fields("suggest-op", {"model_id": "debur"})
        self.assertIn("cannot verify", str(ctx.exception))
        self.assertIn("catalog unreadable", str(ctx.exception))

    def test_free_form_fields_and_unknown_commands_are_left_alone(self) -> None:
        commands.register(_command(_catalog))
        check_strict_fields("suggest-op", {"model_id": "anything goes"})
        check_strict_fields("no-such-op", {"model_id": "anything goes"})


class StrictFieldsApiTest(unittest.IsolatedAsyncioTestCase):
    """``/api/op/start`` refuses an off-list value as a form error (400)."""

    def setUp(self) -> None:
        self._saved = dict(commands.COMMANDS)

    def tearDown(self) -> None:
        commands.COMMANDS.clear()
        commands.COMMANDS.update(self._saved)
        commands._schema_cache.pop("suggest-op", None)

    def _register(self) -> None:
        commands.register(
            _command(
                _catalog,
                strict_fields=("model_id",),
                # An operation with nothing to survey: no CAN bus, no robot.
                entrypoint=lambda: (lambda cfg, **_kw: None),
                uses_can_bus=False,
            )
        )

    async def test_an_off_list_value_is_refused_before_the_runner(self) -> None:
        self._register()
        runner = _Runner()
        transport = httpx.ASGITransport(app=_test_app(_Manager(), runner))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.post(
                "/api/op/start",
                json={"op": "suggest-op", "args": {"model_id": "deburr"}},
            )
        self.assertEqual(response.status_code, 400)
        self.assertIn("'deburr'", response.json()["error"])
        self.assertEqual(runner.starts, 0)

    async def test_a_listed_value_starts(self) -> None:
        self._register()
        runner = _Runner()
        transport = httpx.ASGITransport(app=_test_app(_Manager(), runner))
        async with httpx.AsyncClient(transport=transport, base_url="http://test") as c:
            response = await c.post(
                "/api/op/start",
                json={"op": "suggest-op", "args": {"model_id": "debur"}},
            )
        self.assertEqual(response.status_code, 200, response.text)
        self.assertEqual(runner.starts, 1)


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
