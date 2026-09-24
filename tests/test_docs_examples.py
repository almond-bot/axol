"""Every code example in ``docs/`` and ``README.md`` is checked against the package.

A user copying an example out of the documentation must get working code, so
this module extracts every fenced block from the ``.mdx`` pages and verifies it
against the *real* API rather than against a hand-maintained copy:

- ``python`` blocks parse, every import resolves, every keyword argument and
  attribute used on a known object exists on it (signatures are introspected
  with :mod:`inspect`), documented call signatures (``Axol(...)`` blocks)
  match the constructor, and the hardware-free examples are executed.
- ``bash`` blocks: every ``axol <command> ...`` line is fed to the command's
  own parser (argparse tree or draccus config) with the shared settings file
  pointed at an empty temp path, so unknown flags and bad values fail here
  instead of on the robot.
- ``json`` blocks are loaded through the loader that reads that file on the
  robot (settings store, waypoint file, tracker config, ...).
- API tables (``| Method |`` / ``| Field |`` / ``| Value |``) name real
  attributes, dataclass fields, or enum members of the class they document.

Blocks that need hardware are never executed; they are checked statically.
Set ``AXOL_DOCS_RUN_SLOW=1`` to also execute the examples that JIT-compile the
IK solver (tens of seconds cold).
"""

from __future__ import annotations

import argparse
import ast
import builtins
import contextlib
import dataclasses
import importlib
import inspect
import io
import json
import os
import re
import shlex
import tempfile
import tomllib
import types
import typing
import unittest
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest.mock import patch

REPO = Path(__file__).resolve().parent.parent
DOC_FILES = sorted((REPO / "docs").rglob("*.mdx")) + [REPO / "README.md"]

_FENCE = re.compile(
    r"^(?P<indent>[ \t]*)```(?P<info>[^\n]*)\n(?P<code>.*?)^(?P=indent)```", re.M | re.S
)

RUN_SLOW = os.environ.get("AXOL_DOCS_RUN_SLOW") == "1"


@dataclass(frozen=True)
class Block:
    path: Path
    line: int
    lang: str
    code: str

    @property
    def page(self) -> str:
        return self.path.relative_to(REPO).as_posix()

    @property
    def where(self) -> str:
        return f"{self.page}:{self.line}"


def _iter_blocks() -> list[Block]:
    blocks: list[Block] = []
    for path in DOC_FILES:
        text = path.read_text(encoding="utf-8")
        for match in _FENCE.finditer(text):
            info = match.group("info").strip()
            lang = info.split()[0] if info else ""
            line = text.count("\n", 0, match.start()) + 2  # first code line
            blocks.append(Block(path, line, lang, match.group("code")))
    return blocks


ALL_BLOCKS = _iter_blocks()
PYTHON_BLOCKS = [b for b in ALL_BLOCKS if b.lang == "python"]
BASH_BLOCKS = [b for b in ALL_BLOCKS if b.lang == "bash"]
JSON_BLOCKS = [b for b in ALL_BLOCKS if b.lang == "json"]


def _blank_settings_path() -> Path:
    """A settings.json path that does not exist: the built-in defaults."""
    return Path(tempfile.mkdtemp(prefix="axol-docs-")) / "settings.json"


# ---------------------------------------------------------------------------
# Python examples
# ---------------------------------------------------------------------------

_SIGNATURE_BLOCK = re.compile(
    r"\A(?P<name>[A-Za-z_]\w*)\(\n(?P<params>.*)\n\)\s*\Z", re.S
)

# Free variables an example fragment may legitimately leave to the reader
# ("cam", "robot", ...), by page. The value is the dotted path of the object's
# type when the fragment calls methods on it, else ``None``.
FRAGMENT_CONTEXT: dict[str, dict[str, str | None]] = {
    "docs/api/teleop.mdx": {"axol": "almond_axol.robot.Axol", "cam": None},
    "docs/api/kinematics.mdx": {
        "robot": "almond_axol.robot.RobotBase",
        "q_a": None,
        "q_b": None,
    },
    "docs/api/vr.mdx": {
        "vr": "almond_axol.vr.VRServer",
        "overhead_cam": None,
        "left_cam": None,
        "right_cam": None,
    },
    "docs/api/lerobot.mdx": {"robot": "almond_axol.lerobot.robot.AxolRobot"},
    "docs/operations/custom-policy.mdx": {"load_my_model": None},
}

# Callables whose documented examples can run in the test process: they touch
# no hardware, network, or subprocess. Everything else stays static.
PURE_CALLABLES = (
    "dataclasses.replace",
    "almond_axol.robot.AxolConfig",
    "almond_axol.robot.ArmConfig",
    "almond_axol.robot.JointConfig",
    "almond_axol.robot.FrictionParams",
    "almond_axol.robot.PositionForceConfig",
    "almond_axol.robot.Jelly",
    "almond_axol.robot.JellyConfig",
    "almond_axol.kinematics.KinematicsConfig",
    "almond_axol.vr.VRServerConfig",
    "almond_axol.teleop.VRTeleopConfig",
    "almond_axol.settings.load_store",
    "almond_axol.settings.shared_axol_config",
    "almond_axol.settings.shared_config",
    "almond_axol.settings.shared_overlay",
)
SLOW_CALLABLES = ("almond_axol.kinematics.KinematicsSolver",)


def _import_dotted(dotted: str) -> Any:
    module_name, _, attr = dotted.rpartition(".")
    return getattr(importlib.import_module(module_name), attr)


def _ensure_zed_camera_classes() -> None:
    """Make ``almond_axol.lerobot.camera`` export ``ZedCamera`` without the ZED SDK.

    The package only re-exports the camera classes when ``pyzed`` (installed by
    ``axol zed.install``, never from PyPI) imports, so the documented
    ``from almond_axol.lerobot.camera import ZedCamera`` is correct on a
    provisioned robot but fails on a development machine. For the *static*
    checks here, import ``camera_zed`` once against a stand-in ``pyzed`` and
    reload the package so the names exist; the stand-in is removed again so
    nothing else in the process can mistake it for the real SDK.
    """
    import sys
    from unittest.mock import MagicMock

    package = importlib.import_module("almond_axol.lerobot.camera")
    if hasattr(package, "ZedCamera") or importlib.util.find_spec("pyzed") is not None:
        return
    stubs = {"pyzed": MagicMock(name="pyzed"), "pyzed.sl": MagicMock(name="pyzed.sl")}
    stubs["pyzed"].sl = stubs["pyzed.sl"]
    sys.modules.update(stubs)
    try:
        importlib.import_module("almond_axol.lerobot.camera.camera_zed")
        importlib.reload(package)
    finally:
        for name in stubs:
            sys.modules.pop(name, None)


def _parse_python(block: Block) -> ast.Module:
    code = block.code
    if _SIGNATURE_BLOCK.match(code):
        code = f"def {code.rstrip()}: ..."
    return compile(
        code,
        block.where,
        "exec",
        flags=ast.PyCF_ONLY_AST | ast.PyCF_ALLOW_TOP_LEVEL_AWAIT,
        dont_inherit=True,
    )


def _resolve_import(node: ast.Import | ast.ImportFrom) -> dict[str, Any]:
    """Execute the import statement's lookups, returning ``{bound_name: object}``."""
    bound: dict[str, Any] = {}
    if isinstance(node, ast.Import):
        for alias in node.names:
            module = importlib.import_module(alias.name)
            if alias.asname:
                bound[alias.asname] = module
            else:
                bound[alias.name.split(".")[0]] = importlib.import_module(
                    alias.name.split(".")[0]
                )
        return bound
    assert node.module is not None and node.level == 0, "docs use absolute imports"
    module = importlib.import_module(node.module)
    if node.module == "almond_axol.lerobot.camera":
        _ensure_zed_camera_classes()
    for alias in node.names:
        if not hasattr(module, alias.name):
            raise ImportError(f"cannot import name {alias.name!r} from {node.module!r}")
        bound[alias.asname or alias.name] = getattr(module, alias.name)
    return bound


def _page_imports(page: str) -> dict[str, Any]:
    """Every name imported anywhere on a page: the page's shared namespace."""
    names: dict[str, Any] = {}
    for block in PYTHON_BLOCKS:
        if block.page != page:
            continue
        for node in ast.walk(_parse_python(block)):
            if isinstance(node, (ast.Import, ast.ImportFrom)):
                names.update(_resolve_import(node))
    return names


class _BoundNames(ast.NodeVisitor):
    """Names a block binds itself (assignments, imports, defs, targets)."""

    def __init__(self) -> None:
        self.names: set[str] = set()

    def _bind_target(self, target: ast.expr) -> None:
        if isinstance(target, ast.Name):
            self.names.add(target.id)
        elif isinstance(target, (ast.Tuple, ast.List)):
            for elt in target.elts:
                self._bind_target(elt)
        elif isinstance(target, ast.Starred):
            self._bind_target(target.value)

    def visit_Import(self, node: ast.Import) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name.split(".")[0])

    def visit_ImportFrom(self, node: ast.ImportFrom) -> None:
        for alias in node.names:
            self.names.add(alias.asname or alias.name)

    def visit_Assign(self, node: ast.Assign) -> None:
        for target in node.targets:
            self._bind_target(target)
        self.generic_visit(node)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_AugAssign(self, node: ast.AugAssign) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_NamedExpr(self, node: ast.NamedExpr) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_For(self, node: ast.For) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    visit_AsyncFor = visit_For

    def visit_comprehension(self, node: ast.comprehension) -> None:
        self._bind_target(node.target)
        self.generic_visit(node)

    def visit_withitem(self, node: ast.withitem) -> None:
        if node.optional_vars is not None:
            self._bind_target(node.optional_vars)
        self.generic_visit(node)

    def _visit_def(self, node: ast.FunctionDef | ast.AsyncFunctionDef) -> None:
        self.names.add(node.name)
        args = node.args
        for arg in (*args.posonlyargs, *args.args, *args.kwonlyargs):
            self.names.add(arg.arg)
        if args.vararg:
            self.names.add(args.vararg.arg)
        if args.kwarg:
            self.names.add(args.kwarg.arg)
        self.generic_visit(node)

    visit_FunctionDef = _visit_def
    visit_AsyncFunctionDef = _visit_def

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        self.names.add(node.name)
        self.generic_visit(node)

    def visit_ExceptHandler(self, node: ast.ExceptHandler) -> None:
        if node.name:
            self.names.add(node.name)
        self.generic_visit(node)


def _free_names(tree: ast.AST) -> set[str]:
    bound = _BoundNames()
    bound.visit(tree)
    loaded = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Load)
    }
    return loaded - bound.names - set(dir(builtins))


def _unwrap_optional(hint: Any) -> Any:
    origin = typing.get_origin(hint)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(hint) if a is not type(None)]
        return args[0] if len(args) == 1 else None
    return hint


def _instance_attributes(cls: type) -> set[str]:
    """``self.<name>`` assignments in ``__init__`` across the MRO."""
    names: set[str] = set()
    for klass in cls.__mro__:
        init = klass.__dict__.get("__init__")
        if init is None:
            continue
        try:
            source = inspect.getsource(init)
        except (OSError, TypeError):
            continue
        tree = ast.parse("if True:\n" + source if source[0].isspace() else source)
        for node in ast.walk(tree):
            targets: list[ast.expr] = []
            if isinstance(node, ast.Assign):
                targets = node.targets
            elif isinstance(node, (ast.AnnAssign, ast.AugAssign)):
                targets = [node.target]
            for target in targets:
                if (
                    isinstance(target, ast.Attribute)
                    and isinstance(target.value, ast.Name)
                    and target.value.id == "self"
                ):
                    names.add(target.attr)
    return names


def _dataclass_field_type(cls: type, name: str) -> Any:
    if not dataclasses.is_dataclass(cls):
        return None
    try:
        hints = typing.get_type_hints(cls)
    except Exception:  # noqa: BLE001 - forward refs the module can't resolve
        return None
    return _unwrap_optional(hints.get(name))


def _has_member(owner: Any, name: str) -> bool:
    if hasattr(owner, name):
        return True
    if dataclasses.is_dataclass(owner) and name in {
        f.name for f in dataclasses.fields(owner)
    }:
        return True
    if isinstance(owner, type):
        model_fields = getattr(owner, "model_fields", None)  # pydantic
        if isinstance(model_fields, dict) and name in model_fields:
            return True
        return name in _instance_attributes(owner)
    return False


def _member_type(owner: Any, name: str) -> Any:
    """Best-effort static type of ``owner.name``, or ``None`` when unknown."""
    if isinstance(owner, types.ModuleType):
        value = getattr(owner, name, None)
        return value if isinstance(value, (type, types.ModuleType)) else None
    if not isinstance(owner, type):
        return None
    field_type = _dataclass_field_type(owner, name)
    if isinstance(field_type, type):
        return field_type
    member = inspect.getattr_static(owner, name, None)
    if isinstance(member, property) and member.fget is not None:
        try:
            hint = _unwrap_optional(typing.get_type_hints(member.fget).get("return"))
        except Exception:  # noqa: BLE001
            return None
        return hint if isinstance(hint, type) else None
    if isinstance(member, type):  # nested class / enum member owner
        return member
    return None


class _ApiChecker(ast.NodeVisitor):
    """Check every call and attribute in a block against the real objects.

    ``namespace`` maps names to the objects the block's imports bind (classes,
    functions, modules). Variables assigned from ``Cls(...)`` or bound with
    ``with Cls(...) as x`` are typed as ``Cls``; attribute chains through
    dataclass fields and annotated properties are followed as far as their
    types are known. Anything untyped is skipped, never reported.
    """

    def __init__(self, namespace: dict[str, Any], var_types: dict[str, Any]) -> None:
        self.namespace = namespace
        self.var_types = dict(var_types)
        self.problems: list[str] = []

    # -- type inference ----------------------------------------------------

    def _type_of(self, node: ast.expr) -> Any:
        """A class or module the expression evaluates to / is an instance of."""
        if isinstance(node, ast.Name):
            if node.id in self.var_types:
                return self.var_types[node.id]
            value = self.namespace.get(node.id)
            return value if isinstance(value, (type, types.ModuleType)) else None
        if isinstance(node, ast.Call):
            callee = self._callee(node)
            return callee if isinstance(callee, type) else None
        if isinstance(node, ast.Attribute):
            owner = self._type_of(node.value)
            if owner is None:
                return None
            return _member_type(owner, node.attr)
        return None

    def _callee(self, node: ast.Call) -> Any:
        func = node.func
        if isinstance(func, ast.Name):
            if func.id in self.var_types:
                return None  # calling an instance
            return self.namespace.get(func.id)
        if isinstance(func, ast.Attribute):
            owner = self._type_of(func.value)
            if owner is None:
                return None
            return inspect.getattr_static(owner, func.attr, None)
        return None

    # -- visitors ----------------------------------------------------------

    def _bind(self, target: ast.expr, value: ast.expr) -> None:
        if isinstance(target, ast.Name):
            inferred = self._type_of(value)
            if inferred is not None and not isinstance(inferred, types.ModuleType):
                # ``x = Cls`` (a class object) vs ``x = Cls()`` (an instance): only
                # the instance case types the variable by its class.
                if isinstance(value, (ast.Call, ast.Attribute, ast.Name)) and not (
                    isinstance(value, ast.Name) and value.id in self.namespace
                ):
                    self.var_types[target.id] = inferred

    def visit_Assign(self, node: ast.Assign) -> None:
        self.generic_visit(node)
        for target in node.targets:
            self._bind(target, node.value)

    def visit_AnnAssign(self, node: ast.AnnAssign) -> None:
        self.generic_visit(node)
        if node.value is not None:
            self._bind(node.target, node.value)

    def visit_withitem(self, node: ast.withitem) -> None:
        self.generic_visit(node)
        if node.optional_vars is not None:
            self._bind(node.optional_vars, node.context_expr)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        self.generic_visit(node)
        owner = self._type_of(node.value)
        if owner is None:
            return
        if not _has_member(owner, node.attr):
            self.problems.append(
                f"{ast.unparse(node)}: {_describe(owner)} has no attribute {node.attr!r}"
            )

    def visit_Call(self, node: ast.Call) -> None:
        self.generic_visit(node)
        callee = self._callee(node)
        if callee is None:
            return
        is_method = isinstance(node.func, ast.Attribute) and isinstance(
            self._type_of(node.func.value), type
        )
        if isinstance(callee, property):
            self.problems.append(
                f"{ast.unparse(node.func)} is a property, not a method"
            )
            return
        if not callable(callee):
            self.problems.append(f"{ast.unparse(node.func)} is not callable")
            return
        try:
            signature = inspect.signature(callee)
        except (TypeError, ValueError):
            return  # builtins / C functions without introspectable signatures
        if is_method and not isinstance(callee, (staticmethod, classmethod, type)):
            params = list(signature.parameters.values())
            if params and params[0].name in ("self", "cls"):
                signature = signature.replace(parameters=params[1:])
        if any(kw.arg is None for kw in node.keywords) or any(
            isinstance(a, ast.Starred) for a in node.args
        ):
            return  # ``**kwargs`` / ``*args`` splats: nothing to check statically
        positional = [object()] * len(node.args)
        keywords = {kw.arg: object() for kw in node.keywords if kw.arg is not None}
        try:
            signature.bind_partial(*positional, **keywords)
        except TypeError as exc:
            self.problems.append(f"{ast.unparse(node.func)}(...): {exc}")


def _describe(owner: Any) -> str:
    if isinstance(owner, types.ModuleType):
        return f"module {owner.__name__}"
    return f"{owner.__module__}.{owner.__qualname__}"


def _fragment_context(page: str) -> dict[str, Any]:
    return {
        name: (_import_dotted(dotted) if dotted else None)
        for name, dotted in FRAGMENT_CONTEXT.get(page, {}).items()
    }


def _parse_documented_signature(block: Block) -> tuple[str, list[ast.arg], list[Any]]:
    """``Name(\\n    a: T = d,\\n    *,\\n    b=...\\n)`` → (name, args, defaults)."""
    match = _SIGNATURE_BLOCK.match(block.code)
    assert match is not None
    tree = _parse_python(block)
    func = tree.body[0]
    assert isinstance(func, ast.FunctionDef)
    return match.group("name"), func.args, [func]


class DocsPythonExamplesTest(unittest.TestCase):
    """The ``python`` blocks in ``docs/api/*.mdx``."""

    def test_docs_contain_python_examples(self) -> None:
        self.assertGreaterEqual(len(PYTHON_BLOCKS), 30, "fence extraction regressed")

    def test_python_blocks_are_valid_syntax(self) -> None:
        for block in PYTHON_BLOCKS:
            with self.subTest(block=block.where):
                _parse_python(block)

    def test_python_imports_resolve(self) -> None:
        for block in PYTHON_BLOCKS:
            with self.subTest(block=block.where):
                for node in ast.walk(_parse_python(block)):
                    if isinstance(node, (ast.Import, ast.ImportFrom)):
                        _resolve_import(node)

    def test_python_blocks_only_use_imported_or_documented_names(self) -> None:
        """No example uses a name it neither imports nor its page explains."""
        for block in PYTHON_BLOCKS:
            if _SIGNATURE_BLOCK.match(block.code):
                continue
            with self.subTest(block=block.where):
                allowed = set(_page_imports(block.page)) | set(
                    FRAGMENT_CONTEXT.get(block.page, {})
                )
                undefined = _free_names(_parse_python(block)) - allowed
                self.assertFalse(
                    undefined,
                    f"{block.where} uses undefined names {sorted(undefined)}; "
                    "import them in the block or list them in FRAGMENT_CONTEXT",
                )

    def test_python_blocks_match_the_real_api(self) -> None:
        """Keyword arguments, methods and attributes exist on the real objects."""
        for block in PYTHON_BLOCKS:
            if _SIGNATURE_BLOCK.match(block.code):
                continue
            with self.subTest(block=block.where):
                checker = _ApiChecker(
                    _page_imports(block.page), _fragment_context(block.page)
                )
                checker.visit(_parse_python(block))
                self.assertFalse(
                    checker.problems,
                    f"{block.where}:\n  " + "\n  ".join(checker.problems),
                )

    def test_documented_signatures_match_constructors(self) -> None:
        """``Axol(\\n  config: ... = None,\\n  ...)`` blocks describe the real signature."""
        checked = 0
        for block in PYTHON_BLOCKS:
            if not _SIGNATURE_BLOCK.match(block.code):
                continue
            checked += 1
            with self.subTest(block=block.where):
                name, args, _ = _parse_documented_signature(block)
                target = _page_imports(block.page).get(name)
                self.assertIsNotNone(target, f"{name} is not imported on {block.page}")
                real = inspect.signature(target)
                documented_positional = [*args.posonlyargs, *args.args]
                defaults = [None] * (
                    len(documented_positional) - len(args.defaults)
                ) + list(args.defaults)
                for arg, default in zip(documented_positional, defaults, strict=True):
                    self._assert_param(real, arg, default, keyword_only=False)
                for arg, default in zip(args.kwonlyargs, args.kw_defaults, strict=True):
                    self._assert_param(real, arg, default, keyword_only=True)
        self.assertGreater(checked, 0, "no signature blocks found")

    def _assert_param(
        self,
        real: inspect.Signature,
        arg: ast.arg,
        default: ast.expr | None,
        *,
        keyword_only: bool,
    ) -> None:
        self.assertIn(arg.arg, real.parameters, f"documented parameter {arg.arg!r}")
        param = real.parameters[arg.arg]
        if keyword_only:
            self.assertEqual(param.kind, inspect.Parameter.KEYWORD_ONLY, arg.arg)
        else:
            self.assertIn(
                param.kind,
                (
                    inspect.Parameter.POSITIONAL_ONLY,
                    inspect.Parameter.POSITIONAL_OR_KEYWORD,
                ),
                f"{arg.arg} is documented positional but is {param.kind.description}",
            )
        if default is not None:
            self.assertIsNot(
                param.default, inspect.Parameter.empty, f"{arg.arg} has no default"
            )
            self.assertEqual(
                ast.unparse(default), repr(param.default), f"default of {arg.arg}"
            )

    def test_hardware_free_examples_execute(self) -> None:
        """Examples that only build configs / read settings actually run."""
        pure = {_import_dotted(d) for d in PURE_CALLABLES}
        if RUN_SLOW:
            pure |= {_import_dotted(d) for d in SLOW_CALLABLES}
        executed = 0
        for block in PYTHON_BLOCKS:
            if _SIGNATURE_BLOCK.match(block.code):
                continue
            namespace = _page_imports(block.page)
            tree = _parse_python(block)
            if not _is_executable(tree, namespace, pure):
                continue
            executed += 1
            with (
                self.subTest(block=block.where),
                patch(
                    "almond_axol.serve.settings.SETTINGS_PATH", _blank_settings_path()
                ),
                contextlib.redirect_stdout(io.StringIO()),
            ):
                exec(compile(tree, block.where, "exec"), dict(namespace))  # noqa: S102
        self.assertGreaterEqual(executed, 5, "expected several runnable examples")


def _is_executable(tree: ast.Module, namespace: dict[str, Any], pure: set[Any]) -> bool:
    """True when every call in the block is to a known side-effect-free callable."""
    if _free_names(tree) - set(namespace):
        return False
    banned = (ast.Await, ast.AsyncWith, ast.AsyncFor, ast.While, ast.Try, ast.Global)
    if any(isinstance(node, banned) for node in ast.walk(tree)):
        return False
    checker = _ApiChecker(namespace, {})
    checker.visit(tree)  # populates var_types
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name):
            target = namespace.get(func.id)
            if target is builtins.print:
                continue
            if not any(target is p for p in pure):
                return False
        elif isinstance(func, ast.Attribute):
            owner = checker._type_of(func.value)
            if isinstance(owner, types.ModuleType):
                if owner.__name__.split(".")[0] != "numpy":
                    return False
            elif not any(owner is p for p in pure):
                return False
        else:
            return False
    return True


# ---------------------------------------------------------------------------
# Bash examples: ``axol ...`` command lines
# ---------------------------------------------------------------------------


class _Parsed(Exception):
    """Raised by the patched parsers once argv has been accepted."""

    def __init__(self, value: Any) -> None:
        super().__init__(value)
        self.value = value


def _logical_lines(code: str) -> Iterable[str]:
    """Join ``\\``-continued lines; drop blank and comment-only lines."""
    pending = ""
    for raw in code.splitlines():
        line = pending + raw.rstrip()
        if line.endswith("\\"):
            pending = line[:-1] + " "
            continue
        pending = ""
        if line.strip() and not line.lstrip().startswith("#"):
            yield line
    if pending.strip():
        yield pending


def _axol_commands() -> list[tuple[Block, str, list[str]]]:
    """Every ``axol <command> ...`` line in the docs as ``(block, line, tokens)``."""
    commands: list[tuple[Block, str, list[str]]] = []
    for block in BASH_BLOCKS:
        for line in _logical_lines(block.code):
            try:
                tokens = shlex.split(line, comments=True)
            except ValueError:
                continue
            if tokens and tokens[0] == "sudo":
                tokens = tokens[1:]
            if tokens and tokens[0] == "$(command -v axol)":
                tokens[0] = "axol"
            if tokens[:2] == ["uv", "run"]:
                tokens = tokens[2:]
            if not tokens or tokens[0] != "axol":
                continue
            commands.append((block, line, tokens[1:]))
    return commands


def _substitute_placeholders(argv: list[str]) -> list[str]:
    """Make the docs' illustrative values parseable: ``<ip>`` and ``...``."""
    out: list[str] = []
    for token in argv:
        if token == "...":
            continue
        if token.startswith("<") and token.endswith(">"):
            token = "placeholder"
        out.append(token)
    return out


def _required_placeholders(config_class: type) -> dict[str, Any]:
    """Dummy values for the fields a ``--config_path`` file must supply."""
    values: dict[str, Any] = {}
    hints = typing.get_type_hints(config_class)
    for field in dataclasses.fields(config_class):
        if field.default is not dataclasses.MISSING or (
            field.default_factory is not dataclasses.MISSING
        ):
            continue
        hint = hints.get(field.name, str)
        if typing.get_origin(hint) is typing.Literal:
            values[field.name] = typing.get_args(hint)[0]
        elif hint is int:
            values[field.name] = 0
        elif hint is float:
            values[field.name] = 0.0
        else:
            values[field.name] = "placeholder"
    return values


def _materialize_files(
    argv: list[str], scratch: Path, config_class: type | None
) -> list[str]:
    """Point ``--config_path`` / ``--settings_path`` at real files.

    The docs name files the reader would have written (``my_teleop.json``,
    ``rollout.yaml``); stand-ins are created here holding the config's required
    fields (for ``--config_path``) or an empty settings document.
    """
    out: list[str] = []
    pending: str | None = None
    for token in argv:
        if pending is not None:
            path = scratch / f"{pending.lstrip('-')}{Path(token).suffix or '.json'}"
            if pending == "--config_path" and config_class is not None:
                path.write_text(
                    json.dumps(_required_placeholders(config_class)), encoding="utf-8"
                )
            else:
                path.write_text("{}\n", encoding="utf-8")
            out.append(str(path))
            pending = None
            continue
        if token in ("--config_path", "--settings_path"):
            pending = token
        out.append(token)
    return out


def _parse_axol_command(argv: list[str], scratch: Path) -> Any:
    """Run ``argv`` through the parser that ``axol`` would use, without dispatching."""
    from almond_axol import cli

    command, rest = argv[0], _substitute_placeholders(argv[1:])
    if command in cli._DRACCUS_COMMANDS:
        module_name, _ = cli._DRACCUS_COMMANDS[command]
        if module_name == "mantis_train":
            return None  # lerobot-train's own parser; only the command is ours
        module = importlib.import_module(f"almond_axol.cli.{module_name}")
        real_parse = module.parse

        def parse_then_stop(config_class: type, argv: list[str], **kwargs: Any) -> Any:
            argv = _materialize_files(argv, scratch, config_class)
            raise _Parsed(real_parse(config_class, argv, **kwargs))

        with patch.object(module, "parse", parse_then_stop):
            try:
                module.main(rest)
            except _Parsed as done:
                return done.value
            raise AssertionError("main() returned without parsing its config")

    def parse_then_stop(
        self: argparse.ArgumentParser, args: Any = None, ns: Any = None
    ) -> Any:
        raise _Parsed(real_parse_args(self, args, ns))

    real_parse_args = argparse.ArgumentParser.parse_args
    rest = _materialize_files(rest, scratch, None)
    if command in cli._DIAG_COMMANDS:
        module_name, _ = cli._DIAG_COMMANDS[command]
        module = importlib.import_module(module_name)
        with patch.object(argparse.ArgumentParser, "parse_args", parse_then_stop):
            try:
                module.main(rest)
            except _Parsed as done:
                return done.value
        raise AssertionError("main() returned without parsing argv")
    return cli.build_parser().parse_args([command, *rest])


class DocsCliExamplesTest(unittest.TestCase):
    """Every ``axol ...`` line in a ``bash`` block parses with the real CLI."""

    def test_docs_contain_cli_examples(self) -> None:
        self.assertGreaterEqual(len(_axol_commands()), 100, "bash extraction regressed")

    def test_every_documented_axol_command_parses(self) -> None:
        scratch = Path(tempfile.mkdtemp(prefix="axol-docs-cli-"))
        settings = scratch / "settings.json"  # absent → built-in defaults
        for block, line, argv in _axol_commands():
            with (
                self.subTest(block=block.where, command=line.strip()),
                patch("almond_axol.serve.settings.SETTINGS_PATH", settings),
                contextlib.redirect_stderr(io.StringIO()) as stderr,
                contextlib.redirect_stdout(io.StringIO()),
            ):
                try:
                    _parse_axol_command(argv, scratch)
                except ModuleNotFoundError as exc:
                    if exc.name == "lerobot" or (exc.name or "").startswith("lerobot."):
                        raise unittest.SkipTest("lerobot extra not installed") from exc
                    raise
                except SystemExit as exc:
                    self.fail(
                        f"`axol {shlex.join(argv)}` rejected by its parser "
                        f"(exit {exc.code}):\n{stderr.getvalue().strip()}"
                    )

    def test_documented_uv_extras_exist(self) -> None:
        """``uv sync --extra X`` names an extra declared in pyproject.toml."""
        pyproject = tomllib.loads((REPO / "pyproject.toml").read_text(encoding="utf-8"))
        extras = set(pyproject["project"].get("optional-dependencies", {}))
        seen = 0
        for block in BASH_BLOCKS:
            for line in _logical_lines(block.code):
                tokens = shlex.split(line, comments=True)
                if tokens[:2] != ["uv", "sync"]:
                    continue
                for i, token in enumerate(tokens):
                    if token == "--extra":
                        seen += 1
                        with self.subTest(block=block.where, extra=tokens[i + 1]):
                            self.assertIn(tokens[i + 1], extras)
        self.assertGreater(seen, 0)


# ---------------------------------------------------------------------------
# JSON examples: config files loaded by the code that reads them on the robot
# ---------------------------------------------------------------------------


def _load_json_example(block: Block, scratch: Path) -> None:
    data = json.loads(block.code)
    path = scratch / f"example-{block.line}.json"
    path.write_text(block.code, encoding="utf-8")

    if isinstance(data, dict) and data.get("version") == 2 and "axol" in data:
        # ~/.almond/settings.json
        from almond_axol.serve import settings as serve_settings
        from almond_axol.settings import shared_axol_config, shared_overlay

        store = serve_settings.SettingsStore(path, strict=True)
        unknown = [
            k for k in store.snapshot()["values"] if not serve_settings.is_known_key(k)
        ]
        if unknown:
            raise AssertionError(f"unknown settings keys {unknown}")
        shared_axol_config(store)
        for op in ("teleop", "gravity-comp"):
            shared_overlay(op, store=store)
        return
    if isinstance(data, dict) and "waypoints" in data:
        from almond_axol.waypoints import WaypointSet

        loaded = WaypointSet.load(path)
        if len(loaded) != len(data["waypoints"]):
            raise AssertionError("waypoint file example did not round-trip")
        return
    if isinstance(data, dict) and "bindings" in data and "backend" in data:
        # ~/.almond/tracker/config.json
        from almond_axol.tracker.config import TrackerConfig, load_tracker_config

        config = load_tracker_config(path)
        if config == TrackerConfig():
            raise AssertionError("tracker config example loaded as the empty default")
        unknown = set(data) - set(TrackerConfig.__dataclass_fields__)
        if unknown:
            raise AssertionError(f"unknown tracker config keys {sorted(unknown)}")
        return
    if isinstance(data, dict) and "ssid" in data:
        # ~/.almond/tracker/ultimate_wifi.json
        from almond_axol.tracker.ultimate import ultimate_wifi_values_error

        error = ultimate_wifi_values_error(data)
        if error is not None:
            raise AssertionError(error)
        return
    if isinstance(data, dict) and set(data) == {"left", "right"}:
        # ~/.almond/mantis/tcp_transform.json
        from almond_axol.mantis.calibration import load_tcp_transforms

        tracker_config = scratch / "tracker-config.json"
        tracker_config.write_text(
            json.dumps(
                {
                    "backend": "survive",
                    "ultimate_quat_order": "wxyz",
                    "ultimate_up_axis": "z",
                }
            ),
            encoding="utf-8",
        )
        errors: list[str] = []
        statuses: dict[tuple[str, str], str] = {}
        transforms = load_tcp_transforms(
            path,
            tracker_config_path=tracker_config,
            entry_statuses=statuses,
            document_errors=errors,
        )
        if errors:
            raise AssertionError(f"tcp_transform example rejected: {errors}")
        for side in ("left", "right"):
            documented = set(data[side])
            loaded = set(transforms.get(side, {}))
            if not documented <= loaded:
                missing = sorted(documented - loaded)
                raise AssertionError(
                    f"{side} entries {missing} not loaded (statuses: {statuses})"
                )
        return
    # Anything else (e.g. a vercel.json) only needs to be valid JSON.


class DocsJsonExamplesTest(unittest.TestCase):
    def test_docs_contain_json_examples(self) -> None:
        self.assertGreaterEqual(len(JSON_BLOCKS), 5, "fence extraction regressed")

    def test_json_examples_load_through_their_readers(self) -> None:
        scratch = Path(tempfile.mkdtemp(prefix="axol-docs-json-"))
        for block in JSON_BLOCKS:
            with self.subTest(block=block.where):
                _load_json_example(block, scratch)


# ---------------------------------------------------------------------------
# API tables and the module tree
# ---------------------------------------------------------------------------

_HEADING = re.compile(r"^(?P<level>#{2,4})\s+(?P<text>.*)$")
_BACKTICKED = re.compile(r"`([^`]+)`")
_IDENT = re.compile(r"[A-Za-z_]\w*")
_TABLE_KINDS = {"Method", "Member", "Attribute", "Field", "Value"}


def _page_module(path: Path) -> types.ModuleType | None:
    """The module an API page documents (its front-matter title)."""
    match = re.search(
        r'^title:\s*"(almond_axol[\w.]*)"', path.read_text(encoding="utf-8"), re.M
    )
    if match is None:
        return None
    try:
        return importlib.import_module(match.group(1))
    except ImportError:
        return None


def _class_named(text: str, namespace: dict[str, Any]) -> type | None:
    """The single backticked class in ``text`` resolvable from ``namespace``."""
    found = []
    for token in _BACKTICKED.findall(text):
        ident = _IDENT.match(token)
        if ident and ident.group(0) == token and isinstance(namespace.get(token), type):
            found.append(namespace[token])
    return found[0] if len(found) == 1 else None


@dataclass(frozen=True)
class _TableEntry:
    where: str
    owner: type
    kind: str
    cell: str


def _api_table_entries() -> list[_TableEntry]:
    entries: list[_TableEntry] = []
    for path in DOC_FILES:
        if not path.match("docs/api/*.mdx"):
            continue
        page = path.relative_to(REPO).as_posix()
        namespace = dict(_page_imports(page))
        module = _page_module(path)
        if module is not None:
            namespace = {**vars(module), **namespace}
        context: dict[int, type | None] = {}
        lines = path.read_text(encoding="utf-8").splitlines()
        in_fence = False
        kind: str | None = None
        owner: type | None = None
        for index, line in enumerate(lines):
            if line.startswith("```"):
                in_fence = not in_fence
                kind = None
                continue
            if in_fence:
                continue
            heading = _HEADING.match(line)
            if heading:
                level = len(heading.group("level"))
                context[level] = _class_named(heading.group("text"), namespace)
                for deeper in [k for k in context if k > level]:
                    del context[deeper]
                kind = None
                continue
            if line.startswith("|"):
                cells = [c.strip() for c in line.strip().strip("|").split("|")]
                if kind is None:
                    if cells and cells[0] in _TABLE_KINDS:
                        kind = cells[0]
                        owner = None
                        # The nearest paragraph naming one class wins over the
                        # heading ("`AxolConfig` also exposes ...:" tables).
                        for back in range(index - 1, max(index - 3, -1), -1):
                            if lines[back].strip():
                                owner = _class_named(lines[back], namespace)
                                break
                        if owner is None:
                            for level in sorted(context, reverse=True):
                                if context[level] is not None:
                                    owner = context[level]
                                    break
                    continue
                if set(cells[0]) <= set("-: ") or owner is None:
                    continue
                entries.append(
                    _TableEntry(f"{page}:{index + 1}", owner, kind, cells[0])
                )
            else:
                kind = None
    return entries


def _documented_members(cell: str) -> list[tuple[str, str | None]]:
    """``"`a(x, y=1)` / `b`"`` → ``[("a", "x, y=1"), ("b", None)]``."""
    members = []
    for token in _BACKTICKED.findall(cell):
        token = re.sub(r"^(motor|self|axol|solver)\.", "", token)
        ident = _IDENT.match(token)
        if ident is None:
            continue
        rest = token[ident.end() :]
        params = None
        if rest.startswith("(") and rest.endswith(")"):
            params = rest[1:-1]
        members.append((ident.group(0), params))
    return members


def _check_table_entry(entry: _TableEntry) -> list[str]:
    problems = []
    owner = entry.owner
    for name, params in _documented_members(entry.cell):
        if entry.kind == "Value":
            members = getattr(owner, "__members__", {})
            if name not in members:
                problems.append(f"{owner.__name__} has no member {name!r}")
            continue
        if entry.kind == "Field":
            if dataclasses.is_dataclass(owner):
                names = {f.name for f in dataclasses.fields(owner)}
            elif isinstance(getattr(owner, "model_fields", None), dict):
                names = set(owner.model_fields)
            else:
                names = set(dir(owner)) | _instance_attributes(owner)
            if name not in names:
                problems.append(f"{owner.__name__} has no field {name!r}")
            continue
        if not _has_member(owner, name):
            problems.append(f"{owner.__name__} has no attribute {name!r}")
            continue
        if params is None:
            continue
        member = inspect.getattr_static(owner, name, None)
        if isinstance(member, property) or not callable(getattr(owner, name, None)):
            problems.append(
                f"{owner.__name__}.{name} is documented as a call but is not callable"
            )
            continue
        try:
            signature = inspect.signature(getattr(owner, name))
            call = ast.parse(f"f({params})", mode="eval").body
        except (TypeError, ValueError, SyntaxError):
            continue
        assert isinstance(call, ast.Call)
        accepted = {
            p.name
            for p in signature.parameters.values()
            if p.name not in ("self", "cls")
        }
        if any(
            p.kind == inspect.Parameter.VAR_KEYWORD
            for p in signature.parameters.values()
        ):
            continue
        documented = [a.id for a in call.args if isinstance(a, ast.Name)] + [
            kw.arg for kw in call.keywords if kw.arg
        ]
        for param in documented:
            if param not in accepted:
                problems.append(
                    f"{owner.__name__}.{name}: documented parameter {param!r} "
                    f"is not one of {sorted(accepted)}"
                )
    return problems


class DocsApiTablesTest(unittest.TestCase):
    """``| Method |`` / ``| Field |`` / ``| Value |`` tables name real API."""

    def test_tables_were_found(self) -> None:
        self.assertGreaterEqual(
            len(_api_table_entries()), 100, "table extraction regressed"
        )

    def test_documented_members_exist(self) -> None:
        for entry in _api_table_entries():
            with self.subTest(row=entry.where, cell=entry.cell):
                problems = _check_table_entry(entry)
                self.assertFalse(problems, f"{entry.where}: {'; '.join(problems)}")

    def test_concepts_module_tree_matches_the_package(self) -> None:
        page = REPO / "docs/api/concepts.mdx"
        block = next(
            b
            for b in ALL_BLOCKS
            if b.path == page and b.code.startswith("almond_axol/")
        )
        stack: list[tuple[int, Path]] = [(-1, REPO / "almond_axol")]
        checked = 0
        for line in block.code.splitlines()[1:]:
            match = re.match(r"^(?P<indent>[\s│]*)[├└]── (?P<name>\w+)/", line)
            if not match:
                continue
            depth = len(match.group("indent"))
            while stack and stack[-1][0] >= depth:
                stack.pop()
            path = stack[-1][1] / match.group("name")
            with self.subTest(entry=match.group("name")):
                self.assertTrue(
                    path.is_dir(), f"{path.relative_to(REPO)} is not a package"
                )
            stack.append((depth, path))
            checked += 1
        self.assertGreaterEqual(checked, 10)


if __name__ == "__main__":
    unittest.main()
