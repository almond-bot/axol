# AGENTS.md

## Cursor Cloud specific instructions

### Overview

Almond Axol is a Python CLI + SDK for the Almond Axol dual-arm robot. Since no physical robot hardware is available in the cloud VM, all development and testing uses the **sim** mode (`--sim`), which renders the robot in a browser via viser.

### Writing docs and UI text

These rules cover everything a customer reads: `docs/`, the web UI under `web/`, settings help in `almond_axol/serve/settings.py`, command and diagnostic descriptions, and CLI help and log lines meant for operators. Code comments, docstrings and commit messages are not covered. Write for an operator standing next to the robot, not for the engineer who wrote the code. Read your text back before you commit it, out loud if that helps. If a sentence only makes sense to someone who saw the diff, rewrite it.

**Names.** **Axol** is the dual-arm robot. The mounts are **Standalone**, **Owl Mount**, **Ox Cart** and **Jelly Mobile** (short form **Jelly**). Jelly is a powered mobile base, not a robot: write "Axol on Jelly", never "the Jelly robot". **Mantis** is the pair of handheld data-collection rigs. The **Camera Kit** and **Compute Kit** are add-ons. Never name our own test units ("the jelly robot", "our in-house Axol"). Call a control by the label it has in the UI, in bold, and don't invent names for features.

**Describe the product as it is now.** Leave out history and tuning notes: no dates, no "used to", "older", "now", "no longer" or "new", no A/B results ("30% rougher", "−57% ripple", "over three interleaved rounds"), and no record of how a default was chosen. That belongs in the commit message. Give a number only when the reader needs it: a time to plan for, a limit, a value to type.

**Say what the reader does and what happens.** Leave out implementation detail and design rationale unless it changes what the reader does. Don't explain security boundaries or internal plumbing ("the web API deliberately accepts neither path", "the observer pairs each command with its feedback"). Put internals on an Advanced or API page, or leave them out.

**Short, plain sentences.**
- One idea per sentence and at most four sentences per paragraph. When a paragraph lists several behaviors, flags or cases, make it a list or a table.
- No em dashes. Use a period, a comma, a colon or parentheses. Use at most one parenthetical per sentence, and never nest them. Don't chain clauses with semicolons.
- Every sentence must make sense on its own. Never point at something that isn't there ("the numbers above", "the older quick profile") or at a version the reader never saw.
- Prefer common words. Avoid filler and coinages such as "deliberately", "genuinely", "exactly", "simply", "just", "at once", "its own" ("the core's own law"), "plumbing", "ad hoc", "heals itself", "that beats", "nothing to enable", "lights up". Don't use arrows (→, ↔) in prose.
- Lead with the action. Write "Run `axol tune.factory`", not "Calibration is performed by running ...".

**Page front matter.** `title` is a short noun phrase ("Calibrating Axol"). `description` is one sentence of 20 words or fewer that says what the page helps you do.

**UI text.**
- Buttons are a verb plus an object ("Start recording"). Labels are one to three words.
- A tooltip or hint is one short sentence. It says what the control does, not how it is built.
- An error says what went wrong and what to do next ("Left arm not found on can_alm_axol_l. Check the USB cable, then press Retry.").
- Capitalize the first word only, and keep the same word for the same thing across the page and the docs.
- When you change visible text, update any test that matches on it.

### Running the application

- **Sim teleop** (the primary way to exercise the app without hardware): `uv run axol teleop --sim`
 - Opens a viser 3D viewer at `http://localhost:8002` and a VR WebSocket server on port 8000.
 - With no VR headset connected the arms just hold the rest pose. To actually drive them, either use the `Sim` SDK directly (`sim.motion_control(left=..., right=...)`, see the `Sim` docstring in `almond_axol/robot/sim.py`) or connect a WebSocket client to `wss://localhost:8000/ws` (self-signed cert — disable TLS verification) and stream `VRFrame` JSON with both `l_lock`/`r_lock` true to engage tracking.
 - The viser server persists engage/IK state across teleop restarts only within one process; if a WebSocket client leaves tracking engaged and reconnects, restart the `teleop` process for a clean engage.

### Web front-end (second product)

The browser UIs live under `web/` (a Vite + React monorepo: the WebXR `/vr` teleop app and the `/control` panel served by `axol serve`). Node 22 is available; `web/` is **not** covered by `uv sync`. Standard install/build/dev commands are in `web/README.md` (build the `packages/axol-vr-client` workspace before `app`).

### Linting

- `ruff check .` and `ruff format --check .` — ruff is not a project dependency; it's pinned in `.pre-commit-config.yaml` (see the `rev:` field). Easiest: `uv tool install pre-commit && pre-commit run --all-files`, which uses the pinned version automatically. Or install ruff directly at the same version, e.g. `uv tool install ruff@0.9.7`.

### Testing

- Python unit tests live in `tests/` (mostly stdlib `unittest`, with some pytest-style modules): run the whole suite with `uv run pytest`, which also enforces the 30% branch-aware coverage floor from `pyproject.toml`. `uv run python -m unittest discover -s tests` still works for the `unittest` modules. They run without hardware — hardware-facing modules are exercised through mocks — but several import the `lerobot` extra at module level, so run them from `uv sync --extra sim --extra lerobot --dev` (a bare `--extra sim` sync leaves ~9 modules failing to import). Add a `tests/test_<module>.py` alongside behaviour changes. For anything the suite doesn't cover, validate by importing the package and exercising the `Sim`-based code paths.
- Run the web unit/component suite from `web/` with `npm test`; use `npm run lint`, `npm run format:check`, and `npm run build` for the remaining front-end gates. GitHub Actions (`.github/workflows/ci.yml`) runs the Python and Web gates on every PR.
- MuJoCo is pinned exactly (`mujoco==3.11.0` in `pyproject.toml`), along with `mink`, `qpsolvers` and `daqp`: the Mink IK backend's numerical reference (`tests/test_mink_ik_parity.py`) is exact, and MuJoCo changes collision rows between releases. `almond_axol/robot/gravity.py` handles both the pre-3.10 and 3.10+ `mj_fullM` APIs (`tests/test_gravity.py` covers both). `uv.lock` alone does not protect `uv tool install` / `pip install` (they re-resolve), so the pin is deliberate; move MuJoCo, the Mink stack, the reference fixture and `uv lock` together.
- Two IK backends ship in the core dependencies: `jax` (Pyroki, the default) and `mink`. One switch selects it for every operation: the control panel's **Kinematics → IK solver** setting (`kinematics.backend` in `~/.almond/settings.json`), which fans out to `KinematicsConfig.backend` (teleop / collect-data's teleop) and `AxolRobotConfig.cartesian_controller` (LeRobot ops). Collection sessions refuse a mismatch between the two. See `docs/operations/mink.mdx`.
- The Rust realtime core (`rust/axol-rt`, the required hardware control backend) has a `cargo test` suite: golden filter vectors pinned to the Python originals, wire-protocol round trips, and a damping dissipated-power comparison. `uv run python rust/axol-rt/tools/rt_proto_check.py` exercises the built binary's Unix-socket protocol without CAN. `cargo test stall_detection_live -- --ignored` needs the CAN interfaces up with motors unpowered.

- The arm hub firmware (RP2350) is developed in the circuits-py repo (`designs/axol_hub/firmware`); only the built image ships here, as `almond_axol/hub/axol_hub.uf2`, and every bring-up reflashes a hub reporting an older `bcdDevice` (`almond_axol/hub/firmware.py`). To take a new firmware, copy circuits-py's built `firmware.uf2` over `axol_hub.uf2` and set `FIRMWARE_VERSION` to the `AXOL_HUB_FW_VERSION` it was built with; `tests/test_hub_firmware.py` checks the image reports that version.

### Dependency extras

| Extra | Purpose |
|-------|---------|
| `sim` | viser (browser 3D visualizer) — needed for sim mode |
| `lerobot` | LeRobot data collection/policy — requires hardware + ZED cameras. Not needed for teleop camera streaming: the ZED SDK cameras live in `almond_axol/video/zed_sdk.py` and `almond_axol/lerobot/camera` only wraps them in LeRobot's `Camera`/`CameraConfig` |

For cloud development: `uv sync --extra sim --extra lerobot` (the `lerobot` extra is import-time required by part of the test suite; `--extra sim` alone is enough to run sim teleop).

**On a real robot (Jetson/tegra host), never run a bare `uv sync --extra sim`.** The robot's venv also carries the `lerobot` extra plus out-of-band installs — `pyzed` (from `~/.almond/wheels/`) and PyGObject (`pygobject>=3.50,<3.52`, built against the system gobject-introspection) — and an exact sync silently removes them, which kills camera streaming (no `pyzed` for the SDK fallback — teleop logs a `zed.install` hint — and no `gi` for the gst relay) and data collection (`No module named 'lerobot'`). Restore with `uv sync --extra sim --extra lerobot` then `uv pip install ~/.almond/wheels/pyzed-*.whl "pygobject>=3.50,<3.52"`. Do **not** install the self-built `jaxlib` / `jax_cuda12_*` wheels from `~/.almond/wheels/` — they were compiled against cuDNN 9.8 while JetPack ships 9.3, so the IK worker's first solve crashes (`RET_CHECK failure ... dnn_support != nullptr`); the lock's CPU jaxlib runs IK at full teleop rate.

### Gotchas

- Python 3.13+ is required (`.python-version` pins `3.13`). The VM ships with 3.12; use `uv python install 3.13` if needed.
- The `uv` package manager must be on PATH (`$HOME/.local/bin`).
- Hardware-dependent commands (`can.setup`, `motor.*`, `gravity-comp`, `tune.*`, `zed.*`, `collect-data`, `collect-dagger`, `run-policy`) will fail without physical robot/CAN bus — this is expected.

### Tuning a real robot (guiding an operator)

When asked to diagnose or tune a robot's arms (oscillation, shake, calibration, "one arm still oscillates"), follow the playbook skill `axol-robot-tuning` — `.claude/skills/axol-robot-tuning/SKILL.md` and its `reference/` files. Claude Code loads it from `.claude/skills/`, Codex from `.agents/skills/` (a symlink to the same folder — edit the `.claude` copy); any other agent should read it before acting. It sets the safety rules (no firmware position loops, guards on, the operator runs anything that moves), and the tools it drives: `scripts/robot_report.py` (read-only intake), `scripts/can_trace.py record|summary` (is the command or the controller oscillating), `scripts/creep_motions.py` (test motions), `scripts/tuning_queue.py` (guarded, interleaved A/B runs with a verdict) and `scripts/ext_imu.py` (an IMU the operator mounts on their own end-effector when the robot has no wrist cameras).
