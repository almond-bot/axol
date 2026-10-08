# Known issues: symptom → cause → fix

Everything here happened on a real robot while tuning (jelly at Almond, and
a customer's gripperless robot). Check this list before improvising.

## Motion and control

**A shoulder swings / oscillates violently during calibration or homing,
one arm only.**
Cause: the MyActuator 0xA4 firmware position loop at stock gains, with the
rest of the arm held stiff on firmware loops. Reproduced on a customer's left
shoulder_1: the same 0xA4 move with the other joints limp was clean; with them
held, ~2 Hz growing oscillation. Not the encoder (read clean at ~1.2 kHz over
the full range). Fix: the calibration tools are impedance-only since commit
`676c115`; update the checkout. If a recording shows `a4_cmd` frames, find what
enabled 0xA4 (ground rule 1) and remove it.

**Ctrl-C during a sweep makes a joint jump several degrees and ring.**
Cause (fixed in `49e126d`): homing held the joint at its pre-sweep target.
Update the checkout.

**After each joint's sweep the shoulder lurches back fast; the elbow aims at
its hard stop.**
Cause (fixed in `a3483f3`): a fixed 4 s linear ramp to 0 before homing.
Update the checkout.

**Joints sag for a moment when a tuning tool starts.**
Expected: the motors reset (~2.5 s, no torque, no brakes) before the holds
engage. Start with the arms hanging at rest; an arm raised at start-up will
drop.

**`no status reply — cannot tell whether it is holding` at start.**
A motor isn't answering: arm power off, e-stop pressed, CAN cable. Not a
tuning problem.

**`neither it nor a ±360° single-turn wrap … the zero has not been set`.**
A joint's reading is impossible for its zero. Either a glitch read during a
motor reboot (seen once, right after a violent 0xA4 move), or a lost zero.
Power-cycle; if it persists, `axol motor.set-zero-pos --guided` (Almond-guided).

**Slow motion feels sticky / ripply on shoulder_3 or the wrists.**
By design the runtime cancels little low-speed friction there: Stribeck gain
0 (it added wrist sway when tested in 2026-10), and the Coulomb term is smoothed
through zero (17% of `fc` at 1 °/s). Lever: `stribeck_gain` A/B (Phase 4).

## Calibration

**`! Gravity fit rejected: fitted CoM shift N mm exceeds the 60 mm cap`.**
Benign. The data is fine (measured gravity matches the model to within ~5%
scale); the fit can only move one link's CoM with its mass fixed, and light
links (elbow / wrist_1, 0.25 kg) can't absorb a distributed load error. The
model plus `Fo` stays within ~0.3 Nm. Jelly rejected 7 of 14 joints.

**A customer tuned gravity by hand (link `mass` / `com` in the panel).**
Panel settings override the calibration file. Run `tune.factory
--keep-gravity`: fits run against their gravity, and only friction, Stribeck and
`Fo` are saved. Without it, `Fo` would be fitted against a model the robot
never runs (their heavier wrist_3 alone shifts proximal gravity ~1 Nm).

**Calibration has no effect.**
The file is scoped to the Axol hub's serial; a file from another robot (or
no hub attached) is ignored. `robot_report.py` →
`calibration["calibration.json"].matches_this_hub`.

**Settings keys under `axol.experiments.*`** (e.g. `friction_load_gain`,
`integrator_hz`) no longer exist; they are dropped silently. Harmless.

## Environment

**`cannot enter SCHED_FIFO … refusing to arm without real-time scheduling`.**
Rebuilding the core strips its capabilities. `sudo setcap
cap_sys_nice,cap_ipc_lock=ep rust/axol-rt/target/release/axol-rt` (or `axol
rt.install`). `tune.factory` doesn't need it; `tune.motion` and teleop do.

**`wrist IMU: … camera did not open in time — no IMU metrics`.**
The wrist cameras didn't come up (seen on both after a reboot). Runs still
score joint ripple, but there is no tool-sway metric — don't decide IMU
questions without it. Check the camera stack before relying on IMU.

**`Automatic CAN discovery failed: … requires root` from `axol serve`.**
CAN discovery needs root; run via the installed service, or `axol can.setup`.

**A tool can't open the bus / waits forever.**
Something else owns it: `axol serve` (systemd `axol.service`), teleop, a
tuner. `robot_report.py` → `system.processes_on_the_robot`.

**Updating the checkout.**
`git pull --ff-only` in `~/axol`. Rebuild the core only if `rust/` changed
(`git diff --stat OLD HEAD -- rust`), then re-grant real-time (above).
Dependencies only change with `pyproject.toml` / `uv.lock`; on a Jetson
never run a bare `uv sync` (AGENTS.md has the safe command).

## Measurement

**Results flip between runs.**
Creep ripple drifts ±30% between sessions, and a joint can switch between
a rough and a smooth state from one run to the next (right shoulder_3: ~32 vs
~20 mdeg). Only interleaved rounds in one session count; ≥ 3 rounds.

**A video shows "oscillation" but the numbers say quiet / command jitter.**
Handheld 30 fps video can't resolve a few millimetres, and the arm follows its
input exactly — a jittery tracker looks like a shaking arm. Trust
`can_trace.py summary`.
