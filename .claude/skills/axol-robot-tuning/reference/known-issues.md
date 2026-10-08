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

**Homing "goes wild", or a shoulder drives itself into its end stop**
(stall error 0x0002). A motor's firmware planner acceleration was written to
0 (0/10 dps/s on the X8) — an old tool did this. Only matters on the 0xA4
loop, which nothing uses by default now. Hand off to Almond: they read the
motor's stored values and restore stock (5000). Don't run the firmware tools
yourself (ground rule 1).

**The arm moved by itself right after a CAN reconnect / `can.setup`.**
The USB CAN hub replays up to 10 stale frames per channel after an interface
flap, and a MyActuator acts on a 0xA4 frame without being enabled. `can.setup`
USB-resets the hub now; after any "TX queue stalled", unplug and replug it.

**A run behaves like the previous run's gains.**
Joints still holding after a cut run keep their old settings. Power-cycle the
arm after an aborted run.

**`contact: … torque residual N Nm exceeded 8.0`** — `tune.motion`'s contact
watchdog: something touched the arm or the gravity model is far off. Check
the space, then the calibration.

**A calibration run hit something / the arm went where it shouldn't.**
The calibration tools have no tracking guard (only `tune.motion` does). Full
range passes go to the joint limits (jelly's left shoulder_1 swept −87…177°).
Clear the space, stay at the e-stop. Ctrl-C **once**: a second Ctrl-C aborts
the return home and leaves the arm energized and holding.

**Buzz at ~110 Hz on wrist_2:** never raise its kd (5 buzzed; 2.25 ships).
`kd` is encoded on 0–5; larger values are silently clamped.

**A ~9–13 Hz shudder in the shoulders / mast.** The stand has a mode there.
shoulder_3's host damper is 0 because damping pumped it; a host-damper Q
below 1 once excited it on shoulder_1. If a robot shudders with the shipped
Q 1 damper, A/B `kd_host_q` / `kd_host` on that robot.

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

**Fix the wrist cameras:** `sudo systemctl restart zed_x_daemon`, then check
an IMU opens before a session (a run prints the `wrist IMU (…)` line). Only one
process can own a ZED camera — teleop or the panel holding it blocks a run.

**Teleop over SSH refuses to start (SCHED_FIFO):** `sudo prlimit --pid $$
--rtprio=20:20` in that shell first.

**A flag "doesn't exist":** the system-wide `axol` (an older install) runs
instead of the checkout's. Use `~/axol/.venv/bin/axol` or `uv run axol`.
`axol serve` / the panel also run the installed version until it's updated.

**`tune.motion` keeps no log:** pipe it: `… 2>&1 | tee ~/run.log`.

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

**The scorecard RMS looks huge (1–1.5°).** That is mostly tracking lag
(55–65 ms), not wobble. Judge wobble on the wrist IMU and on `lagfree`.

**The person says it feels "notchy" but the numbers improved.** Their hand is
a sensor too: stick-slip they feel is real. Trust a recording plus their
report together, and ask what exactly they feel and where.

**A video shows "oscillation" but the numbers say quiet / command jitter.**
Handheld 30 fps video can't resolve a few millimetres, and the arm follows its
input exactly — a jittery tracker looks like a shaking arm. Trust
`can_trace.py summary`.
