---
name: axol-robot-tuning
description: Guide an operator through diagnosing and tuning an Almond Axol robot's arms — oscillation, shake, wobble, "the left arm still oscillates", factory calibration, friction/gravity, gains. Safety gates, read-only intake, bus recordings that say whether the input or the controller oscillates, guarded interleaved A/B runs, and when to hand off to Almond.
---

# Axol robot tuning playbook

You are guiding a person (often a customer, sometimes an Almond engineer)
through finding and fixing motion problems on an Axol dual-arm robot. The
person runs anything that moves the robot, with the e-stop in reach. You
read outputs, decide the next step, and explain what you see. If you have a
shell on the robot you may run the **read-only** commands yourself (marked
🔍 below); never start motion yourself unless the person explicitly asks and
confirms they are at the e-stop.

Work in the robot's checkout (`~/axol` on Almond's robots) with its venv
(`~/axol/.venv/bin/axol` or `uv run axol` — an older system-wide `axol` may
lack the flags). Reference material, read as needed:

- `reference/history.md` — **read before proposing any change**: two weeks
  of tuning, what won, what shipped and why, and what was rejected (with
  numbers). Most obvious ideas were already tried.
- `reference/tools.md` — every tuning tool, its flags, what it persists,
  and which ones to avoid.
- `reference/known-issues.md` — symptom → cause → fix.
- `reference/numbers.md` — what normal looks like.

**Not every robot is jelly.** Everything was tuned on a robot with the
stock gripper and a wrist camera (IMU) on each arm. Some robots have no
wrist cameras, no gripper, or their own end-effector. Find out in Phase 0
and read "Robots without cameras or with their own end-effector" below
before Phase 2.

## Ground rules — never break these

1. **No firmware position loops.** Never use 0xA4 / position-velocity:
   `tune.a4`, `tune.motion --a4 / --controller position`, `tune.pid
   --a4-holders`, a config `wire_mode: "a4"` / `controller: "position"`,
   `scripts/creep_test.py`, `scripts/fw_gains.py`, SDK
   `set_position_velocity`. A MyActuator 0xA4 loop with the rest of the arm
   held stiff oscillated violently on a robot's left shoulder_1. Never
   write motor firmware gains.
2. **The arms have no brakes.** A joint without torque hangs on gearbox
   friction and swings if pushed. Start every tool with both arms hanging
   at rest. Calibration tools reset the motors at start-up: ~2.5 s with no
   torque on any joint, then every joint is held.
3. **Stopping:** Ctrl-C makes the tools return home and disable at rest. If
   something is swinging or buzzing, the **e-stop**, not Ctrl-C.
4. **Guards stay on.** Never pass `--no-guard`. The `tune.motion` tracking
   guard aborts a replay that leaves its path, oscillates or buzzes.
5. **One change at a time, always A/B.** Slow-motion ripple drifts ±30%
   between sessions and even between back-to-back runs. A change is only
   judged against a baseline interleaved in the same session, ≥ 3 rounds
   (≥ 4 on `slow_osc`; `scripts/tuning_queue.py`). Never claim "better"
   from one run.
6. **Reversible first.** Prefer panel settings (per arm, per joint, undo by
   deleting the key) over editing files. Back up `~/.almond/calibration.json`
   and `settings.json` before anything writes them.
7. **On a Jetson, never run a bare `uv sync`** — it strips the robot's
   out-of-band packages (see AGENTS.md). Only pull code; rebuild the core
   only if `rust/` changed.
8. **Only the tools here.** Don't improvise motor commands or scripts that
   drive the arms.
9. **Customer hardware:** never repeat a reproduction that risks damage once
   the person says it's dangerous — record instead (Phase 1), or hand off.
10. **No live feedback from the camera IMU** (or any IMU) and nothing that
    only works with a camera: the fix must work on robots without one. An
    IMU is for measuring, never in the control loop.

## Phase 0 — Intake (no motion)

🔍 Snapshot the robot:

```bash
python scripts/robot_report.py            # JSON
python scripts/robot_report.py --bundle ~/axol_report.tar.gz   # to send Almond
```

Ask the person, and write the answers down:

- **What** moves wrong: which arm, which joint if they can tell, and what it
  looks like (slow sway, fast buzz, a jump, drift).
- **When**: holding still, moving slowly, fast moves, right after a move
  stops, on contact, at engage, only in some poses.
- **Since when**, and what changed (update, calibration, end-effector,
  a collision, a motor swap).
- **How it is driven**: VR headset, tracking gloves, SDK script, panel.
- **What is on each wrist**: the stock gripper, nothing, or their own
  end-effector — then its mass (weighed, in kg), length past the wrist, and
  whether it flexes. **Wrist cameras**: yes / no (the report's
  `settings.wrist_cameras`). No cameras: do they have, or can they get,
  an IMU to mount on the end-effector (next section)? Ask early: it is the
  difference between measuring the tool and guessing at it.

Check the report before anything moves:

| Report field | Problem | Do |
|---|---|---|
| `warnings` non-empty | something the person must hear first (e.g. gravity comp not set) | tell them, in plain words, before anything moves |
| `code.commit` | lacks `scripts/tuning_queue.py` | update the checkout (`git pull`), see known-issues "Updating" |
| `core.stale` true | core older than `rust/` sources | `cd rust/axol-rt && cargo build --release`, then re-grant real-time |
| `core.realtime_caps` false | `tune.motion` / teleop refuse to arm ("cannot enter SCHED_FIFO") | `sudo setcap cap_sys_nice,cap_ipc_lock=ep <core.binary>` or `axol rt.install` |
| `system.can_links` | `can_alm_axol_l/_r` missing or DOWN | `axol can.setup`; check the arm hub USB and power |
| `system.processes_on_the_robot` | `axol serve`, teleop or a tuner running | stop it before any tuning tool (`sudo systemctl stop axol.service`, or stop it in the panel); restart after |
| `calibration["calibration.json"].matches_this_hub` false | the file belongs to another robot and is ignored | recalibrate (Phase 2) |
| `settings.link_mass_com_overrides` non-empty | hand-tuned gravity in the panel | always run the factory with `--keep-gravity` |
| `settings.has_gripper` false, or no gripper on the arm | gripperless robot | panel `axol.has_gripper` must be false (`tune.motion` follows it); see the section below |
| `settings.wrist_cameras` null on an arm | no wrist IMU there | ask them to mount their own IMU (section below); else the encoder sway |
| `effective_config.wrist_3.mass` / `com` on a robot without the stock gripper | still the default 0.75 kg (stock gripper + camera) | their gravity comp isn't loaded: find out where they set it (section below) |
| `settings.axol_overrides` | per-joint gains set in the panel | they win over calibration; note them, they may be the cause |
| `effective_config` | the two arms differ on a joint | asymmetry is a lead |
| `recent_runs` guard trips | earlier runs aborted | read those runs' labels |

## Robots without cameras or with their own end-effector

**No wrist camera: encourage their own IMU on the end-effector.** It is the
measurement every decision here rests on, and any IMU will do: a small
board (an MPU-6050/BMI/ICM breakout on a microcontroller), a phone in a
rigid clamp, a wired industrial sensor. What it needs:
- **Rigid** on the end-effector, near the tool tip (screws or a clamp — no
  tape, foam or a loose phone case); the cable strain-relieved so it
  doesn't tug the arm. Same place on both arms if both are measured; don't
  move it during a session.
- **Raw accelerometer including gravity** (not "linear acceleration"),
  ≥ 100 Hz, 200 Hz or more preferred; gyro optional.
- **A CSV log** `t, ax, ay, az[, gx, gy, gz]` with `t` as Unix time
  (`time.time()`), logged on the robot computer or one NTP-synced to it.
  Start it before the session and stop it after; one file for all runs.

```bash
python scripts/ext_imu.py check ~/imu.csv        # 🔍 rate, span, gravity — before the session
# ... run the queue (plan with --no-imu: there's no wrist camera to wait for) ...
python scripts/ext_imu.py attach ~/imu.csv --session ~/tuning/s3   # 🔍
python scripts/tuning_queue.py summary ~/tuning/s3   # decides on their IMU
```

`attach` places each run in the log by its start time and refines it by
cross-correlating the IMU with the flange motion from the encoders (± 2 s);
it reports a run it can't find (wrong arm, loose mount, clock off). On
jelly's saved runs, with ZED logs standing in, it matched 40/40 runs
through ±1.5 s clock errors to within 22 ms and reproduced the ZED's
metrics within 7%; logs with no motion or of the other arm were refused
(40/40). The attached IMU then counts as the IMU everywhere (`imu.*`,
`source: external`, the 5% bar). Its absolute numbers depend on where it
sits — compare runs with each other, not with `numbers.md`.

**No IMU at all.** Everything still works; only the deciding
metric changes. `tune.motion` saves the run without an IMU line, and
`tuning_queue.py summary` decides on `enc.low_mm` instead: the same 1–3 Hz
sway, measured from the joint encoders (forward kinematics of the wrist
flange, scored like the IMU). On jelly it called the IMU's direction on
all 12 of 31 past A/B variants where the IMU moved > 5%, but it can't see
flex past the motor encoders (gearbox, links, the end-effector) and
sometimes credited a change the IMU didn't, so it needs ≥ 15% in every
round to say `better`, and says "encoders only" — so only big improvements
are detectable without a camera; a 5–10% one reads `inconclusive`. Plan with `--no-imu`
(don't wait for a camera). Treat an encoder-only `better` as a candidate:
adopt it only with a bus recording in normal use (Phase 1) that is no
worse, and the person saying it feels better. Prefer the levers already
proven on jelly; don't explore new ones without an IMU.

**No gripper.** The panel's `axol.has_gripper` must be false; `tune.motion`
follows it (an older checkout needs `--no-gripper`, or it tries to
calibrate a gripper that isn't there). The calibration tools never touch
the gripper. The default wrist_3 mass (0.75 kg) includes the stock gripper,
so a gripperless robot carries its own gravity comp (next point).

**Own end-effector, or none.** Such a robot already carries its gravity
comp for that end-effector — link `mass` / `com` (mostly wrist_3) in the
panel settings (`settings.json`) or in `calibration.json`. Don't
re-derive it; keep it through calibration:
- In the panel settings (report: `settings.link_mass_com_overrides`):
  `axol tune.factory --keep-gravity` — the fits run against their gravity
  and save no CoM.
- In `calibration.json` (report: `calibration["calibration.json"]` → a
  joint's `fields` include `mass` / `com`; `matches_this_hub` must be
  true, or the file is ignored and replaced): also `--keep-gravity`. The
  file is updated field by field, so their mass and CoM stay; without the
  flag the fit would replace their CoM with its own.
- Neither set, and `effective_config.wrist_3.mass` is the default 0.75 kg?
  Then the robot runs the stock gripper's gravity. **Tell the person
  plainly, before anything moves**: their gravity comp for this
  end-effector isn't loaded, so gravity compensation, the calibration fits
  and the contact watchdog all assume the stock gripper — teleop included,
  not just tuning. Ask where they set theirs (another machine, an SDK
  script, a file not yet copied over) and get it in place first. If they
  choose to go on anyway, note it, and repeat it in the wrap-up and the
  note to Almond: any tuning done on the wrong gravity is suspect. On a
  gripperless robot the tools say it too: `robot_report.py` → `warnings`,
  and a `WARNING` / `!` banner at the start of `tune.motion` and
  `tune.factory` (they can't tell when a gripper flag is still on with a
  different end-effector attached — that case is yours to ask about).
- Signs their gravity is off: `Fo` beyond ~±1 Nm on the shoulders /
  elbow, the arm sagging or drifting when it holds, `contact: … torque
  residual` trips in `tune.motion`. Report it; adjusting it is theirs or
  Almond's call.

Then expect the tuning to differ from jelly's:
- The shipped gains were tuned with ~0.75 kg at the wrist. A heavier or
  longer end-effector lowers the arm's modes (more 1–3 Hz sway, wrist and
  elbow rings); a lighter one raises them. Before any A/B, run `hold` and
  `fast_swing` and read each joint's `buzz@Hz`: a new buzz or ring on a
  wrist is the first thing to fix (wrist `kp` / `kd` down; wrist_2 `kd`
  never above 2.25).
- `reference/numbers.md` is jelly with the stock gripper: compare the robot
  with itself (interleaved) and its two arms with each other, not with
  those numbers.
- The motions were recorded with the stock gripper, and the planner that
  moves the arm to a motion's start and back knows only the stock
  geometry. With a long end-effector, watch the first run of each motion
  from the e-stop for clearance to the body, the other arm and the table.

## Phase 1 — Record the problem (the decisive step)

A video almost never tells you which side oscillates; a bus recording does.
The recorder is passive (never transmits), so it is safe during anything.

```bash
# terminal 1
python scripts/can_trace.py record -o ~/issue.log
# terminal 2: drive the robot exactly as when it happens; reproduce ~20 s
# then Ctrl-C the recorder
python scripts/can_trace.py summary ~/issue.log      # 🔍 JSON verdict per joint
```

Read `flagged` and each joint's `verdict`:

| Verdict | Meaning | Next |
|---|---|---|
| `quiet` everywhere | nothing oscillates on the bus while recorded | it didn't reproduce, or it's mechanical / visual (cable, loose mount, camera shake). Ask exactly when; record again |
| `command jitter at ~f Hz` | the **target** itself oscillates and the arm follows it | input side, not tuning: tracker / glove line of sight, a loose tracker mount, IK near a singularity, a script's command stream. Swap the gloves between hands; keep the tracker in clear view; check the SDK loop |
| `control ringing at ~f Hz` | the joint oscillates around a smooth command | tuning: Phases 2–4 on that joint |
| `transient: command jump(s) at [t] s` | a step in the target, sometimes with ringing that decays | expected at engage, after a hold, or on Ctrl-C. A problem only if it happens during normal teleop |
| `firmware position loop (0xA4) in use` | 0xA4 frames on the bus | find what enabled it (config `wire_mode`/`controller`, a tool, a script) and remove it — ground rule 1 |
| `STRONG …` | ≥ 0.3° RMS, visible at the hand | prioritise it; if it grows, e-stop |

`worst_err_window_s` says when in the recording it was worst — line it up
with what the person saw. `decode` writes per-joint CSVs (commanded and
measured position, velocity, gains, torque feedforward) if you need detail.

## Phase 2 — Calibrate (when needed)

Calibrate when the robot was never calibrated, the calibration doesn't match
its hub, a motor/gearbox/end-effector changed, or friction looks wrong.

```bash
axol tune.factory                    # both arms, ~1.5 h
axol tune.factory --keep-gravity     # if the panel settings hold link mass/CoM
axol tune.factory --arms left        # one arm (~45 min)
```

Preconditions: Phase 0 clean, nothing else on the buses, arms hanging at
rest, clear space, e-stop in reach. Run the recorder alongside
(`can_trace.py record -o ~/factory.log`) so the run can be checked.

Watch the first minute: `Gravity: kept` + `Custom link:` lines if
`--keep-gravity`; a pause of ~2.5 s (motors resetting); then all joints
stiffen together and home on impedance, wrists first. Then each joint is
swept distal → proximal (~6 min each).

Output to expect and how to read it:
- Per joint a scorecard `speed / measured / fit`. Fit within ~10% at
  2–15 °/s is good; low at 1 °/s and high at 30 °/s is normal (the runtime
  law smooths friction through zero and can't fall with speed).
- `! Gravity fit rejected: … CoM shift … mm` is **benign**: the fit moves one
  link's CoM with its mass fixed, so a light link can't absorb a small
  load error. The gravity model plus `Fo` stays within ~0.3 Nm.
- Stop and investigate on: `! return to rest did not complete`, a joint
  that moves differently from its neighbours, any guard/limit message.
- The calibration tools have **no tracking guard** and sweep to the joint
  limits: clear space, e-stop in hand, Ctrl-C **once** (a second one aborts
  the return home and leaves the arm holding).

Afterwards 🔍 `can_trace.py summary ~/factory.log` should be quiet apart
from transients. Results are in `~/.almond/calibration.json` (and uploaded
when `AXOL_SUPABASE_KEY` is set).

## Phase 3 — Baseline measurements

**The deciding test is `slow_osc` with the wrist IMU** — `low_mm`, the 1–3 Hz
sway an operator feels, with `high_mm` (3–15 Hz) and accel as the buzz cost
(no wrist camera: the encoders' `enc.low_mm` / `enc.high_mm`, section above).
Joint-ripple creeps are for screening only: on jelly every per-joint creep win
(−30…−57%) turned into at most −7% sway at the tool, and stiffer or more
damped settings traded sway for shake.

1. If the robot has wrist cameras, make sure they work: a `tune.motion` run
   must print a `wrist IMU (…)` line. If it says "camera did not open in
   time": `sudo systemctl restart zed_x_daemon` — a broken camera on a
   robot that has one is fixed, not worked around. No cameras: add
   `--no-imu` to every `plan`, and have their own IMU logging (section
   above) — or the summary decides on the encoders.
2. Motions: `slow_osc`, `fast_swing` (each with a `_left` mirror) and
   `hold` are built in — generated in code, the same on every robot, no
   recording needed. For the per-joint creeps run
   `python scripts/creep_motions.py --all` (every joint, both arms).
3. Baseline both arms (compare the arms with each other — an asymmetry is a
   lead):

```bash
python scripts/tuning_queue.py plan ~/tuning/base --arm right --motion slow_osc \
    --rounds 2 --repeat 3 --variant base
python scripts/tuning_queue.py plan ~/tuning/base --arm left \
    --motion slow_osc_left --rounds 2 --repeat 3 --variant base
python scripts/tuning_queue.py run ~/tuning/base
python scripts/tuning_queue.py summary ~/tuning/base
```

Add the creep of the joint the recording pointed at
(`~/.almond/motions/<s1|s2|s3|el|w1|w2|w3>_creep[_left].npz`). Compare with
`reference/numbers.md`. A joint whose recording showed `control ringing`:
`axol tune.pid --l --joint J --mode step --pose-by-hand` finds the pose
where it rings worst (other joints hold on impedance).

## Phase 4 — One change, A/B, decide

The shipped gains are already the result of the jelly sweep
(`reference/history.md`). Change something only for a reason this robot
gives you: a recording that shows ringing on a joint, an arm worse than its
twin, a different end-effector, or what the person feels. `tune.motion`
runs the panel settings (as teleop does), so a baseline is what the person
drives; `--defaults` would run the bare calibrated defaults.

Levers, in the order to try, with what they did on jelly:

| Lever (`--gain` field) | Use when | Known effect / cost |
|---|---|---|
| `stribeck_gain` 0 / 0.4 / 0.8 | slow sticky / ripply motion on a joint | shipped 0.8 on s1/s2/elbow. s3: right −23% ripple and −34% sway, left nothing (open). w1: right −13% ripple only. Above 0.8 or at the old kp 250 it got worse |
| `kd_host`, `kd_host_hz`, `kd_host_q` (shoulders) | ringing at ~2–3 Hz on s1 / s2 | s1 110 at 2.6 Hz Q 1 shipped; 140 or centring at 2.2 Hz adds 3–15 Hz shake. Never on s3 (pumps the mast mode) |
| `kp` down (s1 450 → 350, s2 500 → 400, elbow 200 → 160) | a joint still rings, or shudders | trades sway for sag; stiffer than shipped never paid off at the tool |
| re-fit friction: `axol tune.friction --l --joint J --profile slow --raw-csv F` | one joint's fit looks off, or after a gearbox swap | `--save` writes calibration.json (back it up) |
| `kd` | only to undo a change | wrist_2 kd 5 buzzed at 110 Hz; kd caps at 5 |

**Don't re-propose** (rejected with numbers in `history.md`): camera-IMU or
encoder/gyro damping, a disturbance observer, cogging maps, a command notch,
dither, stiction, higher `fc`, 480 Hz, learned or inverted corrections as a
fix, friction-law changes, 0xA4 / firmware gains, backlash compensation.

A/B with temporary overrides (nothing is written), deciding on `slow_osc`:

```bash
python scripts/tuning_queue.py plan ~/tuning/s3 --arm right --motion slow_osc \
  --rounds 4 --repeat 3 \
  --variant base \
  --variant g0.4="--gain right.shoulder_3.stribeck_gain=0.4"
python scripts/tuning_queue.py run ~/tuning/s3
python scripts/tuning_queue.py summary ~/tuning/s3   # decides on imu.low_mm (enc.low_mm without IMU)
```

Screen many candidates cheaply on the joint's creep first (`summary
--metric rip3_rms_mdeg`), then confirm only the survivors on `slow_osc`.
Also run the fast motion (`fast_swing`) with the winner: it must add no buzz.

`summary` gives per-round means, the change against the baseline, rounds
won, and a conservative verdict:
- `decided_on` says which sway decided: `imu.low_mm` (the wrist camera,
  or their own attached IMU: `external_imu_runs`), or `enc.low_mm` when the
  runs have no IMU.
- `better` — beat the baseline in every round, ≥ 5% IMU sway (≥ 15%
  encoder sway, ≥ 10% ripple), no buzz cost: adopt (encoders only: see the
  section above).
- `trade` — less sway but 3–15 Hz / accel (encoders: 3–15 Hz) up > 10%:
  don't adopt.
- `inconclusive` — keep the default. `rejected: guard trips` — never adopt.

On the shipped gains a real improvement is only ~5–10%, inside the ±10–15%
run noise: ≥ 4 interleaved rounds. The runner halts on its own (`HALTED` in
the session dir says why): a trip on a baseline, two trips in a row, an arm
left holding, a crash. `touch <session>/STOP` stops it between runs.

**Persisting a winner:** the panel (Settings → Advanced → Axol →
`left.shoulder_3.stribeck_gain`) — reversible, per arm, picked up by teleop,
the SDK and every tool. Then record during normal use again (Phase 1) and ask
the person whether it *feels* better.

## Phase 5 — Wrap up

- Restart anything you stopped (`sudo systemctl start axol.service`).
- Tell the person exactly what changed (settings keys and values, files
  rewritten, backups made) — and anything still wrong that they chose to
  live with (e.g. gravity comp not set for their end-effector).
- Send Almond: `python scripts/robot_report.py --bundle ~/axol_report.tar.gz`,
  the recordings (`~/issue.log`, `~/factory.log`), the session dirs
  (`~/tuning/*`: queue, results, logs, HALTED), and a short note of the
  symptom and what you tried.

## Hand off to Almond when

- The summary shows `control ringing` and none of the levers wins.
- A motor reports faults, `no status reply`, or zero / wrap errors
  ("neither it nor a ±360° single-turn wrap …").
- Anything involving firmware, motor replacement, or the realtime core.
- The robot did something you can't explain from a recording.
