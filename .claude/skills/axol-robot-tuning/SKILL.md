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
10. **No live feedback from the camera IMU** and nothing that only works
    with a camera: the fix must work on robots without one.

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
- The end-effector (gripper or not, anything heavier than stock).

Check the report before anything moves:

| Report field | Problem | Do |
|---|---|---|
| `code.commit` | lacks `scripts/tuning_queue.py` | update the checkout (`git pull`), see known-issues "Updating" |
| `core.stale` true | core older than `rust/` sources | `cd rust/axol-rt && cargo build --release`, then re-grant real-time |
| `core.realtime_caps` false | `tune.motion` / teleop refuse to arm ("cannot enter SCHED_FIFO") | `sudo setcap cap_sys_nice,cap_ipc_lock=ep <core.binary>` or `axol rt.install` |
| `system.can_links` | `can_alm_axol_l/_r` missing or DOWN | `axol can.setup`; check the arm hub USB and power |
| `system.processes_on_the_robot` | `axol serve`, teleop or a tuner running | stop it before any tuning tool (`sudo systemctl stop axol.service`, or stop it in the panel); restart after |
| `calibration["calibration.json"].matches_this_hub` false | the file belongs to another robot and is ignored | recalibrate (Phase 2) |
| `settings.link_mass_com_overrides` non-empty | hand-tuned gravity in the panel | always run the factory with `--keep-gravity` |
| `settings.axol_overrides` | per-joint gains set in the panel | they win over calibration; note them, they may be the cause |
| `effective_config` | the two arms differ on a joint | asymmetry is a lead |
| `recent_runs` guard trips | earlier runs aborted | read those runs' labels |

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
sway an operator feels, with `high_mm` (3–15 Hz) and accel as the buzz cost.
Joint-ripple creeps are for screening only: on jelly every per-joint creep win
(−30…−57%) turned into at most −7% sway at the tool, and stiffer or more
damped settings traded sway for shake.

1. Make sure the wrist camera works: a `tune.motion` run must print a
   `wrist IMU (…)` line. If it says "camera did not open in time": `sudo
   systemctl restart zed_x_daemon`. Without the IMU you can screen, not decide.
2. Motions: `python scripts/creep_motions.py --all` (creeps for every joint,
   both arms, plus `slow_osc_left.npz`).
3. Baseline both arms (compare the arms with each other — an asymmetry is a
   lead):

```bash
python scripts/tuning_queue.py plan ~/tuning/base --arm right --motion slow_osc \
    --rounds 2 --repeat 3 --variant base
python scripts/tuning_queue.py plan ~/tuning/base --arm left \
    --motion ~/.almond/motions/slow_osc_left.npz --rounds 2 --repeat 3 --variant base
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
twin, a different end-effector, or what the person feels.

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
python scripts/tuning_queue.py summary ~/tuning/s3          # decides on imu.low_mm
```

Screen many candidates cheaply on the joint's creep first (`summary
--metric rip3_rms_mdeg`), then confirm only the survivors on `slow_osc`.
Also run one fast motion (`shoulder_1_no_load` or `wirst_swing`) with the
winner: it must add no buzz.

`summary` gives per-round means, the change against the baseline, rounds
won, and a conservative verdict:
- `better` — beat the baseline in every round, ≥ 5% sway (≥ 10% on ripple),
  no buzz cost: adopt.
- `trade` — less sway but 3–15 Hz / accel up > 10%: don't adopt.
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
  rewritten, backups made).
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
