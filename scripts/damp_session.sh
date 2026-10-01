#!/usr/bin/env bash
# A tip-damping session on the robot, unattended (~60 min), then the report.
#
#   bash scripts/damp_session.sh            # everything
#   bash scripts/damp_session.sh right      # right arm only (slow_osc + other motions)
#   bash scripts/damp_session.sh left       # left arm only
#
# Every damping run alternates undamped / damped passes (--imu-damp-alternate)
# so each group carries its own baseline; scripts/damp_report.py compares them.
# Run from the checkout (cd ~/axol). Keep the workspace clear: slow_osc and the
# generated training motions move the driven arm through their full range.
set -u
cd "$(dirname "$0")/.."
WHAT="${1:-all}"
START="$(date +%Y%m%d-%H%M%S)"
LOG="$HOME/damp_session_$START.log"
TRAIN="$HOME/train"   # the generated motions from 2026-09-30 (train_*.npz, fast_*.npz)
echo "damp session $START ($WHAT) — log $LOG"

# The settings under test (shared by every run below).
BAND=(--imu-damp-hp 0.3 --imu-damp-lp 40 --imu-damp-max 1.5 --imu-damp-alternate --repeat 4)

tm() { # label, then tune.motion arguments
    local label="$1"; shift
    echo "=== $label  $(date +%T)" | tee -a "$LOG"
    timeout 1200 uv run axol tune.motion "$@" --label "$label" 2>&1 | tee -a "$LOG" |
        grep --line-buffered -E "damping \(|damping reference|wrist IMU \(|Saved tuning|switched|rror|Traceback|unaway|refus"
}

saved_id() { # the last run id the log saved
    grep -oE "Saved tuning run [0-9]{8}-[0-9]{6}-[0-9a-f]{6}" "$LOG" | tail -1 | awk '{print $4}'
}

preflight() {
    # The realtime core must be built from this checkout (the config protocol
    # moves with it) and carry its scheduling capabilities.
    local BIN=rust/axol-rt/target/release/axol-rt
    if [ ! -x "$BIN" ] || [ -n "$(find rust/axol-rt/src -newer "$BIN" -name '*.rs' | head -1)" ]; then
        echo "axol-rt is out of date: (cd rust/axol-rt && cargo build --release) && sudo setcap cap_ipc_lock,cap_sys_nice=ep $BIN"
        exit 1
    fi
    getcap "$BIN" | grep -q cap_sys_nice || {
        echo "axol-rt lost its capabilities: sudo setcap cap_ipc_lock,cap_sys_nice=ep $BIN"; exit 1; }
    systemctl is-active --quiet zed_x_daemon || {
        echo "zed_x_daemon is not active: sudo systemctl restart zed_x_daemon"; exit 1; }
    uv run axol motor.health 2>&1 | grep -vE "^INFO|^WARNING" | tee -a "$LOG" | grep -q "did not respond\|timed out" && {
        echo "a motor is not answering (see above) — check power / e-stop"; exit 1; }
    return 0
}

right() {
    local S=(--motion slow_osc --arms right)
    local J=(--imu-damp-joint right.shoulder_1 --imu-damp-joint right.elbow=0.6)
    # 1. The reference: raw command (yesterday's best) against the expected
    #    path (command through the tracking models: the arm's lag is not wobble).
    tm "R cmd 120"          "${S[@]}" --imu-damp 120 --imu-damp-ref command "${J[@]}" "${BAND[@]}"
    tm "R model 120"        "${S[@]}" --imu-damp 120 --imu-damp-ref model   "${J[@]}" "${BAND[@]}"
    tm "R model 160"        "${S[@]}" --imu-damp 160 --imu-damp-ref model   "${J[@]}" "${BAND[@]}"
    # 1b. The same damper in the realtime core: no Python loop in the IMU's
    #     path, so the phase wrap should move up and allow more gain.
    tm "R core cmd 120"     "${S[@]}" --imu-damp-core --imu-damp 120 --imu-damp-ref command "${J[@]}" "${BAND[@]}"
    tm "R core model 120"   "${S[@]}" --imu-damp-core --imu-damp 120 --imu-damp-ref model   "${J[@]}" "${BAND[@]}"
    tm "R core model 180"   "${S[@]}" --imu-damp-core --imu-damp 180 --imu-damp-ref model   "${J[@]}" "${BAND[@]}"
    tm "R core model 250"   "${S[@]}" --imu-damp-core --imu-damp 250 --imu-damp-ref model   "${J[@]}" "${BAND[@]}"
    # 2. The ~11 Hz buzz: low-pass the elbow's channel only.
    tm "R model 120 ellp6"  "${S[@]}" --imu-damp 120 --imu-damp-ref model   "${J[@]}" --imu-damp-joint-lp right.elbow=6 "${BAND[@]}"
    # 3. Encoder-only damping (no camera), lag removed by the model reference.
    tm "R enc model 120"    "${S[@]}" --imu-damp 120 --imu-damp-source encoder --imu-damp-ref model --imu-damp-joint right.shoulder_1 "${BAND[@]}"
    tm "R enc model 250"    "${S[@]}" --imu-damp 250 --imu-damp-source encoder --imu-damp-ref model --imu-damp-joint right.shoulder_1 "${BAND[@]}"
    # 4. Does it hold beyond slow_osc? (generated motions inside slow_osc's envelope)
    for m in train_02 fast_00 train_04; do
        if [ -f "$TRAIN/$m.npz" ]; then
            tm "R model 120 $m" --motion "$TRAIN/$m.npz" --arms right --imu-damp 120 --imu-damp-ref model "${J[@]}" "${BAND[@]}"
        else
            echo "  (no $TRAIN/$m.npz — skipped)" | tee -a "$LOG"
        fi
    done
}

left() {
    local S=(--motion slow_osc --arms left)
    # 1. The left camera's mount, from one plain pass.
    tm "L general" "${S[@]}" --repeat 1
    local id; id="$(saved_id)"
    if [ -n "$id" ]; then
        uv run axol tune.motion --motion slow_osc --gyro-mount-fit "$id" 2>&1 | grep -E "camera mount|Wrote|poor" | tee -a "$LOG"
    fi
    # 2. The left arm's tracking models (for --imu-damp-ref model).
    for j in shoulder_1 shoulder_2 elbow; do
        uv run axol motion.chirp "left.$j" --carrier 3 2>&1 | grep -E "Wrote|outside" | tee -a "$LOG"
        tm "L chirp $j" --motion "$HOME/.almond/motions/chirp_left_$j.npz" --arms left
        id="$(saved_id)"
        [ -n "$id" ] && uv run axol tune.tf "left.$j" "$id" --save 2>&1 | tail -4 | tee -a "$LOG"
    done
    # 3. Damping with the right arm's best settings as the starting point.
    local J=(--imu-damp-joint left.shoulder_1 --imu-damp-joint left.elbow=0.6)
    tm "L cmd 120"       "${S[@]}" --imu-damp 120 --imu-damp-ref command "${J[@]}" "${BAND[@]}"
    tm "L model 120"     "${S[@]}" --imu-damp 120 --imu-damp-ref model   "${J[@]}" "${BAND[@]}"
    tm "L enc model 120" "${S[@]}" --imu-damp 120 --imu-damp-source encoder --imu-damp-ref model --imu-damp-joint left.shoulder_1 "${BAND[@]}"
}

preflight
case "$WHAT" in
    right) right ;;
    left) left ;;
    *) right; left ;;
esac
echo "=== done $(date +%T)" | tee -a "$LOG"
uv run python scripts/damp_report.py --since "$START" 2>&1 | grep -vE "INFO|pjrt|^W0" | tee "$HOME/damp_report_$START.txt"
echo "report: $HOME/damp_report_$START.txt"
