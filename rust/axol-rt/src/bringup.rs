//! Motor bring-up and teardown shared by `hold` and `serve`, mirroring the
//! Python drivers' cold-enable sequences.

use std::io;
use std::time::Duration;

use crate::can::CanSock;
use crate::proto;
use crate::safety::purge_tx_queue;
use crate::txn;

pub const TIMEOUT: Duration = Duration::from_millis(100);
const SEND_TIMEOUT: Duration = Duration::from_millis(20);
/// Post-0x76 reboot settle; the Python driver measures ~1.12 s and waits 2.
pub const RESET_SETTLE: Duration = Duration::from_millis(2200);

/// Static per-joint inputs to bring-up (from a params file or CONFIG).
#[derive(Clone, Debug)]
pub struct MotorSpec {
    pub joint: String,
    pub motor_id: u8,
    /// Gains for the initial hold phase (production config values).
    pub kp: f64,
    pub kd: f64,
    /// The gripper is special-cased throughout: POSITION_FORCE mode instead
    /// of MIT, brought up (enabled + calibrated) by the Python side before
    /// the core arms, exempt from the deviation abort (contact is its job).
    pub gripper: bool,
    /// Target-tuple index this motor plays (arm joints 0-6, gripper 7);
    /// decouples the wire layout from bring-up iteration order.
    pub slot: usize,
    /// In-core target-tracker limits (rad/s, rad/s²) for tracked-mode
    /// targets — see `filter::Trapezoid`. Unused by the gripper and `hold`.
    pub max_vel: f64,
    pub max_accel: f64,
    /// Tanh friction-model parameters (`filter::friction`), applied in-core
    /// against the tracker velocity in tracked mode. Zero for the gripper.
    pub fc: f64,
    pub k: f64,
    pub fv: f64,
    pub fo: f64,
}

#[derive(Clone, Copy, PartialEq)]
pub enum Vendor {
    MyActuator,
    Damiao,
}

/// A motor that passed bring-up prep: identified, fault-free, ranges known.
#[derive(Clone)]
pub struct ReadyMotor {
    pub id: u8,
    pub joint: String,
    pub vendor: Vendor,
    pub ranges: proto::MitRanges,
    /// Measured position at prep time (motor frame, rad).
    pub hold_pos: f64,
    pub kp: f64,
    pub kd: f64,
    pub gripper: bool,
    pub slot: usize,
    /// Tracker limits + friction params, carried over from the spec.
    pub max_vel: f64,
    pub max_accel: f64,
    pub fc: f64,
    pub k: f64,
    pub fv: f64,
    pub fo: f64,
}

/// Phase 1 of a cold bring-up: MyActuator 0x76 system reset (all motors at
/// once, one settle) and Damiao clear-errors. Torque-neutral on a disabled
/// motor. Runs *before* the Python side resolves joint offsets, so the
/// multi-turn wrap state Python verifies is the post-reset one.
pub fn prep(sock: &CanSock, specs: &[MotorSpec]) -> io::Result<()> {
    let mut any_ma = false;
    for spec in specs {
        if spec.gripper {
            // The gripper's bring-up (enable, calibration, mode switch) is
            // Python's, and it may be holding an object — never touched here.
            continue;
        }
        if spec.motor_id <= 5 {
            sock.send(
                proto::MA_REQ + spec.motor_id as u16,
                &proto::ma_cmd(proto::MA_RESET),
            )?;
            any_ma = true;
        } else {
            sock.send(spec.motor_id as u16, &proto::DM_CLEAR_ERRORS)?;
        }
    }
    if any_ma {
        std::thread::sleep(RESET_SETTLE);
    }
    sock.drain()?;
    Ok(())
}

/// Phase 2: capability detection, fault checks, range reads, and position
/// reads. Read-only — the motors stay torque-off until [`enable`].
pub fn prepare(sock: &CanSock, iface: &str, specs: &[MotorSpec]) -> io::Result<Vec<ReadyMotor>> {
    let err = |msg: String| io::Error::other(format!("{iface}: {msg}"));
    let mut motors = Vec::new();

    for spec in specs.iter().filter(|s| s.motor_id <= 5) {
        let id = spec.motor_id;
        let version = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_VERSION), TIMEOUT)?
            .map(|(d, _)| proto::ma_decode_version(&d));
        let model = read_ma_model(sock, id)?;
        let (p_max, t_max) = proto::ma_mit_ranges(version, model.as_deref());

        let Some((s1, _)) =
            txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_STATUS1), TIMEOUT)?
        else {
            return Err(err(format!("{} (0x{id:02X}): no status reply", spec.joint)));
        };
        let (_, errors) = proto::ma_decode_status1(&s1);
        if errors != 0 {
            return Err(err(format!(
                "{} (0x{id:02X}): latched fault 0x{errors:04X} — not enabling",
                spec.joint
            )));
        }
        let Some((pos_frame, _)) =
            txn::ma_request(sock, id, proto::ma_cmd(proto::MA_MULTI_TURN_ANGLE), TIMEOUT)?
        else {
            return Err(err(format!(
                "{} (0x{id:02X}): no position reply",
                spec.joint
            )));
        };
        motors.push(ReadyMotor {
            id,
            joint: spec.joint.clone(),
            vendor: Vendor::MyActuator,
            ranges: proto::MitRanges {
                p_max,
                v_max: proto::MA_V_MAX,
                kp_max: proto::MA_KP_MAX,
                kd_max: proto::MA_KD_MAX,
                t_max,
            },
            hold_pos: proto::ma_decode_position(&pos_frame),
            kp: spec.kp,
            kd: spec.kd,
            gripper: false,
            slot: spec.slot,
            max_vel: spec.max_vel,
            max_accel: spec.max_accel,
            fc: spec.fc,
            k: spec.k,
            fv: spec.fv,
            fo: spec.fo,
        });
    }

    for spec in specs.iter().filter(|s| s.motor_id >= 6) {
        let id = spec.motor_id as u16;
        let mode = read_dm_register(sock, id, proto::DM_REG_CTRL_MODE)?;
        // Wrists run MIT (1); the gripper must already be in POSITION_FORCE
        // (4), set by the Python side's calibration flow before arming.
        let expected = if spec.gripper { 4.0 } else { 1.0 };
        if mode != expected {
            return Err(err(format!(
                "{} (0x{id:02X}): control mode {mode} (expected {expected}) — not enabling",
                spec.joint
            )));
        }
        let p_max = read_dm_register(sock, id, proto::DM_REG_PMAX)?;
        let v_max = read_dm_register(sock, id, proto::DM_REG_VMAX)?;
        let t_max = read_dm_register(sock, id, proto::DM_REG_TMAX)?;

        let Some((fb, _)) = txn::dm_request_feedback(sock, id, TIMEOUT)? else {
            return Err(err(format!(
                "{} (0x{id:02X}): no feedback reply",
                spec.joint
            )));
        };
        let decoded = proto::dm_decode_feedback(&fb, p_max, v_max, t_max);
        motors.push(ReadyMotor {
            id: spec.motor_id,
            joint: spec.joint.clone(),
            vendor: Vendor::Damiao,
            ranges: proto::MitRanges {
                p_max,
                v_max,
                kp_max: 500.0,
                kd_max: 5.0,
                t_max,
            },
            hold_pos: decoded.position,
            kp: spec.kp,
            kd: spec.kd,
            gripper: spec.gripper,
            slot: spec.slot,
            max_vel: spec.max_vel,
            max_accel: spec.max_accel,
            fc: spec.fc,
            k: spec.k,
            fv: spec.fv,
            fo: spec.fo,
        });
    }
    Ok(motors)
}

pub fn read_ma_model(sock: &CanSock, motor_id: u8) -> io::Result<Option<String>> {
    let mut raw = Vec::new();
    for block in [0x01u8, 0x02] {
        let req = [proto::MA_READ_MODEL, 0x01, block, 0, 0, 0, 0, 0];
        match txn::ma_request(sock, motor_id, req, TIMEOUT)? {
            Some((d, _)) => raw.extend_from_slice(&proto::ma_decode_model_block(&d)),
            None => return Ok(None),
        }
    }
    let end = raw.iter().position(|&b| b == 0).unwrap_or(raw.len());
    Ok(Some(String::from_utf8_lossy(&raw[..end]).into_owned()))
}

/// Register read with the Python driver's retry (single unacked reply frame).
pub fn read_dm_register(sock: &CanSock, motor_id: u16, rid: u8) -> io::Result<f64> {
    for _ in 0..5 {
        if let Some((value, _)) = txn::dm_read_register(sock, motor_id, rid, TIMEOUT)? {
            return Ok(value);
        }
    }
    Err(io::Error::other(format!(
        "damiao 0x{motor_id:02X}: register {rid} read timed out"
    )))
}

pub fn enable(sock: &CanSock, iface: &str, motors: &[ReadyMotor]) -> io::Result<()> {
    // A full socket buffer must not strand motors that were enabled earlier in
    // this batch or prevent the rollback below from running to completion.
    sock.set_send_timeout(SEND_TIMEOUT)?;
    // Record a motor before its enable write: a failed request may still have
    // reached the controller, even when its acknowledgement did not return.
    let mut attempted = Vec::new();
    for m in motors {
        if m.gripper {
            continue; // enabled by the Python side's calibration flow
        }
        attempted.push(m.clone());
        let result = match m.vendor {
            Vendor::MyActuator => {
                match txn::ma_request(sock, m.id, proto::ma_cmd(proto::MA_RELEASE_BRAKE), TIMEOUT) {
                    Ok(Some(_)) => Ok(()),
                    Ok(None) => Err(io::Error::other(format!(
                        "{}: brake release not acknowledged",
                        m.joint
                    ))),
                    Err(err) => Err(err),
                }
            }
            Vendor::Damiao => sock.send(m.id as u16, &proto::DM_ENABLE),
        };
        if let Err(err) = result {
            if !disable_inner(sock, &attempted) {
                let purged = purge_tx_queue(iface);
                eprintln!(
                    "{iface}: motor-enable rollback had failed writes{}",
                    if purged {
                        "; stale TX queue purged"
                    } else {
                        "; QUEUE PURGE FAILED, flap the interface before re-powering"
                    }
                );
            }
            return Err(err);
        }
    }
    Ok(())
}

/// Best-effort disable of every motor; Damiao gets the repeated-send
/// treatment the Python driver uses.
pub fn disable(sock: &CanSock, motors: &[ReadyMotor]) {
    let _ = disable_inner(sock, motors);
}

/// Return false if any rollback frame could not be written. Callers performing
/// partial-enable rollback use that signal to purge potentially stale enables.
fn disable_inner(sock: &CanSock, motors: &[ReadyMotor]) -> bool {
    let mut complete = true;
    for m in motors {
        for _ in 0..3 {
            let sent = match m.vendor {
                Vendor::MyActuator => sock.send(
                    proto::MA_REQ + m.id as u16,
                    &proto::ma_cmd(proto::MA_SHUTDOWN),
                ),
                Vendor::Damiao => sock.send(m.id as u16, &proto::DM_DISABLE),
            };
            if let Err(err) = sent {
                complete = false;
                eprintln!("disable {}: {err}", m.joint);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
    complete
}
