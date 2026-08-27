//! Motor bring-up and teardown shared by `hold` and `serve`, mirroring the
//! Python drivers' cold-enable sequences.

use std::io;
use std::time::Duration;

use crate::can::CanSock;
use crate::proto;
use crate::txn;

pub const TIMEOUT: Duration = Duration::from_millis(100);
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
}

/// Phase 1 of a cold bring-up: MyActuator 0x76 system reset (all motors at
/// once, one settle) and Damiao clear-errors. Torque-neutral on a disabled
/// motor. Runs *before* the Python side resolves joint offsets, so the
/// multi-turn wrap state Python verifies is the post-reset one.
pub fn prep(sock: &CanSock, specs: &[MotorSpec]) -> io::Result<()> {
    let mut any_ma = false;
    for spec in specs {
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
            return Err(err(format!("{} (0x{id:02X}): no position reply", spec.joint)));
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
        });
    }

    for spec in specs.iter().filter(|s| s.motor_id >= 6) {
        let id = spec.motor_id as u16;
        let mode = read_dm_register(sock, id, proto::DM_REG_CTRL_MODE)?;
        if mode != 1.0 {
            return Err(err(format!(
                "{} (0x{id:02X}): control mode {mode} is not MIT — not enabling",
                spec.joint
            )));
        }
        let p_max = read_dm_register(sock, id, proto::DM_REG_PMAX)?;
        let v_max = read_dm_register(sock, id, proto::DM_REG_VMAX)?;
        let t_max = read_dm_register(sock, id, proto::DM_REG_TMAX)?;

        let Some((fb, _)) = txn::dm_request_feedback(sock, id, TIMEOUT)? else {
            return Err(err(format!("{} (0x{id:02X}): no feedback reply", spec.joint)));
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

pub fn enable(sock: &CanSock, motors: &[ReadyMotor]) -> io::Result<()> {
    for m in motors {
        match m.vendor {
            Vendor::MyActuator => {
                if txn::ma_request(sock, m.id, proto::ma_cmd(proto::MA_RELEASE_BRAKE), TIMEOUT)?
                    .is_none()
                {
                    return Err(io::Error::other(format!(
                        "{}: brake release not acknowledged",
                        m.joint
                    )));
                }
            }
            Vendor::Damiao => sock.send(m.id as u16, &proto::DM_ENABLE)?,
        }
    }
    Ok(())
}

/// Best-effort disable of every motor; Damiao gets the repeated-send
/// treatment the Python driver uses.
pub fn disable(sock: &CanSock, motors: &[ReadyMotor]) {
    for m in motors {
        for _ in 0..3 {
            let sent = match m.vendor {
                Vendor::MyActuator => {
                    sock.send(proto::MA_REQ + m.id as u16, &proto::ma_cmd(proto::MA_SHUTDOWN))
                }
                Vendor::Damiao => sock.send(m.id as u16, &proto::DM_DISABLE),
            };
            if let Err(err) = sent {
                eprintln!("disable {}: {err}", m.joint);
            }
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}
