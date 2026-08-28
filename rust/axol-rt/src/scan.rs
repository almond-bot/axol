//! Read-only scan of every motor on the arm buses: identity, state, and
//! round-trip latency. Sends no enable or motion commands.

use std::io;
use std::time::Duration;

use crate::can::CanSock;
use crate::proto;
use crate::txn;

const TIMEOUT: Duration = Duration::from_millis(100);

pub fn run(ifaces: &[String]) -> io::Result<()> {
    for iface in ifaces {
        println!("== {iface} ==");
        match CanSock::open(iface) {
            Ok(sock) => scan_bus(&sock)?,
            Err(err) => println!("  cannot open: {err}"),
        }
        println!();
    }
    Ok(())
}

fn scan_bus(sock: &CanSock) -> io::Result<()> {
    sock.drain()?;
    for (i, &id) in proto::MA_IDS.iter().enumerate() {
        let joint = proto::JOINT_NAMES[i];
        match scan_myactuator(sock, id) {
            Ok(Some(line)) => println!("  0x{id:02X} {joint:<10} myactuator  {line}"),
            Ok(None) => println!("  0x{id:02X} {joint:<10} myactuator  NO RESPONSE"),
            Err(err) => println!("  0x{id:02X} {joint:<10} myactuator  error: {err}"),
        }
    }
    for (i, &id) in proto::DM_IDS.iter().enumerate() {
        let joint = proto::JOINT_NAMES[proto::MA_IDS.len() + i];
        match scan_damiao(sock, id as u16) {
            Ok(Some(line)) => println!("  0x{id:02X} {joint:<10} damiao      {line}"),
            Ok(None) => println!("  0x{id:02X} {joint:<10} damiao      NO RESPONSE"),
            Err(err) => println!("  0x{id:02X} {joint:<10} damiao      error: {err}"),
        }
    }
    Ok(())
}

fn scan_myactuator(sock: &CanSock, id: u8) -> io::Result<Option<String>> {
    let Some((ver, rtt)) =
        txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_VERSION), TIMEOUT)?
    else {
        return Ok(None);
    };
    let version = proto::ma_decode_version(&ver);

    let pos = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_MULTI_TURN_ANGLE), TIMEOUT)?
        .map(|(d, _)| proto::ma_decode_position(&d));
    let status2 = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_MOTOR_STATUS_2), TIMEOUT)?
        .map(|(d, _)| proto::ma_decode_status2(&d));
    let status1 = txn::ma_request(sock, id, proto::ma_cmd(proto::MA_READ_STATUS1), TIMEOUT)?
        .map(|(d, _)| proto::ma_decode_status1(&d));

    let pos_s = pos.map_or("?".into(), |p| format!("{:+8.2}°", p.to_degrees()));
    let (temp_s, volt_s, err_s) = match (status2, status1) {
        (Some((temp, _, _)), Some((volts, errors))) => (
            format!("{temp:3.0}°C"),
            format!("{volts:5.1}V"),
            if errors == 0 {
                "ok".to_string()
            } else {
                format!("ERR 0x{errors:04X}")
            },
        ),
        _ => ("?".into(), "?".into(), "?".into()),
    };
    Ok(Some(format!(
        "fw {version}  pos {pos_s}  {temp_s}  {volt_s}  {err_s}  rtt {:.2}ms",
        rtt.as_secs_f64() * 1e3
    )))
}

fn scan_damiao(sock: &CanSock, id: u16) -> io::Result<Option<String>> {
    let Some((vbus, rtt)) = txn::dm_read_register(sock, id, proto::DM_REG_VBUS, TIMEOUT)? else {
        return Ok(None);
    };
    let mode = txn::dm_read_register(sock, id, proto::DM_REG_CTRL_MODE, TIMEOUT)?
        .map_or("?".into(), |(v, _)| format!("{v:.0}"));
    let p_max = txn::dm_read_register(sock, id, proto::DM_REG_PMAX, TIMEOUT)?.map(|(v, _)| v);
    let v_max = txn::dm_read_register(sock, id, proto::DM_REG_VMAX, TIMEOUT)?.map(|(v, _)| v);
    let t_max = txn::dm_read_register(sock, id, proto::DM_REG_TMAX, TIMEOUT)?.map(|(v, _)| v);

    let fb = match (p_max, v_max, t_max) {
        (Some(p), Some(v), Some(t)) => txn::dm_request_feedback(sock, id, TIMEOUT)?
            .map(|(d, _)| proto::dm_decode_feedback(&d, p, v, t)),
        _ => None,
    };
    let (pos_s, temp_s, status_s) = match fb {
        Some(fb) => (
            format!("{:+8.2}°", fb.position.to_degrees()),
            format!("{:3.0}°C", fb.t_mos),
            format!("st 0x{:X}", fb.status),
        ),
        None => ("?".into(), "?".into(), "?".into()),
    };
    Ok(Some(format!(
        "mode {mode}  pos {pos_s}  {temp_s}  vbus {vbus:5.1}V  {status_s}  rtt {:.2}ms",
        rtt.as_secs_f64() * 1e3
    )))
}
