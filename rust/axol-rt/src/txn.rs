//! Request/response transactions over one bus socket, with round-trip timing.

use std::io;
use std::time::{Duration, Instant};

use crate::can::{CanSock, Frame};
use crate::proto;

/// Wait for the first frame matching `accept`, skipping unrelated traffic.
/// Returns `None` on deadline expiry.
fn wait_for(
    sock: &CanSock,
    deadline: Instant,
    mut accept: impl FnMut(&Frame) -> bool,
) -> io::Result<Option<Frame>> {
    loop {
        let now = Instant::now();
        if now >= deadline {
            return Ok(None);
        }
        match sock.recv_timeout(deadline - now)? {
            None => return Ok(None),
            Some(frame) if accept(&frame) => return Ok(Some(frame)),
            Some(_) => continue,
        }
    }
}

/// MyActuator single request/response: reply arrives on 0x240+id with the
/// command byte echoed in byte 0.
pub fn ma_request(
    sock: &CanSock,
    motor_id: u8,
    payload: [u8; 8],
    timeout: Duration,
) -> io::Result<Option<([u8; 8], Duration)>> {
    let started = Instant::now();
    sock.send(proto::MA_REQ + motor_id as u16, &payload)?;
    let resp_id = (proto::MA_RESP + motor_id as u16) as u32;
    let frame = wait_for(sock, started + timeout, |f| {
        f.id == resp_id && f.data[0] == payload[0]
    })?;
    Ok(frame.map(|f| (f.data, started.elapsed())))
}

/// Damiao register read via 0x7FF / 0x33.
pub fn dm_read_register(
    sock: &CanSock,
    motor_id: u16,
    rid: u8,
    timeout: Duration,
) -> io::Result<Option<(f64, Duration)>> {
    let started = Instant::now();
    sock.send(proto::DM_REG_ARB, &proto::dm_read_register(motor_id, rid))?;
    let frame = wait_for(sock, started + timeout, |f| {
        proto::dm_is_register_reply(&f.data, motor_id, rid)
    })?;
    Ok(frame.map(|f| (proto::dm_decode_register(&f.data, rid), started.elapsed())))
}

/// Damiao feedback request via 0x7FF / 0xCC; reply on MST_ID (0x10+id).
pub fn dm_request_feedback(
    sock: &CanSock,
    motor_id: u16,
    timeout: Duration,
) -> io::Result<Option<([u8; 8], Duration)>> {
    let started = Instant::now();
    sock.send(proto::DM_REG_ARB, &proto::dm_request_feedback(motor_id))?;
    let fb_id = (proto::DM_FEEDBACK_BASE + motor_id) as u32;
    let frame = wait_for(sock, started + timeout, |f| {
        f.id == fb_id && (f.data[0] & 0x0F) == (motor_id & 0x0F) as u8
    })?;
    Ok(frame.map(|f| (f.data, started.elapsed())))
}
