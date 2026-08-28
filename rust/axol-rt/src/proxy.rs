//! Raw CAN transport owned by Rust for Python maintenance clients.
//!
//! Python sends framed `S` messages over a Unix socket and receives framed
//! `F` messages. It never opens a SocketCAN socket; existing motor protocol
//! decoders can therefore remain in Python while every actual CAN syscall is
//! centralized in this process.

use crate::can::CanSock;
use crate::experiment::{FeedbackStore, RawSample};
use crate::safety::{guarded_send, purge_tx_queue, SendOutcome, STALL_DETECT};
use crate::timing::TimingAggregator;
use std::collections::HashMap;
use std::io::{self, Read, Write};
use std::net::Shutdown;
use std::os::unix::net::{UnixListener, UnixStream};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Condvar, Mutex};
use std::time::{Duration, Instant, SystemTime, UNIX_EPOCH};

const MAX_MESSAGE: usize = 32 * 1024 * 1024;

fn read_message(stream: &mut UnixStream) -> io::Result<Option<Vec<u8>>> {
    let mut header = [0u8; 4];
    match stream.read_exact(&mut header) {
        Ok(()) => {}
        Err(err) if err.kind() == io::ErrorKind::UnexpectedEof => return Ok(None),
        Err(err) => return Err(err),
    }
    let size = u32::from_le_bytes(header) as usize;
    if size == 0 || size > MAX_MESSAGE {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            format!("proxy message size {size}"),
        ));
    }
    let mut payload = vec![0u8; size];
    stream.read_exact(&mut payload)?;
    Ok(Some(payload))
}

fn write_message(stream: &Arc<Mutex<UnixStream>>, payload: &[u8]) -> io::Result<()> {
    let mut stream = stream.lock().unwrap();
    stream.write_all(&(payload.len() as u32).to_le_bytes())?;
    stream.write_all(payload)
}

fn write_error(stream: &Arc<Mutex<UnixStream>>, message: &str) {
    let mut payload = Vec::with_capacity(1 + message.len());
    payload.push(b'E');
    payload.extend_from_slice(message.as_bytes());
    let _ = write_message(stream, &payload);
}

fn receive_loop(
    socket: Arc<CanSock>,
    output: Arc<Mutex<UnixStream>>,
    muted: Arc<AtomicBool>,
    stop: Arc<AtomicBool>,
    timing: Arc<Mutex<TimingAggregator>>,
    feedback: FeedbackStore,
    observer_mode: Arc<AtomicBool>,
) -> io::Result<()> {
    let mut feedback_generation = 0u64;
    let mut last_forward: HashMap<u32, Instant> = HashMap::new();
    let mut next_timing = Instant::now();
    while !stop.load(Ordering::Acquire) {
        let frame = match socket.recv() {
            Ok(Some(frame)) => frame,
            Ok(None) => continue,
            Err(err) => {
                let message = format!("CAN receive failed: {err}");
                write_error(&output, &message);
                let _ = output.lock().unwrap().shutdown(Shutdown::Both);
                return Err(io::Error::new(err.kind(), message));
            }
        };
        let received = Instant::now();
        feedback_generation = feedback_generation.wrapping_add(1);
        {
            let mut samples = feedback.0.lock().unwrap();
            samples.insert(
                frame.id,
                RawSample {
                    data: frame.data,
                    at: received,
                    generation: feedback_generation,
                },
            );
            feedback.1.notify_all();
        }
        let timing_payload = {
            let mut timing = timing.lock().unwrap();
            timing.observe(frame.id, &frame.data, received);
            if received >= next_timing {
                next_timing = received + Duration::from_millis(100);
                timing.snapshot_json(received).map(|json| {
                    let mut payload = Vec::with_capacity(1 + json.len());
                    payload.push(b'T');
                    payload.extend_from_slice(json.as_bytes());
                    payload
                })
            } else {
                None
            }
        };
        if let Some(payload) = timing_payload {
            if write_message(&output, &payload).is_err() {
                break;
            }
        }
        if muted.load(Ordering::Relaxed) {
            continue;
        }
        let register_reply = (0x11..=0x18).contains(&frame.id)
            && frame.data[1] <= 0x0f
            && matches!(frame.data[2], 0x33 | 0x55 | 0xaa | 0xcc);
        if observer_mode.load(Ordering::Relaxed)
            && !register_reply
            && last_forward
                .get(&frame.id)
                .is_some_and(|at| received.duration_since(*at) < Duration::from_millis(33))
        {
            continue;
        }
        last_forward.insert(frame.id, received);
        let timestamp_ns = SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
            .min(u64::MAX as u128) as u64;
        let mut payload = Vec::with_capacity(1 + 4 + 1 + 8 + 8);
        payload.push(b'F');
        payload.extend_from_slice(&frame.id.to_le_bytes());
        payload.push(frame.len);
        payload.extend_from_slice(&frame.data);
        payload.extend_from_slice(&timestamp_ns.to_le_bytes());
        if write_message(&output, &payload).is_err() {
            break;
        }
    }
    Ok(())
}

pub fn run(socket_path: &str, iface: &str) -> io::Result<()> {
    let _ = std::fs::remove_file(socket_path);
    let listener = UnixListener::bind(socket_path)?;
    let (mut stream, _) = listener.accept()?;

    let socket = Arc::new(CanSock::open(iface)?);
    socket.set_recv_timeout(Duration::from_millis(100))?;
    socket.set_send_timeout(Duration::from_millis(20))?;

    let output = Arc::new(Mutex::new(stream.try_clone()?));
    let muted = Arc::new(AtomicBool::new(false));
    let stop = Arc::new(AtomicBool::new(false));
    let timing = Arc::new(Mutex::new(TimingAggregator::new()));
    let feedback: FeedbackStore = Arc::new((Mutex::new(HashMap::new()), Condvar::new()));
    let observer_mode = Arc::new(AtomicBool::new(false));
    write_message(&output, b"R")?;
    let receiver = std::thread::spawn({
        let socket = Arc::clone(&socket);
        let output = Arc::clone(&output);
        let muted = Arc::clone(&muted);
        let stop = Arc::clone(&stop);
        let timing = Arc::clone(&timing);
        let feedback = Arc::clone(&feedback);
        let observer_mode = Arc::clone(&observer_mode);
        move || receive_loop(socket, output, muted, stop, timing, feedback, observer_mode)
    });

    let result = (|| -> io::Result<()> {
        let mut enobufs_since: Option<Instant> = None;
        while let Some(payload) = read_message(&mut stream)? {
            match payload[0] {
                b'S' => {
                    // S, arbitration id u32 LE, DLC u8, eight data bytes.
                    if payload.len() != 14 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("bad send message size {}", payload.len()),
                        ));
                    }
                    let id = u32::from_le_bytes(payload[1..5].try_into().unwrap());
                    let len = payload[5] as usize;
                    if id > 0x7ff || len > 8 {
                        return Err(io::Error::new(
                            io::ErrorKind::InvalidData,
                            format!("bad CAN frame id={id:#x} len={len}"),
                        ));
                    }
                    match guarded_send(&socket, id as u16, &payload[6..6 + len], &mut enobufs_since)
                    {
                        Ok(SendOutcome::Sent | SendOutcome::Dropped) => {}
                        Ok(SendOutcome::Stalled) => {
                            let purged = purge_tx_queue(iface);
                            let message = format!(
                                "CAN {iface} TX queue stalled >{}s (e-stop or \
                                 unpowered motors); commands stopped{}",
                                STALL_DETECT.as_secs(),
                                if purged {
                                    ", stale queue purged"
                                } else {
                                    " — QUEUE PURGE FAILED; flap the interface \
                                     before re-powering"
                                }
                            );
                            write_error(&output, &message);
                            return Err(io::Error::other(message));
                        }
                        Err(err) => {
                            let message = format!("CAN {iface} send failed: {err}");
                            write_error(&output, &message);
                            return Err(io::Error::new(err.kind(), message));
                        }
                    }
                }
                b'M' if payload.len() == 2 => {
                    muted.store(payload[1] != 0, Ordering::Release);
                }
                b'Z' if payload.len() == 1 => timing.lock().unwrap().reset(),
                b'O' if payload.len() == 2 => {
                    observer_mode.store(payload[1] != 0, Ordering::Release)
                }
                b'X' => {
                    muted.store(false, Ordering::Release);
                    match crate::experiment::run(
                        &payload[1..],
                        &socket,
                        &feedback,
                        &mut enobufs_since,
                        &stream,
                    ) {
                        Ok(response) => write_message(&output, &response)?,
                        Err(err) if err.kind() == io::ErrorKind::Interrupted => {
                            write_message(&output, b"K")?;
                        }
                        Err(err) => {
                            let message = format!("tuning experiment failed: {err}");
                            write_error(&output, &message);
                            return Err(io::Error::new(err.kind(), message));
                        }
                    }
                }
                b'Q' if payload.len() == 1 => break,
                tag => {
                    return Err(io::Error::new(
                        io::ErrorKind::InvalidData,
                        format!("unknown proxy tag {tag:#x}"),
                    ));
                }
            }
        }
        Ok(())
    })();

    stop.store(true, Ordering::Release);
    let receiver_result = receiver
        .join()
        .map_err(|_| io::Error::other("CAN proxy receive thread panicked"))?;
    let _ = std::fs::remove_file(socket_path);
    result.and(receiver_result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn framing_roundtrip() {
        let path = format!("/tmp/axol-rt-proxy-test-{}", std::process::id());
        let listener = UnixListener::bind(&path).unwrap();
        let client = UnixStream::connect(&path).unwrap();
        let (mut server, _) = listener.accept().unwrap();
        let output = Arc::new(Mutex::new(client));
        write_message(&output, b"hello").unwrap();
        assert_eq!(read_message(&mut server).unwrap(), Some(b"hello".to_vec()));
        let _ = std::fs::remove_file(path);
    }
}
