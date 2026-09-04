//! axol-rt: realtime CAN services for Almond Axol and Jelly.
//!
//! Read-only tools:
//!
//!   axol-rt scan  [ifaces...]                     identity + state of every motor
//!   axol-rt bench [--hz N] [--secs N] [--serial] [ifaces...]
//!                                                 paced full-bus telemetry loop
//!
//! Motion/control services:
//!
//!   axol-rt hold --params FILE [--secs N] [--hz N] [--abort-deg N] [--yes]
//!                                                 enable + hold current pose
//!   axol-rt serve --socket PATH                   realtime core driven over a
//!                                                 Unix socket (see serve.rs)
//!   axol-rt proxy --socket PATH --iface IFACE     maintenance transport,
//!                                                 tuning, timing aggregation
//!   axol-rt jelly --socket PATH --iface IFACE     Jelly wheel controller

mod bench;
mod bringup;
mod can;
mod experiment;
mod filter;
mod hold;
mod jelly;
mod proto;
mod proxy;
mod safety;
mod scan;
mod serve;
mod timing;
mod txn;

const DEFAULT_IFACES: [&str; 2] = ["can_alm_axol_l", "can_alm_axol_r"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match args.split_first() {
        Some((cmd, rest)) => (cmd.as_str(), rest),
        None => {
            eprintln!("usage: axol-rt <scan|bench|hold|serve|proxy|jelly> [options]");
            std::process::exit(2);
        }
    };

    let result = match cmd {
        "scan" => {
            let ifaces = parse_ifaces(rest);
            scan::run(&ifaces)
        }
        "bench" => {
            let mut hz = 240.0;
            let mut secs = 5.0;
            let mut mode = bench::Mode::Pipelined;
            let mut ifaces: Vec<String> = Vec::new();
            let mut iter = rest.iter();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--hz" => hz = expect_f64(iter.next(), "--hz"),
                    "--secs" => secs = expect_f64(iter.next(), "--secs"),
                    "--serial" => mode = bench::Mode::Serial,
                    other => ifaces.push(other.to_string()),
                }
            }
            if ifaces.is_empty() {
                ifaces = DEFAULT_IFACES.iter().map(|s| s.to_string()).collect();
            }
            bench::run(&ifaces, hz, secs, mode)
        }
        "hold" => {
            let mut params: Option<String> = None;
            let mut secs = 5.0;
            let mut hz = 240.0;
            let mut abort_deg = 10.0;
            let mut yes = false;
            let mut iter = rest.iter();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--params" => params = iter.next().cloned(),
                    "--secs" => secs = expect_f64(iter.next(), "--secs"),
                    "--hz" => hz = expect_f64(iter.next(), "--hz"),
                    "--abort-deg" => abort_deg = expect_f64(iter.next(), "--abort-deg"),
                    "--yes" => yes = true,
                    other => {
                        eprintln!("hold: unknown argument {other}");
                        std::process::exit(2);
                    }
                }
            }
            let Some(params) = params else {
                eprintln!("hold: --params FILE is required (see tools/gen_hold_params.py)");
                std::process::exit(2);
            };
            hold::run(&params, secs, hz, abort_deg, yes)
        }
        "serve" => {
            let mut socket: Option<String> = None;
            let mut iter = rest.iter();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--socket" => socket = iter.next().cloned(),
                    other => {
                        eprintln!("serve: unknown argument {other}");
                        std::process::exit(2);
                    }
                }
            }
            let Some(socket) = socket else {
                eprintln!("serve: --socket PATH is required");
                std::process::exit(2);
            };
            serve::run(&socket)
        }
        "proxy" => {
            let mut socket: Option<String> = None;
            let mut iface: Option<String> = None;
            let mut iter = rest.iter();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--socket" => socket = iter.next().cloned(),
                    "--iface" => iface = iter.next().cloned(),
                    other => {
                        eprintln!("proxy: unknown argument {other}");
                        std::process::exit(2);
                    }
                }
            }
            let Some(socket) = socket else {
                eprintln!("proxy: --socket PATH is required");
                std::process::exit(2);
            };
            let Some(iface) = iface else {
                eprintln!("proxy: --iface IFACE is required");
                std::process::exit(2);
            };
            proxy::run(&socket, &iface)
        }
        "jelly" => {
            let mut socket: Option<String> = None;
            let mut iface: Option<String> = None;
            let mut iter = rest.iter();
            while let Some(arg) = iter.next() {
                match arg.as_str() {
                    "--socket" => socket = iter.next().cloned(),
                    "--iface" => iface = iter.next().cloned(),
                    other => {
                        eprintln!("jelly: unknown argument {other}");
                        std::process::exit(2);
                    }
                }
            }
            let Some(socket) = socket else {
                eprintln!("jelly: --socket PATH is required");
                std::process::exit(2);
            };
            let Some(iface) = iface else {
                eprintln!("jelly: --iface IFACE is required");
                std::process::exit(2);
            };
            jelly::run(&socket, &iface)
        }
        other => {
            eprintln!("unknown command: {other}");
            std::process::exit(2);
        }
    };

    if let Err(err) = result {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn parse_ifaces(rest: &[String]) -> Vec<String> {
    if rest.is_empty() {
        DEFAULT_IFACES.iter().map(|s| s.to_string()).collect()
    } else {
        rest.to_vec()
    }
}

fn expect_f64(value: Option<&String>, flag: &str) -> f64 {
    value.and_then(|v| v.parse().ok()).unwrap_or_else(|| {
        eprintln!("{flag} needs a numeric value");
        std::process::exit(2);
    })
}
