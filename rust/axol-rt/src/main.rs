//! axol-rt: realtime CAN control core for the Almond Axol arms (early scaffolding).
//!
//! Current subcommands are strictly read-only — they send no enable or motion
//! commands, so they are safe to run against a powered robot at rest:
//!
//!   axol-rt scan  [ifaces...]                     identity + state of every motor
//!   axol-rt bench [--hz N] [--secs N] [--serial] [ifaces...]
//!                                                 paced full-bus telemetry loop
//!
//! Motion (requires a params file from tools/gen_hold_params.py, and --yes):
//!
//!   axol-rt hold --params FILE [--secs N] [--hz N] [--abort-deg N] [--yes]
//!                                                 enable + hold current pose

mod bench;
mod can;
mod hold;
mod proto;
mod scan;
mod txn;

const DEFAULT_IFACES: [&str; 2] = ["can_alm_axol_l", "can_alm_axol_r"];

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (cmd, rest) = match args.split_first() {
        Some((cmd, rest)) => (cmd.as_str(), rest),
        None => {
            eprintln!("usage: axol-rt <scan|bench> [options] [ifaces...]");
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
    value
        .and_then(|v| v.parse().ok())
        .unwrap_or_else(|| {
            eprintln!("{flag} needs a numeric value");
            std::process::exit(2);
        })
}
