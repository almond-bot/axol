import os
import subprocess
import sys
import tempfile
from importlib.resources import files


def cmd_setup_can():
    script_data = files("almond_axol").joinpath("script/setup_can.sh").read_bytes()
    with tempfile.NamedTemporaryFile(suffix=".sh", delete=False) as f:
        f.write(script_data)
        tmp = f.name
    try:
        os.chmod(tmp, 0o755)
        sys.exit(subprocess.call(["bash", tmp]))
    finally:
        os.unlink(tmp)


COMMANDS = {
    "setup-can": cmd_setup_can,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: almond-axol <command>")
        print()
        print("Commands:")
        print("  setup-can    Configure persistent CAN bus interfaces")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in COMMANDS:
        print(f"Unknown command: {cmd}")
        print(f"Run 'almond-axol --help' for usage.")
        sys.exit(1)

    COMMANDS[cmd]()
