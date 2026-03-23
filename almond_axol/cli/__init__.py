import sys

from . import setup_can

_COMMANDS = {
    "setup-can": setup_can.run,
}


def main():
    if len(sys.argv) < 2 or sys.argv[1] in ("-h", "--help"):
        print("Usage: almond-axol <command>")
        print()
        print("Commands:")
        for name in _COMMANDS:
            print(f"  {name}")
        sys.exit(0)

    cmd = sys.argv[1]
    if cmd not in _COMMANDS:
        print(f"Unknown command: {cmd}")
        print("Run 'almond-axol --help' for usage.")
        sys.exit(1)

    _COMMANDS[cmd]()
