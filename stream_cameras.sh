#!/usr/bin/env bash
# Start HEVC streaming from all three ZED-X One cameras.
# Streams are sent on:
#   overhead  (S/N 308053933) → port 30000
#   left-arm  (S/N 305042468) → port 30002
#   right-arm (S/N 304438879) → port 30004

set -euo pipefail

IFACE="${1:-enP8p1s0}"

echo "Starting ZED camera streams on interface $IFACE ..."

axol zed.stream \
    --overhead  308053933 \
    --left-arm  305042468 \
    --right-arm 304438879 \
    --setup-ip  "$IFACE"
