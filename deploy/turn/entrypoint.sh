#!/bin/sh
# Generate coturn's config at boot so credentials come from Fly secrets (never
# baked into the image) and the public IP is injected for correct TURN
# allocations. See README.md for the deploy steps.
set -eu

# Fly machines only know their private IP; the dedicated public IPv4 must be
# supplied so coturn advertises a reachable relay address (XOR-RELAYED-ADDRESS).
# Set REAL_EXTERNAL_IP to the address from `fly ips list` (see README).
EXTERNAL_IP="${REAL_EXTERNAL_IP:-$(detect-external-ip)}"

: "${TURN_USERNAME:?set TURN_USERNAME (fly secrets set TURN_USERNAME=...)}"
: "${TURN_PASSWORD:?set TURN_PASSWORD (fly secrets set TURN_PASSWORD=...)}"
TURN_REALM="${TURN_REALM:-axol.turn}"

cat >/tmp/turnserver.conf <<EOF
listening-port=3478

# On Fly, UDP listeners must bind to fly-global-services so replies carry the
# public source address; 0.0.0.0 covers the TCP path. Relayed media egresses
# from fly-global-services for the same reason.
listening-ip=0.0.0.0
listening-ip=fly-global-services
relay-ip=fly-global-services

# The public dedicated IPv4 that coturn advertises in TURN allocations.
external-ip=${EXTERNAL_IP}

# Static long-term credential — matches AXOL_TURN_USERNAME / AXOL_TURN_PASSWORD
# on the robot machine.
lt-cred-mech
user=${TURN_USERNAME}:${TURN_PASSWORD}
realm=${TURN_REALM}

fingerprint
no-multicast-peers
no-cli
EOF

echo "starting coturn (external-ip=${EXTERNAL_IP}, realm=${TURN_REALM})"
exec turnserver -c /tmp/turnserver.conf --log-file=stdout --simple-log
