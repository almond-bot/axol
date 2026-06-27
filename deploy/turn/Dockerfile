# TURN relay for off-tailnet remote teleop (see ../../docs/guides/remote-teleop.mdx).
#
# Runs coturn on Fly.io so an operator outside the robot's network has a
# publicly reachable relay for the WebRTC camera media (which can't traverse a
# Tailscale Funnel / ngrok tunnel). The static long-term credential matches the
# AXOL_TURN_USERNAME / AXOL_TURN_PASSWORD env vars read on the robot machine.
#
# Pin a specific coturn release for reproducibility if you prefer (e.g. 4.6.2).
FROM coturn/coturn:latest

COPY entrypoint.sh /usr/local/bin/axol-turn-entrypoint.sh

USER root
RUN chmod +x /usr/local/bin/axol-turn-entrypoint.sh

ENTRYPOINT ["/usr/local/bin/axol-turn-entrypoint.sh"]
