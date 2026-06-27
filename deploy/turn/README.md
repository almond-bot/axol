# Axol TURN relay on Fly.io

A [coturn](https://github.com/coturn/coturn) TURN server for **off-tailnet
remote teleop**. When an operator reaches the robot through a Tailscale Funnel
(or ngrok) tunnel, the control WebSocket goes through the tunnel but the WebRTC
**camera media** can't — it needs a publicly reachable relay. This deploys one
on Fly.io (which, unlike Vercel/Render, supports raw UDP).

See the guide: [`docs/guides/remote-teleop.mdx`](../../docs/guides/remote-teleop.mdx).

## Why Fly.io (and the gotchas it handles)

- TURN needs a **dedicated IPv4** — UDP doesn't work on Fly's shared v4 or on IPv6.
- Fly doesn't rewrite UDP **ports**, and declaring any one UDP port routes **all**
  inbound UDP to the machine, so coturn's `49152-65535` relay range is covered
  without listing every port.
- UDP listeners bind to `fly-global-services`, and coturn is told its public IP
  via `external-ip` (`REAL_EXTERNAL_IP`) so TURN allocations advertise a
  reachable address. The entrypoint script handles this.

## Deploy

From this directory (`deploy/turn/`), with [`flyctl`](https://fly.io/docs/flyctl/install/)
installed and `fly auth login` done:

```bash
# 1. Create the app (uses the app name in fly.toml; change it there if taken).
fly apps create axol-turn

# 2. Allocate a DEDICATED IPv4 (required for UDP; ~$2/mo). Then note the address.
fly ips allocate-v4 --app axol-turn
fly ips list --app axol-turn          # copy the v4 under ADDRESS

# 3. Set credentials + the public IP. Pick any username; generate a strong secret.
fly secrets set --app axol-turn \
  TURN_USERNAME=axol \
  TURN_PASSWORD="$(openssl rand -hex 16)" \
  REAL_EXTERNAL_IP=<the-v4-from-step-2>

# 4. Deploy.
fly deploy --app axol-turn
```

`fly secrets set` triggers a redeploy, so step 4 may be a no-op the first time —
that's fine. Confirm it's healthy:

```bash
fly logs --app axol-turn      # look for "starting coturn (external-ip=...)"
```

## Verify it relays

Open the WebRTC [Trickle ICE tester](https://webrtc.github.io/samples/src/content/peerconnection/trickle-ice/):

- Server: `turn:<your-v4>:3478`, username `axol`, the password you set
- **Gather candidates** → you should see a candidate of type **`relay`**.
  Only `host`/`srflx` means TURN isn't reachable (recheck the dedicated IPv4 and
  `REAL_EXTERNAL_IP`).

## Point Axol at it

On the **robot machine**, before `axol teleop` / `axol serve` (see
[`almond_axol/vr/ice.py`](../../almond_axol/vr/ice.py)):

```bash
export AXOL_TURN_URL="turn:<your-v4>:3478"
export AXOL_TURN_USERNAME="axol"
export AXOL_TURN_PASSWORD="<the password you set>"
```

Retrieve the password later with `fly secrets list` (shows only digests) — keep
your own copy, or just rotate it with another `fly secrets set`.

## Notes

- **Always-on**: this config doesn't enable machine auto-stop, so the relay
  stays up. TURN is only used during a session, but a relay that's asleep when
  the operator connects would fail to set up the call.
- **Bandwidth**: TURN relays all media, so the machine's egress carries your full
  camera bitrate (several Mbps per 1080p stream, both directions). Bump the VM if
  you push many streams.
- **TLS (`turns:` on 443)** isn't enabled here — plain `turn:3478` over UDP works
  on most networks. Add it only if an operator's network blocks UDP/non-443; it
  needs a domain + cert, which Fly can manage.
- **Tear down**: `fly apps destroy axol-turn` (also releases the paid IPv4).
