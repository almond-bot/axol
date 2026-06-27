/**
 * Resolve the operator-entered host into a connectable authority (`host:port`).
 *
 * On a LAN the operator types a bare host (`axol-host.local`, `192.168.1.20`)
 * and we default to the VR server's own port (8000). To reach the box through a
 * Tailscale Funnel / ngrok tunnel, the public endpoint is HTTPS on 443, so the
 * operator points at the tunnel host instead — supported here by:
 *   - an explicit port:   `almond-zed-box.tailb8a199.ts.net:443`
 *   - or an https/wss URL: `https://almond-zed-box.tailb8a199.ts.net`
 *     (a scheme with no port implies 443)
 *
 * An explicit port always wins; otherwise a secure scheme implies 443 and a
 * bare host falls back to `defaultPort`.
 */
export function resolveAuthority(input: string, defaultPort: number): string {
  let s = input.trim()
  const scheme = s.match(/^([a-zA-Z][a-zA-Z0-9+.-]*):\/\//)?.[1]?.toLowerCase()
  s = s.replace(/^[a-zA-Z][a-zA-Z0-9+.-]*:\/\//, "").replace(/\/.*$/, "")
  if (/:\d+$/.test(s)) return s
  if (scheme === "https" || scheme === "wss") return `${s}:443`
  return `${s}:${defaultPort}`
}

/** Build the teleop WebSocket URL (`wss://host:port/ws`) for an entered host. */
export function axolWsUrl(input: string, defaultPort: number): string {
  return `wss://${resolveAuthority(input, defaultPort)}/ws`
}

/** Build the HTTPS origin (`https://host:port`) used for cert authorization. */
export function axolHttpsOrigin(input: string, defaultPort: number): string {
  return `https://${resolveAuthority(input, defaultPort)}`
}
