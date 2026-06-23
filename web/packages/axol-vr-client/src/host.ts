/**
 * True for loopback hosts. These are reached over a wired USB `adb reverse`
 * tunnel (Quest → Jetson) rather than the LAN, which sidesteps the WiFi
 * power-save buffering behind the ~150 ms pose gaps.
 *
 * On this path WebRTC can't be used — the `adb reverse` TCP port-forward
 * carries only the one WebSocket port, not the UDP/ICE a data channel needs —
 * so pose frames go over the WebSocket, which the server also accepts them on.
 */
export function isLoopbackHost(hostname: string): boolean {
  const h = hostname.trim().toLowerCase().replace(/^\[/, "").replace(/\]$/, "")
  return h === "localhost" || h === "127.0.0.1" || h === "::1" || h.endsWith(".localhost")
}
