/**
 * Certificate-origin helpers for the wired Quest-over-USB pose link.
 *
 * The USB link runs over `adb reverse`, so the headset reaches the VR server at
 * `https://localhost:8000` no matter what the WiFi connection is doing. Its
 * self-signed certificate is therefore a separate origin from the LAN host's,
 * and it can — and must — be authorized before the operator connects, otherwise
 * the poses silently fall back to WiFi. The LAN host origin is still required
 * for the session itself and camera video, so both buttons stay available.
 */

/** Origin whose certificate the headset must trust for the USB pose link. */
export function usbCertOrigin(port: number): string {
  return `https://localhost:${port}`
}

/**
 * Whether to offer the LAN host's certificate button.
 *
 * USB carries the controller poses only: the session itself, and all camera
 * video, still run over `wss://{hostname}:8000`. So the host certificate has to
 * be authorizable in USB mode too — hiding it there leaves a headset that has
 * not already trusted the LAN origin with no way to connect at all.
 */
export function hostCertAuthorizeVisible(hostname: string): boolean {
  return hostname.trim() !== ""
}
