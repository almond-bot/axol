/**
 * Gating for the wired Quest-over-USB pose link.
 *
 * The USB link runs over `adb reverse`, so the headset reaches the VR server at
 * `https://localhost:8000` no matter what the WiFi connection is doing. Its
 * self-signed certificate is therefore a separate origin from the LAN host's,
 * and it can — and must — be authorized before the operator connects, otherwise
 * the poses silently fall back to WiFi.
 */

/** Origin whose certificate the headset must trust for the USB pose link. */
export function usbCertOrigin(port: number): string {
  return `https://localhost:${port}`
}

/**
 * Whether to offer the LAN host's certificate button.
 *
 * In USB mode the host origin is the wrong one to authorize, so only the USB
 * button is shown.
 */
export function hostCertAuthorizeVisible(usbSelected: boolean, hostname: string): boolean {
  return !usbSelected && hostname.trim() !== ""
}
