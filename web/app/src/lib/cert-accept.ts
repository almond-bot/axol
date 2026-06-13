/**
 * Streamline the browser's self-signed TLS override into a single tap.
 *
 * Opens the device's `/__accept` page in a script-spawned popup. Because that's
 * a top-level navigation to the device origin, the browser offers its cert
 * interstitial there; the user clicks "proceed" once and the device page closes
 * itself. We can't read across origins, so we resolve when the popup closes (or
 * after a timeout) and let the caller retry the now-trusted connection.
 *
 * Caveats: this does NOT remove the warning, only the open-tab-and-return dance.
 * The exception is per-origin (scheme + host + port) and session-scoped, so each
 * of `:8000` (VR) and `:8001` (control) is approved separately, and it must be
 * redone after a browser restart or cert rotation. A genuinely trusted cert is
 * the only way to drop the gesture entirely.
 *
 * @param origin Device origin to approve, e.g. `https://axol-host.local:8000`.
 */
export function authorizeCert(origin: string): Promise<void> {
  return new Promise((resolve) => {
    const url = `${origin.replace(/\/$/, "")}/__accept`
    const popup = window.open(url, "axol-cert-accept", "width=480,height=360")
    if (!popup) {
      // Popup blocked — fall back to a plain new tab; the user returns manually.
      window.open(url, "_blank", "noopener")
      resolve()
      return
    }
    // Resolve when the device page self-closes (or the user closes the popup).
    // The timeout is a safety net so a left-open popup never wedges the flow.
    const started = Date.now()
    const timer = setInterval(() => {
      if (popup.closed || Date.now() - started > 120_000) {
        clearInterval(timer)
        resolve()
      }
    }, 400)
  })
}
