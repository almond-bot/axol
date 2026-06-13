/**
 * Best-effort detection of a VR headset browser (Meta Quest et al.) so the
 * root route can send headsets to the immersive app and everything else to the
 * desktop control panel.
 */
export function isHeadsetBrowser(): boolean {
  if (typeof navigator === "undefined") return false
  const ua = navigator.userAgent
  return /OculusBrowser|Quest|Pico|VR/i.test(ua)
}
