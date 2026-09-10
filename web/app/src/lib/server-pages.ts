/**
 * Backend-served pages the panel links to from its navigation.
 *
 * A package built on `almond-axol` can serve pages of its own from the
 * `axol serve` process (`almond_axol.serve.register_app_extension`) and
 * advertise them with `register_page`; `GET /api/info` then lists them as
 * `pages: [{label, path, description?}]`. This panel is often not served by
 * that backend — the hosted site drives a station on the LAN — so such a page
 * is reachable only on the backend's origin, never as a relative link here.
 * `serverPageUrl` builds the right href for either case.
 */

export interface ServerPage {
  /** Navigation text. */
  label: string
  /** Absolute path on the backend origin, e.g. "/finetune". */
  path: string
  /** Optional hover text. */
  description?: string
}

/**
 * Validate the `pages` field of `/api/info` (an untrusted JSON value) into the
 * entries the nav can render: non-empty label, absolute non-API path.
 */
export function normalizeServerPages(raw: unknown): ServerPage[] {
  if (!Array.isArray(raw)) return []
  const pages: ServerPage[] = []
  const seen = new Set<string>()
  for (const item of raw) {
    if (!item || typeof item !== "object") continue
    const { label, path, description } = item as Record<string, unknown>
    if (typeof label !== "string" || !label.trim()) continue
    if (typeof path !== "string" || !isAbsolutePagePath(path)) continue
    if (seen.has(path)) continue
    seen.add(path)
    const page: ServerPage = { label: label.trim(), path }
    if (typeof description === "string" && description.trim()) page.description = description.trim()
    pages.push(page)
  }
  return pages
}

/** "/x", "/x/y?q" — but not "//host", "x", or anything under /api/. */
export function isAbsolutePagePath(path: string): boolean {
  return path.startsWith("/") && !path.startsWith("//") && !path.startsWith("/api/")
}

/**
 * The href for a backend page given the panel's server base (`""` when the
 * panel is served by the backend itself, otherwise its `https://host:port`
 * origin). A same-origin page keeps the relative path so dev proxies and
 * local bundles work unchanged.
 */
export function serverPageUrl(serverBase: string, path: string): string {
  if (!serverBase) return path
  return `${serverBase.replace(/\/+$/, "")}${path}`
}
