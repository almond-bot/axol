import { useEffect } from "react"
import { AlertTriangle, Loader2, Plug, Server, ShieldCheck, X } from "lucide-react"
import { serverHttpBase } from "@/lib/supervisor"
import { authorizeCert } from "@/lib/cert-accept"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

export type ConnState = "loading" | "ok" | "err" | "idle"

/**
 * One-time setup, kept out of the main layout: connect to the machine running
 * `axol serve`. Opened from the nav pill. (Install commands live in the
 * Quickstart dialog in the nav bar.)
 */
export function SetupDialog({
  open,
  onClose,
  host,
  onChangeHost,
  conn,
  onConnect,
}: {
  open: boolean
  onClose: () => void
  host: string
  onChangeHost: (value: string) => void
  conn: { state: ConnState; message?: string }
  onConnect: () => void
}) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null
  const base = serverHttpBase(host)

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onClose} aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-lg rounded-2xl border border-white/10 bg-[#161616] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <Server className="size-4 text-[#eff483]" />
            <span className="font-heading text-base font-semibold">Axol Host</span>
          </div>
          <button
            type="button"
            onClick={onClose}
            className="text-white/40 transition-colors hover:text-white/80"
            aria-label="Close"
          >
            <X className="size-5" />
          </button>
        </div>

        <div className="flex flex-col gap-6 p-5">
          <section className="flex flex-col gap-2">
            <div className="flex items-center justify-between gap-3">
              <Label htmlFor="setup-server-host">Axol Host Address</Label>
              <ConnBadge state={conn.state} />
            </div>
            <p className="text-xs text-white/45">The host connected to Axol.</p>
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                onConnect()
              }}
            >
              <Input
                id="setup-server-host"
                value={host}
                onChange={(e) => onChangeHost(e.target.value)}
                placeholder="192.168.1.42"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <Button
                type="submit"
                variant="outline"
                size="sm"
                className="shrink-0"
                disabled={conn.state === "loading"}
              >
                {conn.state === "loading" ? <Loader2 className="animate-spin" /> : <Plug />}
                Connect
              </Button>
            </form>
            {conn.state === "err" && (
              <div className="flex flex-col gap-2 text-xs text-red-400">
                <span className="flex items-center gap-1.5">
                  <AlertTriangle className="size-3" />
                  Can&apos;t reach {base || "the server"}.
                </span>
                {base && (
                  <>
                    <span className="text-white/45">
                      If it&apos;s running, its self-signed TLS certificate needs a one-time
                      approval. Authorize it, accept the warning in the popup, then it reconnects.
                    </span>
                    <Button
                      type="button"
                      variant="outline"
                      size="sm"
                      className="self-start"
                      onClick={() => authorizeCert(base).then(onConnect)}
                    >
                      <ShieldCheck />
                      Authorize certificate
                    </Button>
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  )
}

function ConnBadge({ state }: { state: ConnState }) {
  if (state === "ok") return <Badge variant="success">Connected</Badge>
  if (state === "err") return <Badge variant="destructive">Offline</Badge>
  if (state === "idle") return <Badge variant="neutral">Disconnected</Badge>
  return <Badge variant="warning">Connecting…</Badge>
}

/** Compact connection status button for the nav; opens the setup dialog. */
export function ConnectionPill({
  state,
  host,
  onClick,
}: {
  state: ConnState
  host: string
  onClick: () => void
}) {
  const dot = state === "ok" ? "bg-emerald-400" : state === "err" ? "bg-red-400" : "bg-amber-400"
  const label = state === "ok" ? host || "Connected" : state === "err" ? "Offline" : "Connecting…"
  return (
    <button
      type="button"
      onClick={onClick}
      title="Server connection & setup"
      className="flex items-center gap-2 rounded-md border border-white/10 bg-white/[0.03] px-2.5 py-1.5 text-xs text-white/70 transition-colors hover:border-white/25 hover:bg-white/[0.06]"
    >
      <span
        className={`size-2 rounded-full ${dot} ${state === "loading" ? "animate-pulse" : ""}`}
      />
      <span className="font-mono whitespace-nowrap">{label}</span>
    </button>
  )
}
