import { useEffect, useState } from "react"
import { AlertTriangle, Check, Copy, Loader2, Plug, Rocket, Server, X } from "lucide-react"
import { serverHttpBase } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { Badge } from "@/components/ui/badge"

export type ConnState = "loading" | "ok" | "err"

const QUICKSTART: { label: string; hint?: string; cmd: string }[] = [
  {
    label: "1. Install uv",
    cmd: "curl -LsSf https://astral.sh/uv/install.sh | sh",
  },
  {
    label: "2. Install the Axol CLI globally",
    hint: "straight from GitHub",
    cmd: 'uv tool install --python 3.13 "almond-axol[sim] @ git+ssh://git@github.com/almond-bot/axol.git"',
  },
  {
    label: "3. Launch this control panel",
    cmd: "axol serve",
  },
]

/**
 * One-time setup, kept out of the main layout: connect to the machine running
 * `axol serve`, plus copyable install commands. Opened from the nav pill.
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
            <span className="font-heading text-base font-semibold">Setup</span>
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
              <Label htmlFor="setup-server-host">Server address</Label>
              <ConnBadge state={conn.state} />
            </div>
            <p className="text-xs text-white/45">
              The machine running <span className="font-mono">axol serve</span>. Every command and
              the live logs are sent there.
            </p>
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
              <div className="flex flex-col gap-1 text-xs text-red-400">
                <span className="flex items-center gap-1.5">
                  <AlertTriangle className="size-3" />
                  Can&apos;t reach {base || "the server"}.
                </span>
                {base && (
                  <span className="text-white/45">
                    If it&apos;s running, the TLS certificate may need a one-time approval — open{" "}
                    <a
                      href={base}
                      target="_blank"
                      rel="noreferrer"
                      className="text-[#eff483] underline underline-offset-2"
                    >
                      {base}
                    </a>{" "}
                    in a new tab, accept the warning, then Connect again.
                  </span>
                )}
              </div>
            )}
          </section>

          <section className="flex flex-col gap-3 border-t border-white/10 pt-5">
            <div className="flex items-center gap-2">
              <Rocket className="size-4 text-[#eff483]" />
              <span className="font-heading text-sm font-semibold">Quickstart</span>
              <span className="text-xs text-white/40">install the CLI &amp; run the server</span>
            </div>
            {QUICKSTART.map((step) => (
              <div key={step.label} className="flex flex-col gap-1.5">
                <div className="flex items-baseline justify-between gap-3">
                  <span className="text-sm font-medium text-white/80">{step.label}</span>
                  {step.hint && <span className="shrink-0 text-xs text-white/35">{step.hint}</span>}
                </div>
                <CommandLine cmd={step.cmd} />
              </div>
            ))}
          </section>
        </div>
      </div>
    </div>
  )
}

function ConnBadge({ state }: { state: ConnState }) {
  if (state === "ok") return <Badge variant="success">Connected</Badge>
  if (state === "err") return <Badge variant="destructive">Offline</Badge>
  return <Badge variant="warning">Connecting…</Badge>
}

function CommandLine({ cmd }: { cmd: string }) {
  const [copied, setCopied] = useState(false)

  async function copy() {
    try {
      await navigator.clipboard.writeText(cmd)
      setCopied(true)
      setTimeout(() => setCopied(false), 1500)
    } catch {
      // clipboard unavailable (e.g. non-secure context) — ignore.
    }
  }

  return (
    <div className="flex items-center gap-2 rounded-lg border border-white/10 bg-black/30 px-3 py-2">
      <code className="flex-1 overflow-x-auto font-mono text-xs whitespace-pre text-white/80">
        {cmd}
      </code>
      <button
        type="button"
        onClick={copy}
        title="Copy"
        className="shrink-0 text-white/40 transition-colors hover:text-white/80"
      >
        {copied ? <Check className="size-4 text-[#eff483]" /> : <Copy className="size-4" />}
      </button>
    </div>
  )
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
      <span className="max-w-[14ch] truncate font-mono">{label}</span>
    </button>
  )
}
