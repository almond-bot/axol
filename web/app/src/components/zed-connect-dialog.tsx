import { useEffect, useState } from "react"
import { AlertTriangle, Camera, Check, Loader2, Plug, X } from "lucide-react"
import { fetchBoxInfo, zedConnect, type ServerInfo, type ZedLinkStatus } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

const selectClass =
  "h-9 w-full rounded-md border border-input bg-white/[0.02] px-3 text-sm text-foreground outline-none focus-visible:border-ring/70 disabled:opacity-50"

/**
 * Lightweight ZED box link dialog. Verifies the box's `axol serve` is reachable
 * and stores the box URL + wired interfaces on the host. Clock-sync + camera
 * streaming still start when a collect-data / run-policy task begins.
 */
export function ZedConnectDialog({
  open,
  onClose,
  hostInfo,
  initial,
  onConnected,
}: {
  open: boolean
  onClose: () => void
  hostInfo: ServerInfo | null
  initial: ZedLinkStatus | null
  onConnected: (status: ZedLinkStatus) => void
}) {
  const [url, setUrl] = useState(initial?.boxUrl ?? "")
  const [hostIface, setHostIface] = useState(initial?.hostIface ?? hostInfo?.ethIface ?? "")
  const [boxIface, setBoxIface] = useState(initial?.boxIface ?? "")
  const [boxIfaces, setBoxIfaces] = useState<string[]>([])
  const [probe, setProbe] = useState<{ state: "idle" | "loading" | "ok" | "err"; msg?: string }>({
    state: "idle",
  })
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null

  async function test() {
    if (!url.trim()) return
    setProbe({ state: "loading" })
    try {
      const info = await fetchBoxInfo(url.trim())
      setBoxIfaces(info.ethIfaces ?? [])
      setProbe({ state: "ok", msg: info.hostname })
      if (!boxIface && info.ethIface) setBoxIface(info.ethIface)
    } catch (e) {
      setProbe({ state: "err", msg: String(e) })
    }
  }

  async function connect() {
    if (!url.trim()) return
    setBusy(true)
    setError(null)
    try {
      const status = await zedConnect(url.trim(), hostIface || undefined, boxIface || undefined)
      onConnected(status)
      onClose()
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onClose} aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-lg rounded-2xl border border-white/10 bg-[#161616] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <Camera className="size-4 text-[#eff483]" />
            <span className="font-heading text-base font-semibold">Connect ZED box</span>
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

        <div className="flex flex-col gap-5 p-5">
          <p className="text-xs text-white/45">
            The ZED box runs its own <span className="font-mono">axol serve</span>. This only checks
            it&apos;s reachable and remembers the wired link — streaming starts when you record or
            run a policy.
          </p>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="zed-box-url">ZED box address</Label>
            <div className="flex gap-2">
              <Input
                id="zed-box-url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="192.168.1.50:8090"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <Button
                type="button"
                variant="outline"
                size="sm"
                className="shrink-0"
                disabled={!url.trim() || probe.state === "loading"}
                onClick={test}
              >
                {probe.state === "loading" ? <Loader2 className="animate-spin" /> : <Plug />}
                Test
              </Button>
            </div>
            {probe.state === "ok" && (
              <p className="flex items-center gap-1.5 text-xs text-emerald-400/90">
                <Check className="size-3" />
                Reachable — {probe.msg}
              </p>
            )}
            {probe.state === "err" && (
              <p className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertTriangle className="size-3" />
                {probe.msg}
              </p>
            )}
          </div>

          <div className="grid grid-cols-2 gap-4">
            <div className="flex flex-col gap-1.5">
              <Label>Host interface</Label>
              <IfaceSelect
                value={hostIface}
                candidates={hostInfo?.ethIfaces ?? []}
                onChange={setHostIface}
              />
            </div>
            <div className="flex flex-col gap-1.5">
              <Label>ZED box interface</Label>
              <IfaceSelect value={boxIface} candidates={boxIfaces} onChange={setBoxIface} />
            </div>
          </div>

          {error && <p className="text-sm text-red-400">{error}</p>}

          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={connect} disabled={busy || !url.trim()}>
              {busy ? <Loader2 className="animate-spin" /> : <Plug />}
              Connect
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}

function IfaceSelect({
  value,
  candidates,
  onChange,
}: {
  value: string
  candidates: string[]
  onChange: (v: string) => void
}) {
  const known = candidates.includes(value) || value === ""
  if (candidates.length > 0 && known) {
    return (
      <select className={selectClass} value={value} onChange={(e) => onChange(e.target.value)}>
        <option value="" className="bg-[#1a1a1a]">
          Select…
        </option>
        {candidates.map((c) => (
          <option key={c} value={c} className="bg-[#1a1a1a]">
            {c}
          </option>
        ))}
      </select>
    )
  }
  return (
    <Input
      value={value}
      onChange={(e) => onChange(e.target.value)}
      placeholder="eth0"
      spellCheck={false}
      autoCapitalize="off"
      autoCorrect="off"
    />
  )
}
