import { useEffect, useState } from "react"
import { AlertTriangle, Camera, Loader2, Plug, X } from "lucide-react"
import { zedConnect, type ZedLinkStatus } from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

/**
 * Lightweight ZED box link dialog. Verifies the box's `axol serve` is reachable
 * and stores the box URL on the host. Connecting also starts PTP clock sync so
 * the clocks are locked before a task; camera streaming starts when a
 * collect-data / run-policy task begins. The PTP interfaces on both machines
 * are derived automatically from the box address.
 */
export function ZedConnectDialog({
  open,
  onClose,
  initial,
  onConnected,
}: {
  open: boolean
  onClose: () => void
  initial: ZedLinkStatus | null
  onConnected: (status: ZedLinkStatus) => void
}) {
  const [url, setUrl] = useState(initial?.boxUrl ?? "")
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null

  async function connect() {
    if (!url.trim()) return
    setBusy(true)
    setError(null)
    try {
      const status = await zedConnect(url.trim())
      onConnected(status)
      onClose()
    } catch (e) {
      setError(String(e).replace(/^Error:\s*/, ""))
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
            The ZED box runs its own <span className="font-mono">axol serve</span>. Connecting
            checks it&apos;s reachable and starts PTP clock sync between the two machines; camera
            streaming starts when you record or run a policy.
          </p>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="zed-box-url">ZED box IP</Label>
            <form
              className="flex gap-2"
              onSubmit={(e) => {
                e.preventDefault()
                connect()
              }}
            >
              <Input
                id="zed-box-url"
                value={url}
                onChange={(e) => setUrl(e.target.value)}
                placeholder="192.168.1.50"
                spellCheck={false}
                autoCapitalize="off"
                autoCorrect="off"
              />
              <Button
                type="submit"
                variant="outline"
                size="sm"
                className="shrink-0"
                disabled={busy || !url.trim()}
              >
                {busy ? <Loader2 className="animate-spin" /> : <Plug />}
                Connect
              </Button>
            </form>
            <p className="text-xs text-white/35">
              Just the IP — port <span className="font-mono">8090</span> (
              <span className="font-mono">axol serve</span>) is assumed.
            </p>
            {error && (
              <p className="flex items-center gap-1.5 text-xs text-red-400">
                <AlertTriangle className="size-3 shrink-0" />
                {error}
              </p>
            )}
          </div>
        </div>
      </div>
    </div>
  )
}
