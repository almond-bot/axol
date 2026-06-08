import { useEffect, useState } from "react"
import { KeyRound, Loader2, Plug, X } from "lucide-react"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"

/**
 * Prompts for the sudo password needed to bring up the CAN interfaces. Only
 * shown when the robot connect reports `needsSudo` (passwordless sudo isn't
 * configured and the interfaces are down). The password is sent once to the
 * connect endpoint and never stored.
 */
export function SudoDialog({
  open,
  busy,
  error,
  onClose,
  onSubmit,
}: {
  open: boolean
  busy: boolean
  error: string | null
  onClose: () => void
  onSubmit: (password: string) => void
}) {
  const [password, setPassword] = useState("")

  useEffect(() => {
    // eslint-disable-next-line react-hooks/set-state-in-effect
    if (open) setPassword("")
  }, [open])

  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null

  function submit() {
    if (password && !busy) onSubmit(password)
  }

  return (
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onClose} aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-md rounded-2xl border border-white/10 bg-[#161616] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <KeyRound className="size-4 text-[#eff483]" />
            <span className="font-heading text-base font-semibold">sudo password</span>
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
            Bringing up the CAN interfaces needs root. Enter your sudo password to continue —
            it&apos;s used once and not stored.
          </p>

          <div className="flex flex-col gap-1.5">
            <Label htmlFor="sudo-password">Password</Label>
            <Input
              id="sudo-password"
              type="password"
              value={password}
              autoFocus
              onChange={(e) => setPassword(e.target.value)}
              onKeyDown={(e) => e.key === "Enter" && submit()}
              placeholder="••••••••"
              spellCheck={false}
              autoCapitalize="off"
              autoCorrect="off"
            />
          </div>

          {error && <p className="text-sm text-red-400">{error}</p>}

          <div className="flex justify-end gap-2">
            <Button variant="ghost" onClick={onClose}>
              Cancel
            </Button>
            <Button onClick={submit} disabled={busy || !password}>
              {busy ? <Loader2 className="animate-spin" /> : <Plug />}
              Connect
            </Button>
          </div>
        </div>
      </div>
    </div>
  )
}
