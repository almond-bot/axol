import { useEffect, useState } from "react"
import { createPortal } from "react-dom"
import { Check, Copy, Rocket, X } from "lucide-react"
import { buttonVariants } from "@/components/ui/button"
import { cn } from "@/lib/utils"

const QUICKSTART: { label: string; hint?: string; cmd: string }[] = [
  {
    label: "Install Axol on the robot machine",
    hint: "installs uv + the CLI, and starts the control panel server at boot",
    cmd: "curl https://axol.almond.bot/install -fsS | bash",
  },
]

/** Nav-bar button that opens the Quickstart install/run cheatsheet. */
export function QuickstartButton() {
  const [open, setOpen] = useState(false)
  return (
    <>
      <button
        type="button"
        onClick={() => setOpen(true)}
        className={cn(buttonVariants({ variant: "ghost", size: "sm" }))}
      >
        <Rocket />
        Quickstart
      </button>
      <QuickstartDialog open={open} onClose={() => setOpen(false)} />
    </>
  )
}

function QuickstartDialog({ open, onClose }: { open: boolean; onClose: () => void }) {
  useEffect(() => {
    if (!open) return
    const onKey = (e: KeyboardEvent) => e.key === "Escape" && onClose()
    window.addEventListener("keydown", onKey)
    return () => window.removeEventListener("keydown", onKey)
  }, [open, onClose])

  if (!open) return null

  return createPortal(
    <div className="fixed inset-0 z-50 flex items-start justify-center overflow-y-auto bg-black/60 p-4 backdrop-blur-sm sm:p-8">
      <div className="absolute inset-0" onClick={onClose} aria-hidden />
      <div className="relative z-10 my-auto w-full max-w-lg rounded-2xl border border-white/10 bg-[#161616] shadow-2xl">
        <div className="flex items-center justify-between border-b border-white/10 px-5 py-4">
          <div className="flex items-center gap-2">
            <Rocket className="size-4 text-[#eff483]" />
            <span className="font-heading text-base font-semibold">Quickstart</span>
            <span className="text-xs text-white/40">one command installs everything</span>
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

        <div className="flex flex-col gap-3 p-5">
          {QUICKSTART.map((step) => (
            <div key={step.label} className="flex flex-col gap-1.5">
              <div className="flex items-baseline justify-between gap-3">
                <span className="text-sm font-medium text-white/80">{step.label}</span>
                {step.hint && <span className="shrink-0 text-xs text-white/35">{step.hint}</span>}
              </div>
              <CommandLine cmd={step.cmd} />
            </div>
          ))}
          <p className="text-xs text-white/45">
            The server starts automatically (and at every boot, staying in sync with the latest
            release). Once it&apos;s running, press <span className="text-white/70">Connect</span>{" "}
            in the top bar and enter the machine&apos;s IP address.
          </p>
        </div>
      </div>
    </div>,
    document.body
  )
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
      <code className="flex-1 font-mono text-xs break-words whitespace-pre-wrap text-white/80">
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
