import {
  createContext,
  useCallback,
  useContext,
  useEffect,
  useMemo,
  useRef,
  useState,
  type ReactNode,
} from "react"
import { AlertTriangle, CheckCircle2, Info, X } from "lucide-react"
import { cn } from "@/lib/utils"

export type ToastVariant = "error" | "warning" | "success" | "info"

interface ToastItem {
  id: number
  message: string
  variant: ToastVariant
  /** Milliseconds before auto-dismiss (errors linger a little longer). */
  duration: number
}

interface ToastApi {
  show: (message: string, variant?: ToastVariant, duration?: number) => void
  error: (message: string, duration?: number) => void
  warning: (message: string, duration?: number) => void
  success: (message: string, duration?: number) => void
  info: (message: string, duration?: number) => void
}

const ToastContext = createContext<ToastApi | null>(null)

const DEFAULT_DURATION: Record<ToastVariant, number> = {
  error: 7000,
  warning: 6000,
  success: 4000,
  info: 5000,
}

/**
 * App-wide floating alerts. Toasts stack at the top of the screen and fade
 * themselves out after a short delay (hovering one pauses its timer). This is
 * the single way the app surfaces transient alerts — connection failures,
 * camera problems, and the like — instead of inline red text.
 */
export function ToastProvider({ children }: { children: ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([])
  const nextId = useRef(0)

  const remove = useCallback((id: number) => {
    setToasts((prev) => prev.filter((t) => t.id !== id))
  }, [])

  const show = useCallback(
    (message: string, variant: ToastVariant = "info", duration?: number) => {
      const id = nextId.current++
      setToasts((prev) => [
        ...prev,
        { id, message, variant, duration: duration ?? DEFAULT_DURATION[variant] },
      ])
    },
    []
  )

  const api = useMemo<ToastApi>(
    () => ({
      show,
      error: (m, d) => show(m, "error", d),
      warning: (m, d) => show(m, "warning", d),
      success: (m, d) => show(m, "success", d),
      info: (m, d) => show(m, "info", d),
    }),
    [show]
  )

  return (
    <ToastContext.Provider value={api}>
      {children}
      <ToastViewport toasts={toasts} onClose={remove} />
    </ToastContext.Provider>
  )
}

export function useToast(): ToastApi {
  const ctx = useContext(ToastContext)
  if (!ctx) throw new Error("useToast must be used within a ToastProvider")
  return ctx
}

const VARIANT_STYLES: Record<
  ToastVariant,
  { wrap: string; icon: ReactNode }
> = {
  error: {
    wrap: "border-red-400/30 bg-red-400/[0.08] text-red-100",
    icon: <AlertTriangle className="size-4 shrink-0 text-red-400" />,
  },
  warning: {
    wrap: "border-amber-400/30 bg-amber-400/[0.08] text-amber-100",
    icon: <AlertTriangle className="size-4 shrink-0 text-amber-400" />,
  },
  success: {
    wrap: "border-emerald-400/30 bg-emerald-400/[0.08] text-emerald-100",
    icon: <CheckCircle2 className="size-4 shrink-0 text-emerald-400" />,
  },
  info: {
    wrap: "border-white/15 bg-white/[0.06] text-white/85",
    icon: <Info className="size-4 shrink-0 text-sky-400" />,
  },
}

function ToastViewport({
  toasts,
  onClose,
}: {
  toasts: ToastItem[]
  onClose: (id: number) => void
}) {
  return (
    <div className="pointer-events-none fixed inset-x-0 top-4 z-[100] flex flex-col items-center gap-2 px-4">
      {toasts.map((t) => (
        <Toast key={t.id} toast={t} onClose={() => onClose(t.id)} />
      ))}
    </div>
  )
}

function Toast({ toast, onClose }: { toast: ToastItem; onClose: () => void }) {
  const { wrap, icon } = VARIANT_STYLES[toast.variant]
  const timer = useRef<number | undefined>(undefined)

  const arm = useCallback(() => {
    timer.current = window.setTimeout(onClose, toast.duration)
  }, [onClose, toast.duration])

  useEffect(() => {
    arm()
    return () => window.clearTimeout(timer.current)
  }, [arm])

  return (
    <div
      role="alert"
      onMouseEnter={() => window.clearTimeout(timer.current)}
      onMouseLeave={arm}
      className={cn(
        "animate-toast-in pointer-events-auto flex w-full max-w-md items-start gap-2.5 rounded-xl border px-4 py-3 text-sm shadow-2xl backdrop-blur-md",
        wrap
      )}
    >
      {icon}
      <span className="min-w-0 flex-1 break-words leading-snug">{toast.message}</span>
      <button
        type="button"
        onClick={onClose}
        aria-label="Dismiss"
        className="shrink-0 opacity-60 transition-opacity hover:opacity-100"
      >
        <X className="size-4" />
      </button>
    </div>
  )
}
