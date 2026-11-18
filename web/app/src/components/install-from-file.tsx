import { ExternalLink, FileUp, Loader2, Upload } from "lucide-react"
import { useEffect, useRef, useState, type ReactNode } from "react"
import {
  ApiRequestError,
  BUNDLE_CHUNK_BYTES,
  beginBundleUpload,
  checkBundleUpload,
  discardBundle,
  fetchUpdateStatus,
  recheckBundle,
  uploadBundleChunk,
  type BundleStatus,
  type UpdateStatus,
} from "@/lib/supervisor"
import { releaseTag } from "@/lib/update-view"
import { Button, buttonVariants } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { cn } from "@/lib/utils"

const RELEASES_URL = "https://github.com/almond-bot/axol/releases"
// How often one chunk is retried after the connection drops, before giving up.
const MAX_RETRIES = 5

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms))

/** "360 of 800 MB" */
function megabytes(sent: number, size: number): string {
  return `${Math.floor(sent / 1e6)} of ${Math.ceil(size / 1e6)} MB`
}

function percent(sent: number, size: number): number {
  return size > 0 ? Math.min(100, Math.floor((sent / size) * 100)) : 0
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e)
}

/**
 * Installs a release from its offline bundle, for a Host Computer without
 * internet access. The operator picks the bundle's .tar; it is uploaded in
 * chunks (an interrupted upload resumes when the same file is chosen again),
 * the host checks the Almond signature and every file, and Install update
 * hands it to the same install and restart as the update banner.
 */
export function InstallFromFile({
  update,
  updating,
  opRunning,
  hostInstalling,
  onStatus,
  onInstall,
  chunkBytes = BUNDLE_CHUNK_BYTES,
  retryDelayMs = 2000,
}: {
  update: UpdateStatus
  /** An update is installing. */
  updating: boolean
  /** An operation or session is running, so the host can't restart. */
  opRunning: boolean
  /** The host is installing packages, so the host can't restart. */
  hostInstalling: boolean
  /** Hands a fresh update status (with the bundle's state) to the page. */
  onStatus: (status: UpdateStatus) => void
  /** Installs the checked bundle. */
  onInstall: () => void
  /** Bytes per upload request (tests use small files). */
  chunkBytes?: number
  /** Wait before retrying a dropped chunk, times the attempt number. */
  retryDelayMs?: number
}) {
  const [open, setOpen] = useState(false)
  const [progress, setProgress] = useState<{ sent: number; size: number } | null>(null)
  const [error, setError] = useState<string | null>(null)
  const [working, setWorking] = useState(false)
  const cancelled = useRef(false)
  const input = useRef<HTMLInputElement>(null)
  const bundle: BundleStatus | undefined = update.bundle
  const serverState = bundle?.state ?? "idle"

  async function refresh(): Promise<BundleStatus | undefined> {
    const status = await fetchUpdateStatus()
    onStatus(status)
    return status.bundle
  }

  // While the host checks a file, follow it until it is ready or refused.
  useEffect(() => {
    if (!open || serverState !== "checking") return
    let active = true
    const timer = setInterval(() => {
      fetchUpdateStatus()
        .then((status) => {
          if (active) onStatus(status)
        })
        .catch(() => {})
    }, 1000)
    return () => {
      active = false
      clearInterval(timer)
    }
  }, [open, serverState, onStatus])

  async function upload(file: File) {
    cancelled.current = false
    setError(null)
    setWorking(true)
    setProgress({ sent: 0, size: file.size })
    try {
      let status = await beginBundleUpload(file.name, file.size)
      const uploadId = status.uploadId
      if (!uploadId) throw new Error("The Host Computer did not start the upload. Try again.")
      let offset = status.received
      let failures = 0
      while (offset < file.size) {
        if (cancelled.current) return
        setProgress({ sent: offset, size: file.size })
        try {
          status = await uploadBundleChunk(
            uploadId,
            offset,
            file.slice(offset, offset + chunkBytes)
          )
          offset = status.received
          failures = 0
        } catch (e) {
          // The host refused the upload: retrying would not help.
          if (e instanceof ApiRequestError) throw e
          failures += 1
          if (failures > MAX_RETRIES)
            throw new Error(
              "The connection to the Host Computer dropped. Check the network, then choose the same file to continue the upload.",
              { cause: e }
            )
          await sleep(retryDelayMs * failures)
          // Continue from what the host actually stored.
          status = await beginBundleUpload(file.name, file.size).catch(() => status)
          offset = status.received
        }
      }
      setProgress({ sent: file.size, size: file.size })
      await checkBundleUpload(uploadId)
      await refresh()
    } catch (e) {
      if (!cancelled.current) setError(errorText(e))
      await refresh().catch(() => {})
    } finally {
      setProgress(null)
      setWorking(false)
    }
  }

  async function act(action: () => Promise<unknown>) {
    setError(null)
    setWorking(true)
    try {
      await action()
      await refresh()
    } catch (e) {
      setError(errorText(e))
    } finally {
      setWorking(false)
    }
  }

  async function cancelUpload() {
    cancelled.current = true
    await act(discardBundle)
  }

  const uploading = progress !== null
  const blocker = opRunning
    ? "An operation is running. Stop it first."
    : hostInstalling
      ? "The robot is installing updates. Wait for it to finish."
      : null

  const chooser = (label: string) => (
    <Button size="sm" disabled={working || updating} onClick={() => input.current?.click()}>
      <FileUp />
      {label}
    </Button>
  )

  let body: ReactNode
  let actions: ReactNode
  if (uploading) {
    const pct = percent(progress.sent, progress.size)
    body = (
      <div className="flex flex-col gap-2">
        <p className="text-sm text-white/70">Uploading… {pct}%</p>
        <div className="h-1.5 overflow-hidden rounded bg-white/10">
          <div className="h-full bg-white/60" style={{ width: `${pct}%` }} />
        </div>
        <p className="font-mono text-xs text-white/45">{megabytes(progress.sent, progress.size)}</p>
      </div>
    )
    actions = (
      <Button variant="ghost" size="sm" onClick={() => void cancelUpload()}>
        Cancel upload
      </Button>
    )
  } else if (serverState === "checking") {
    body = (
      <p className="flex items-center gap-2 text-sm text-white/70">
        <Loader2 className="size-4 animate-spin" />
        Checking the file…
      </p>
    )
    actions = null
  } else if (serverState === "ready" && bundle?.version) {
    body = (
      <>
        <p className="font-mono text-xs text-white/70">
          {releaseTag(update.version)} → {releaseTag(bundle.version)}
        </p>
        <p className="text-sm leading-relaxed whitespace-normal text-white/45">
          The file is ready to install. The Host Computer installs it and restarts, which can take
          several minutes.
        </p>
        {blocker && <p className="text-xs whitespace-normal text-amber-200/70">{blocker}</p>}
      </>
    )
    actions = (
      <>
        <Button
          variant="ghost"
          size="sm"
          disabled={working || updating}
          onClick={() => void act(discardBundle)}
        >
          Remove file
        </Button>
        <Button
          size="sm"
          disabled={blocker !== null || working || updating}
          onClick={() => {
            onInstall()
            setOpen(false)
          }}
        >
          <Upload />
          Install update
        </Button>
      </>
    )
  } else if (serverState === "error" && bundle?.error) {
    const older = bundle.errorCode === "older"
    body = (
      <p className="text-sm whitespace-normal text-amber-200/80">
        {bundle.error}
        {older && " Install it only to go back to that version."}
      </p>
    )
    actions = older ? (
      <>
        {chooser("Choose another file")}
        <Button
          variant="ghost"
          size="sm"
          disabled={working || updating}
          onClick={() => void act(() => recheckBundle(true))}
        >
          Use older version
        </Button>
      </>
    ) : (
      chooser("Choose another file")
    )
  } else {
    const paused = serverState === "uploading" && bundle?.name
    body = (
      <p className="text-sm leading-relaxed whitespace-normal text-white/45">
        {paused
          ? `The upload of ${bundle.name} stopped at ${percent(bundle.received, bundle.size)}%. Choose the same file to continue it.`
          : "Install a release from its offline bundle. On a computer with internet access, download axol-offline-v<version>-linux-aarch64.tar from the release page, then choose it here."}
      </p>
    )
    actions = chooser("Choose file")
  }

  return (
    <>
      <Button
        variant="ghost"
        size="sm"
        disabled={updating}
        title={updating ? "An update is installing." : undefined}
        onClick={() => setOpen(true)}
        className="h-7 px-2 text-xs text-white/60"
      >
        <FileUp />
        Install from file
      </Button>

      <input
        ref={input}
        type="file"
        accept=".tar,application/x-tar"
        className="hidden"
        aria-label="Offline bundle file"
        onChange={(e) => {
          const file = e.target.files?.[0]
          e.target.value = ""
          if (file) void upload(file)
        }}
      />

      {open && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 p-4"
          onClick={() => {
            if (!uploading) setOpen(false)
          }}
        >
          <Card
            className="w-full max-w-sm gap-4 bg-[#1a1a1a] p-5 text-left"
            onClick={(e) => e.stopPropagation()}
          >
            <h3 className="font-heading text-base font-semibold">Install from file</h3>
            {body}
            {error && <p className="text-xs whitespace-normal text-red-300/80">{error}</p>}
            <div className="flex flex-wrap items-center gap-2 pt-1">
              <a
                href={RELEASES_URL}
                target="_blank"
                rel="noreferrer"
                className={cn(buttonVariants({ variant: "ghost", size: "sm" }), "text-white/60")}
              >
                Releases
                <ExternalLink />
              </a>
              <span className="ml-auto" />
              {!uploading && (
                <Button variant="ghost" size="sm" onClick={() => setOpen(false)}>
                  Close
                </Button>
              )}
              {actions}
            </div>
          </Card>
        </div>
      )}
    </>
  )
}
