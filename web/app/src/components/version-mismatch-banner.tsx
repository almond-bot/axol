import { RefreshCw } from "lucide-react"
import type { VersionMismatch } from "@/lib/version"
import { Button } from "@/components/ui/button"

function label(commit: string, version: string | null): string {
  // Shorten only the SHA, keeping any -dirty.<hash> suffix — two identities
  // sharing HEAD but differing in working-tree state must render differently.
  const [sha, ...suffix] = commit.split("-")
  const short = [sha.slice(0, 7), ...suffix].join("-")
  return version ? `v${version} (${short})` : short
}

/**
 * Banner shown in the update-banner spot when this UI and the backend are on
 * different code (see lib/version.ts). Warning only — nothing is blocked,
 * since a mismatched panel must still be able to stop a running operation.
 */
export function VersionMismatchBanner({ mismatch }: { mismatch: VersionMismatch }) {
  const reloadFixes = !mismatch.local && !mismatch.serverDev && !mismatch.serverOlder
  const hint = mismatch.local
    ? "This computer's copy of the UI is out of date. Run `npm run build` in web/, then reload."
    : mismatch.serverDev
      ? "The robot runs a development build from a different commit. Open the UI from the robot's address instead."
      : mismatch.serverOlder
        ? "The robot software is older than this UI, so some features may not work. Update the robot software."
        : "Reload the page to get the UI that matches the robot software."
  return (
    <div className="flex flex-wrap items-center justify-between gap-3 rounded-lg border border-amber-400/25 bg-amber-400/[0.05] p-3 text-xs text-amber-200/80">
      <div className="flex min-w-0 flex-col gap-0.5">
        <span className="font-medium text-amber-200">
          This UI doesn&apos;t match the robot software
        </span>
        <span className="font-mono text-amber-200/60">
          UI {label(mismatch.uiCommit, mismatch.uiVersion)} &middot; robot{" "}
          {label(mismatch.serverCommit, mismatch.serverVersion)}
        </span>
        <span className="text-amber-200/60">{hint}</span>
      </div>
      {reloadFixes && (
        <Button variant="outline" size="sm" onClick={() => window.location.reload()}>
          <RefreshCw />
          Reload
        </Button>
      )}
    </div>
  )
}
