import type { UpdateChannel, UpdatePhase, UpdateStatus } from "@/lib/supervisor"

const REPO_URL = "https://github.com/almond-bot/axol"
const RELEASES_URL = `${REPO_URL}/releases`

/** The channel's name as the panel shows it. */
export const CHANNEL_LABEL: Record<UpdateChannel, string> = {
  release: "Release",
  main: "Main",
}

/** Human-readable label for the current update step (shown while updating). */
export const PHASE_LABEL: Record<UpdatePhase, string> = {
  upgrading: "Upgrading…",
  provisioning: "Installing packages…",
  restarting: "Restarting…",
  rebooting: "Rebooting host…",
}

/** A commit shortened to its first seven characters. */
export function shortCommit(commit: string | null | undefined): string {
  return commit ? commit.slice(0, 7) : "unknown"
}

/** Release version with the conventional "v" prefix, e.g. "v0.1.2". */
export function releaseTag(version: string | null | undefined): string {
  return version ? `v${version}` : "unknown"
}

/** What the host runs now, as the start of a jump line: "v0.2.44" or "main 1111111". */
function installedLabel(update: UpdateStatus): string {
  return update.channel === "main"
    ? `main ${shortCommit(update.commit)}`
    : releaseTag(update.version)
}

/**
 * What moving the host to `to` installs: a "from → to" line and a link to
 * what changes. `to` is the installed channel for an update within it, or the
 * other channel for a switch.
 */
export function describeChange(
  update: UpdateStatus,
  to: UpdateChannel
): { jump: string; link: string; linkLabel: string } {
  const from = update.channel ?? "release"
  const { commit, remoteCommit, version, remoteVersion } = update
  if (to === "release") {
    return {
      jump: `${installedLabel(update)} → ${releaseTag(remoteVersion)}`,
      link: RELEASES_URL,
      linkLabel: "Release notes",
    }
  }
  if (from === "main") {
    return {
      jump: `main ${shortCommit(commit)} → ${shortCommit(remoteCommit)}`,
      link:
        commit && remoteCommit
          ? `${REPO_URL}/compare/${commit}...${remoteCommit}`
          : `${REPO_URL}/commits/main`,
      linkLabel: "View changes",
    }
  }
  return {
    jump: `${releaseTag(version)} → main${remoteCommit ? ` ${shortCommit(remoteCommit)}` : ""}`,
    link: version ? `${REPO_URL}/compare/v${version}...main` : `${REPO_URL}/commits/main`,
    linkLabel: "View changes",
  }
}
