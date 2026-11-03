import posthog from "posthog-js"
import type { ServerInfo, SessionStatus } from "@/lib/supervisor"
import type {
  CollectionMethod,
  DataCollectionOperation,
  DataCollectionPrice,
  DataCollectionRequest,
  DataCollectionService,
} from "@/lib/data-collection"
import type { VersionMismatch } from "@/lib/version"

/**
 * Product analytics, error capture and session replay for the control panel,
 * through PostHog, in the same project as almond.bot.
 *
 * The panel has no login, so the browser stays an anonymous person and the
 * robot it drives is a PostHog group (`robot`, keyed by the serve host's
 * hostname). Events are explicit and few; autocapture is off. Replay masks
 * every input, and camera canvases are never recorded.
 *
 * Nothing here runs without `VITE_POSTHOG_KEY` at build time, or in a dev
 * build, so sim sessions and local checkouts send nothing. Every call is a
 * no-op when analytics are off, so callers never guard.
 */

export const DEFAULT_ANALYTICS_HOST = "https://us.i.posthog.com"

export interface AnalyticsEnv {
  VITE_POSTHOG_KEY?: string
  VITE_POSTHOG_HOST?: string
  DEV?: boolean
}

/** The PostHog project to send to, or null when analytics stay off. */
export function analyticsConfig(
  env: AnalyticsEnv = import.meta.env
): { key: string; host: string } | null {
  if (env.DEV) return null
  const key = env.VITE_POSTHOG_KEY?.trim()
  if (!key) return null
  return { key, host: env.VITE_POSTHOG_HOST?.trim() || DEFAULT_ANALYTICS_HOST }
}

let enabled = false

/** Starts PostHog once, at app boot. Returns whether analytics are on. */
export function initAnalytics(env?: AnalyticsEnv): boolean {
  const config = analyticsConfig(env)
  if (!config) return false
  try {
    posthog.init(config.key, {
      api_host: config.host,
      // The panel is a single-page app: count every route change as a view.
      capture_pageview: "history_change",
      capture_pageleave: true,
      autocapture: false,
      capture_exceptions: true,
      person_profiles: "identified_only",
      session_recording: { maskAllInputs: true },
    })
    enabled = true
  } catch {
    enabled = false
  }
  return enabled
}

export function analyticsEnabled(): boolean {
  return enabled
}

/** Every event the panel sends, with its properties. */
export interface AnalyticsEvents {
  /** The panel reached a serve host. */
  host_connected: {
    host_version: string | null
    host_commit: string | null
    release_install: boolean | null
  }
  host_connect_failed: { error: string }
  robot_connected: {
    device: "axol" | "mantis"
    automatic: boolean
    motor_count: number
    reachable_count: number
  }
  robot_connect_failed: { device: "axol" | "mantis"; automatic: boolean; error: string }
  robot_disconnected: Record<string, never>
  operation_started: { operation: string; sim: boolean; device: "axol" | "mantis" }
  operation_start_failed: { operation: string; error: string }
  operation_ended: {
    operation: string
    status: SessionStatus
    exit_code: number | null
    duration_s: number | null
    error: string | null
    sim: boolean
  }
  /** The customer left for Stripe's checkout page. Shaped like the site's. */
  checkout_started: CheckoutProperties
  checkout_returned: { outcome: "paid" | "cancelled" }
  version_mismatch_shown: {
    local: boolean
    server_dev: boolean
    server_older: boolean
    ui_version: string | null
    server_version: string | null
  }
  update_started: { from_version: string | null; to_version: string }
}

export function track<E extends keyof AnalyticsEvents>(
  event: E,
  properties: AnalyticsEvents[E]
): void {
  if (!enabled) return
  try {
    posthog.capture(event, properties)
  } catch {
    // Analytics never break the panel.
  }
}

/** Files this browser's events under the robot it is driving. */
export function identifyRobot(info: ServerInfo): void {
  if (!enabled) return
  try {
    posthog.group("robot", info.hostname, {
      name: info.hostname,
      version: info.version ?? null,
      commit: info.commit ?? null,
      release_install: info.releaseInstall ?? null,
    })
  } catch {
    // Analytics never break the panel.
  }
}

/** Drops the robot group, when the panel moves to another host. */
export function forgetRobot(): void {
  if (!enabled) return
  try {
    posthog.resetGroups()
  } catch {
    // Analytics never break the panel.
  }
}

export interface CheckoutProperties {
  service: DataCollectionService
  collection_method: CollectionMethod | null
  hours: number
  operation: DataCollectionOperation
  /** Hours × the hourly price, in major units, when the price was known. */
  value: number | null
  currency: string | null
}

export function checkoutProperties(
  request: Pick<DataCollectionRequest, "service" | "method" | "hours" | "operation">,
  price: DataCollectionPrice | null
): CheckoutProperties {
  return {
    service: request.service,
    collection_method: request.method,
    hours: request.hours,
    operation: request.operation,
    value: price ? (price.unitAmount * request.hours) / 100 : null,
    currency: price ? price.currency.toUpperCase() : null,
  }
}

export function versionMismatchProperties(
  mismatch: VersionMismatch
): AnalyticsEvents["version_mismatch_shown"] {
  return {
    local: mismatch.local,
    server_dev: mismatch.serverDev,
    server_older: mismatch.serverOlder,
    ui_version: mismatch.uiVersion,
    server_version: mismatch.serverVersion,
  }
}
