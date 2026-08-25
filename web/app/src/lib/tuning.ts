import { apiUrl } from "@/lib/supervisor"

/**
 * Client for the tuning-run store (`/api/tuning/*`): the persisted artifacts
 * every tuning suite saves — sine/step PID probes, reference-motion replays,
 * and the offline wifi/filtering/kinematics analyses. Each run carries its
 * metrics scorecard plus decimated time series for charting.
 */

export interface TuningRunMeta {
  id: string
  /** Suite name: "sine" | "step" | "motion" | "wifi" | "filtering" | "kinematics". */
  kind: string
  side: string | null
  joint: string | null
  /** Gain overrides the run was made with (kp/kd/kd_host/…). */
  gains: Record<string, number>
  /** Reproduction parameters (amplitude, frequency, motion name, …). */
  params: Record<string, unknown>
  /** The run's scorecard; NaN metrics arrive as null. */
  metrics: Record<string, unknown>
  label: string | null
  /** Shared id linking the runs of one sweep. */
  group: string | null
  startedAt: number
  seriesKeys: string[]
  samples: number
}

export interface TuningRunData {
  meta: TuningRunMeta
  /**
   * Decimated series, keyed by name. A multi-column source array (e.g. a
   * motion run's N×14 joint matrix) arrives as one key per column,
   * `"<name>/<index>"` — column names live in `meta.params` ("columns").
   */
  series: Record<string, (number | null)[]>
}

export interface TuningMotion {
  name: string
  rate: number
  samples: number
  durationS: number
  meta: Record<string, unknown>
}

async function json<T>(res: Response): Promise<T> {
  const body = await res.json().catch(() => ({}))
  if (!res.ok) {
    throw new Error((body as { error?: string }).error ?? `HTTP ${res.status}`)
  }
  return body as T
}

export async function fetchTuningRuns(): Promise<{ runs: TuningRunMeta[] }> {
  return json(await fetch(apiUrl("/api/tuning/runs")))
}

export async function fetchTuningRun(
  id: string,
  maxPoints = 4000
): Promise<TuningRunData> {
  return json(
    await fetch(apiUrl(`/api/tuning/runs/${id}?max_points=${maxPoints}`))
  )
}

export async function deleteTuningRun(id: string): Promise<{ deleted: string }> {
  return json(await fetch(apiUrl(`/api/tuning/runs/${id}`), { method: "DELETE" }))
}

export async function clearTuningRuns(): Promise<{ removed: number }> {
  return json(await fetch(apiUrl("/api/tuning/runs"), { method: "DELETE" }))
}

export async function fetchTuningMotions(): Promise<{ motions: TuningMotion[] }> {
  return json(await fetch(apiUrl("/api/tuning/motions")))
}
