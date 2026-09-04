const POSE_SEQUENCE_RESERVED_KEY = "axol.webxr.pose-sequence-reserved.v2"

// Reserve enough counters for well over two hours at 120 Hz. Each page skips
// to an independently randomized block before reserving it: sessionStorage is
// cloned when a browser tab is duplicated, so merely taking the next block can
// otherwise give both pages the same sequence range and let their poses mix.
const POSE_SEQUENCE_BLOCK_SIZE = 1_000_000
const POSE_SEQUENCE_RANDOM_BLOCKS = 1 << 24

export type PoseSequence = {
  current: number
  reservedThrough: number
}

type SequenceStorage = Pick<Storage, "getItem" | "setItem">

type PoseSequenceRuntime = {
  storage?: SequenceStorage | null
  nowMs?: () => number
  randomUint32?: () => number
}

function defaultStorage(): SequenceStorage | null {
  try {
    return globalThis.sessionStorage ?? null
  } catch {
    return null
  }
}

function defaultRandomUint32(): number {
  const values = new Uint32Array(1)
  try {
    const crypto = globalThis.crypto
    if (crypto?.getRandomValues) {
      crypto.getRandomValues(values)
      return values[0]!
    }
  } catch {
    // WebXR runs in a secure context and normally has crypto. Retain a
    // best-effort fallback for tests and unusually restricted browsers.
  }
  return Math.floor(Math.random() * 0x1_0000_0000)
}

function runtimeStorage(runtime: PoseSequenceRuntime): SequenceStorage | null {
  return runtime.storage === undefined ? defaultStorage() : runtime.storage
}

function reserveRandomBlock(
  after: number,
  runtime: PoseSequenceRuntime,
  storage: SequenceStorage | null
): PoseSequence {
  // Leave room for both the randomized skip and the block itself. A corrupted
  // near-MAX_SAFE_INTEGER reservation is rebased to the wall clock; the server
  // already recovers a lower range after an older transport goes stale.
  const maxSkip = Math.floor(
    (Number.MAX_SAFE_INTEGER - POSE_SEQUENCE_BLOCK_SIZE - after) / POSE_SEQUENCE_BLOCK_SIZE
  )
  if (maxSkip < 1) after = Math.floor((runtime.nowMs ?? Date.now)() * 1000)

  const availableSkips = Math.max(
    1,
    Math.min(
      POSE_SEQUENCE_RANDOM_BLOCKS,
      Math.floor(
        (Number.MAX_SAFE_INTEGER - POSE_SEQUENCE_BLOCK_SIZE - after) / POSE_SEQUENCE_BLOCK_SIZE
      )
    )
  )
  const random = (runtime.randomUint32 ?? defaultRandomUint32)() >>> 0
  const current = after + (1 + (random % availableSkips)) * POSE_SEQUENCE_BLOCK_SIZE
  const reservedThrough = current + POSE_SEQUENCE_BLOCK_SIZE
  try {
    storage?.setItem(POSE_SEQUENCE_RESERVED_KEY, String(reservedThrough))
  } catch {
    // If storage is unavailable the source id is in-memory too, so no later
    // page can present the same producer id with a copied counter reservation.
  }
  return { current, reservedThrough }
}

/** Allocate a reload-safe, duplicated-tab-resistant counter range. */
export function initialPoseSequence(runtime: PoseSequenceRuntime = {}): PoseSequence {
  const wallClockBase = Math.floor((runtime.nowMs ?? Date.now)() * 1000)
  const storage = runtimeStorage(runtime)
  let priorReservation = 0
  try {
    const stored = Number(storage?.getItem(POSE_SEQUENCE_RESERVED_KEY))
    if (Number.isSafeInteger(stored) && stored >= 0) priorReservation = stored
  } catch {
    // Best-effort persistence only.
  }
  return reserveRandomBlock(Math.max(wallClockBase, priorReservation), runtime, storage)
}

export function nextPoseSequence(
  sequence: PoseSequence,
  runtime: PoseSequenceRuntime = {}
): number {
  if (sequence.current >= sequence.reservedThrough) {
    Object.assign(sequence, reserveRandomBlock(sequence.current, runtime, runtimeStorage(runtime)))
  }
  sequence.current += 1
  return sequence.current
}
