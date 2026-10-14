// Minimum gap between two firings of the both-sticks-clicked gesture (ms).
// Longer than a reflexive double click, shorter than a deliberate repeat.
export const BOTH_CLICK_DEBOUNCE_MS = 600
// The two stick presses must land within this window of each other to count
// as "clicked together": outside box mode a single click is Jelly's lift
// (held), so a stick that has been held for a while when the other one is
// pressed is the lift, not the gesture.
export const BOTH_CLICK_TOGETHER_MS = 350

/**
 * Box-mode toggle gesture state: each stick's click on the previous frame and
 * when it was last pressed (to require the two presses to land together),
 * both clicked on the previous frame (edge detection), and when the gesture
 * last fired: a second rising edge inside the debounce — a "double click", or
 * one stick's contact bouncing while the other is held — is ignored so the
 * mode can't toggle twice and land where it started.
 */
export type BothStickClickState = {
  prevL: boolean
  prevR: boolean
  lAt: number
  rAt: number
  prevBoth: boolean
  lastFiredAt: number
}

export function initialBothStickClickState(): BothStickClickState {
  return {
    prevL: false,
    prevR: false,
    lAt: 0,
    rAt: 0,
    prevBoth: false,
    lastFiredAt: -Infinity,
  }
}

/**
 * Advance the gesture by one frame (`now` in ms). Returns true on the frame
 * the gesture fires: the rising edge of "both down" when the two presses
 * landed within `BOTH_CLICK_TOGETHER_MS` of each other, at most once per
 * `BOTH_CLICK_DEBOUNCE_MS`. Mutates `state`.
 */
export function stepBothStickClick(
  state: BothStickClickState,
  lClick: boolean,
  rClick: boolean,
  now: number
): boolean {
  if (lClick && !state.prevL) state.lAt = now
  if (rClick && !state.prevR) state.rAt = now
  state.prevL = lClick
  state.prevR = rClick
  const both = lClick && rClick
  let fired = false
  if (both && !state.prevBoth) {
    const together = Math.abs(state.lAt - state.rAt) <= BOTH_CLICK_TOGETHER_MS
    if (together && now - state.lastFiredAt >= BOTH_CLICK_DEBOUNCE_MS) {
      state.lastFiredAt = now
      fired = true
    }
  }
  state.prevBoth = both
  return fired
}
