/**
 * Split a hands-on prompt into its text and button label. A command names the
 * button by ending the prompt with "[button: Label]"; otherwise it's Continue.
 */
export function parsePrompt(prompt: string): { text: string; button: string } {
  const m = /\s*\[button: ([^\]]+)\]$/.exec(prompt)
  return m
    ? { text: prompt.slice(0, m.index).trim(), button: m[1].trim() }
    : { text: prompt, button: "Continue" }
}
