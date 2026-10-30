import { Bot, Forklift, Joystick, Server, type LucideIcon } from "lucide-react"

/** One icon per device, the same on Devices and Diagnostics: Mantis is a
 *  handheld teleop and data collection device, Jelly a powered base with a lift. */
export const DEVICE_ICONS: Record<"host" | "axol" | "mantis" | "jelly", LucideIcon> = {
  host: Server,
  axol: Bot,
  mantis: Joystick,
  jelly: Forklift,
}
