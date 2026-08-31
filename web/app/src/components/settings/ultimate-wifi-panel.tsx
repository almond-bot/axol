import { useEffect, useState } from "react"
import { Eye, EyeOff, Loader2, Save, Wifi } from "lucide-react"
import {
  fetchUltimateWifiConfig,
  saveUltimateWifiConfig,
  type UltimateWifiConfig,
} from "@/lib/supervisor"
import { Button } from "@/components/ui/button"
import { Input } from "@/components/ui/input"
import { Label } from "@/components/ui/label"
import { useToast } from "@/components/ui/toast"

function message(error: unknown): string {
  return String(error).replace(/^Error:\s*/, "")
}

/** Redacted editor for pyvut's private two-tracker shared-map network. */
export function UltimateWifiPanel() {
  const toast = useToast()
  const [config, setConfig] = useState<UltimateWifiConfig | null>(null)
  const [ssid, setSsid] = useState("")
  const [country, setCountry] = useState("US")
  const [frequency, setFrequency] = useState("5240")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(true)
  const [saving, setSaving] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    let cancelled = false
    fetchUltimateWifiConfig()
      .then((found) => {
        if (cancelled) return
        setConfig(found)
        setSsid(found.ssid)
        setCountry(found.country || "US")
        setFrequency(found.freq > 0 ? String(found.freq) : "5240")
        // The API never returns a password. Keep the editor blank even when
        // one is already saved, so it cannot leak into browser state.
        setPassword("")
        setError(null)
      })
      .catch((reason) => {
        if (!cancelled) setError(message(reason))
      })
      .finally(() => {
        if (!cancelled) setLoading(false)
      })
    return () => {
      cancelled = true
    }
  }, [])

  const parsedFrequency = Number(frequency)
  const valid =
    ssid.length > 0 &&
    /^[A-Za-z]{2}$/.test(country) &&
    Number.isInteger(parsedFrequency) &&
    parsedFrequency > 0 &&
    (config?.passwordSet || password.length > 0)

  async function save() {
    if (!valid) return
    setSaving(true)
    setError(null)
    try {
      const update = {
        ssid,
        country: country.toUpperCase(),
        freq: parsedFrequency,
        ...(password.length > 0 ? { pass: password } : {}),
      }
      const saved = await saveUltimateWifiConfig(update)
      setConfig(saved)
      setCountry(saved.country)
      setFrequency(String(saved.freq))
      // Blank now means preserve the host-side secret on later edits.
      setPassword("")
      setShowPassword(false)
      toast.success("Ultimate shared-map Wi-Fi saved securely on the host.")
    } catch (reason) {
      setError(message(reason))
      toast.error(`Could not save Ultimate Wi-Fi: ${message(reason)}`)
    } finally {
      setSaving(false)
    }
  }

  const statusLabel =
    config?.status === "valid"
      ? "Configured"
      : config?.status === "permissions-warning"
        ? "Permissions need repair"
        : config?.status === "invalid"
          ? "Invalid"
          : config?.status === "missing"
            ? "Not configured"
            : "Checking…"
  const statusClass =
    config?.status === "valid"
      ? "bg-emerald-400/10 text-emerald-300"
      : config?.status === "invalid"
        ? "bg-red-400/10 text-red-300"
        : "bg-amber-400/10 text-amber-300"

  return (
    <section className="flex flex-col gap-4 rounded-lg border border-white/10 bg-white/[0.02] p-4">
      <div className="flex flex-wrap items-center gap-2">
        <Wifi className="size-4 text-white/50" />
        <span className="text-sm font-medium text-white/80">Ultimate shared-map Wi-Fi</span>
        <span className={`rounded-full px-2 py-0.5 text-[11px] ${statusClass}`}>{statusLabel}</span>
      </div>
      <p className="max-w-prose text-xs leading-relaxed text-white/45">
        This is pyvut&apos;s protected shared-map AP configuration/fallback for synchronizing two
        trackers, not your router Wi-Fi. Dongle firmware may supply the active host credentials. The
        saved password is write-only and is never returned to this browser.
      </p>

      {loading ? (
        <p className="flex items-center gap-2 text-xs text-white/45">
          <Loader2 className="size-4 animate-spin" /> Loading host configuration…
        </p>
      ) : (
        <div className="grid gap-4 sm:grid-cols-2">
          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <Label htmlFor="ultimate-wifi-ssid">Private network name (SSID)</Label>
            <Input
              id="ultimate-wifi-ssid"
              value={ssid}
              onChange={(event) => setSsid(event.target.value)}
              autoComplete="off"
              spellCheck={false}
              placeholder="AXOL_TRACKERS"
            />
          </div>
          <div className="flex flex-col gap-1.5 sm:col-span-2">
            <Label htmlFor="ultimate-wifi-password">
              Password {config?.passwordSet ? "(leave blank to preserve)" : "(required)"}
            </Label>
            <div className="relative">
              <Input
                id="ultimate-wifi-password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                autoComplete="new-password"
                spellCheck={false}
                placeholder={
                  config?.passwordSet ? "Saved password remains unchanged" : "Choose a password"
                }
                className="pr-10"
              />
              <button
                type="button"
                onClick={() => setShowPassword((shown) => !shown)}
                className="absolute inset-y-0 right-0 flex w-10 items-center justify-center text-white/35 hover:text-white/70"
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? <EyeOff className="size-4" /> : <Eye className="size-4" />}
              </button>
            </div>
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ultimate-wifi-country">Country code</Label>
            <Input
              id="ultimate-wifi-country"
              value={country}
              onChange={(event) => setCountry(event.target.value.toUpperCase().slice(0, 2))}
              autoComplete="off"
              spellCheck={false}
              maxLength={2}
              placeholder="US"
            />
          </div>
          <div className="flex flex-col gap-1.5">
            <Label htmlFor="ultimate-wifi-frequency">Center frequency (MHz)</Label>
            <Input
              id="ultimate-wifi-frequency"
              type="number"
              min={1}
              step={1}
              value={frequency}
              onChange={(event) => setFrequency(event.target.value)}
              inputMode="numeric"
              placeholder="5240"
            />
          </div>
        </div>
      )}

      {(error || config?.error) && (
        <p className="text-xs leading-relaxed text-red-300/80">{error ?? config?.error}</p>
      )}
      {!loading && !valid && (
        <p className="text-[11px] leading-relaxed text-amber-300/75">
          Enter an SSID, a two-letter country code, a positive whole-number frequency, and a
          password on the first save.
        </p>
      )}
      {!loading && (
        <div className="flex flex-wrap items-center gap-3">
          <Button type="button" size="sm" onClick={save} disabled={!valid || saving}>
            {saving ? <Loader2 className="animate-spin" /> : <Save />}
            Save private Wi-Fi
          </Button>
          {config?.configured && (
            <span className="text-[11px] text-white/30">
              Password {config.passwordSet ? "saved" : "missing"} ·{" "}
              {config.status === "valid" ? "protected host file" : "host file needs attention"}
            </span>
          )}
        </div>
      )}
    </section>
  )
}
