import { lazy, StrictMode, Suspense } from "react"
import { createRoot } from "react-dom/client"
import { BrowserRouter, Navigate, Route, Routes } from "react-router-dom"
import { Loader2 } from "lucide-react"
// Self-host the UI fonts (Inter + Geist) so first paint and in-headset text
// never reach out to Google Fonts — the app has to run on an offline LAN.
import "@fontsource-variable/inter/index.css"
import "@fontsource-variable/geist/index.css"
import "./index.css"
import { isHeadsetBrowser } from "./lib/headset"
import { ToastProvider } from "./components/ui/toast"

// Lazy-loaded so the control panel route doesn't pull in the heavy
// three.js / @react-three/xr bundle (and vice versa). Without this, opening
// /control still downloads + parses several MB of 3D/XR code on first paint,
// which shows as a black screen until it finishes.
const VrApp = lazy(() => import("./routes/VrApp"))
const ControlPanel = lazy(() => import("./routes/ControlPanel"))

/** Send headsets to the immersive app, everything else to the control panel. */
function RootRedirect() {
  return <Navigate to={isHeadsetBrowser() ? "/vr" : "/control"} replace />
}

function RouteFallback() {
  return (
    <div className="flex min-h-screen items-center justify-center">
      <Loader2 className="size-6 animate-spin text-white/30" />
    </div>
  )
}

createRoot(document.getElementById("root")!).render(
  <StrictMode>
    <BrowserRouter>
      <ToastProvider>
        <Suspense fallback={<RouteFallback />}>
          <Routes>
            <Route path="/" element={<RootRedirect />} />
            <Route path="/vr" element={<VrApp />} />
            <Route path="/control" element={<ControlPanel />} />
            <Route path="*" element={<Navigate to="/" replace />} />
          </Routes>
        </Suspense>
      </ToastProvider>
    </BrowserRouter>
  </StrictMode>
)
