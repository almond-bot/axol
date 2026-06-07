import App from "../App"

/**
 * The immersive WebXR experience. Wrapped in a fixed, full-screen container so
 * the R3F canvas fills the viewport (the page itself is a normal scrolling
 * document for the control panel route).
 */
export default function VrApp() {
  return (
    <div style={{ position: "fixed", inset: 0 }}>
      <App />
    </div>
  )
}
