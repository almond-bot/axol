import {
  Suspense,
  useEffect,
  useRef,
  useState,
  type ComponentProps,
  type ReactNode,
  type RefObject,
} from "react"
import { Canvas, useFrame, useThree } from "@react-three/fiber"
import { Text } from "@react-three/drei"
import { createXRStore, XR, useXR } from "@react-three/xr"
import * as THREE from "three"
import {
  AxolConnectionStatus,
  AxolVRClient,
  AxolState,
  useAxolPoseSocket,
  useAxolTracking,
  useAxolVideo,
  useAxolVRClient,
} from "@almond/axol-vr-client"
import { Headset, Loader2, ShieldCheck } from "lucide-react"
import { configureTextBuilder } from "troika-three-text"
import interFontUrl from "@fontsource/inter/files/inter-latin-700-normal.woff"
import { Button } from "@/components/ui/button"
import { Card } from "@/components/ui/card"
import { Input } from "@/components/ui/input"
import { SiteNav } from "@/components/site-nav"
import { authorizeCert } from "@/lib/cert-accept"
import { cn } from "@/lib/utils"

// Pin drei's <Text> (troika) to a locally-bundled font. By default troika
// fetches its font from a jsdelivr CDN, which hangs forever on an offline LAN
// and — because every in-headset <Text> suspends until the font loads, and the
// whole XR scene (camera feed included) sits under one <Suspense> — leaves the
// headset showing nothing. Serving the font from the same origin makes VR
// teleop work with no internet. Must run before the first <Text> renders.
// Must be .woff (or .ttf/.otf) — troika's font parser can't decode .woff2
// (brotli), so a .woff2 fails with "woff2 fonts not supported" and no HUD
// text renders.
configureTextBuilder({ defaultFontURL: interFontUrl })

// Code-point ranges the bundled Inter .woff above actually contains — the
// "latin" subset from @fontsource/inter (its unicode.json). Any character
// outside this set isn't in our font, so troika hands it to its unicode-font
// fallback, which fetches descriptors + font files from a jsdelivr CDN *inside
// a web worker* (so a main-thread fetch override can't stop it, and pointing it
// at a bad URL just makes it retry the CDN). On an offline LAN that fetch never
// resolves and the <Text> hangs forever, blanking the whole XR scene. We avoid
// it entirely by never feeding <Text> a glyph we can't draw: see HudText.
const INTER_LATIN_RANGES: ReadonlyArray<readonly [number, number]> = [
  [0x0000, 0x00ff],
  [0x0131, 0x0131],
  [0x0152, 0x0153],
  [0x02bb, 0x02bc],
  [0x02c6, 0x02c6],
  [0x02da, 0x02da],
  [0x02dc, 0x02dc],
  [0x0304, 0x0304],
  [0x0308, 0x0308],
  [0x0329, 0x0329],
  [0x2000, 0x206f],
  [0x20ac, 0x20ac],
  [0x2122, 0x2122],
  [0x2191, 0x2191],
  [0x2193, 0x2193],
  [0x2212, 0x2212],
  [0x2215, 0x2215],
  [0xfeff, 0xfeff],
  [0xfffd, 0xfffd],
]

function interCovers(codePoint: number): boolean {
  for (const [lo, hi] of INTER_LATIN_RANGES) {
    if (codePoint >= lo && codePoint <= hi) return true
  }
  return false
}

// Replace any glyph the bundled font can't draw with U+FFFD ("□", itself in the
// latin subset) so unsupported text degrades to a visible placeholder instead
// of triggering the offline-hanging CDN fallback described above.
function sanitizeHudText(text: string): string {
  let out = ""
  for (const ch of text) {
    out += interCovers(ch.codePointAt(0)!) ? ch : "\uFFFD"
  }
  return out
}

// Drop-in replacement for drei's <Text> for anything rendered in-headset: it
// strips its string children to glyphs the bundled font can actually draw, so
// the troika unicode fallback (and its CDN fetch) can never fire. Use this
// instead of <Text> for HUD content, especially anything dynamic.
function HudText({ children, ...props }: ComponentProps<typeof Text>) {
  const safe = Array.isArray(children)
    ? children.map((c) => (typeof c === "string" ? sanitizeHudText(c) : c))
    : typeof children === "string"
      ? sanitizeHudText(children)
      : children
  return <Text {...props}>{safe}</Text>
}

// The VR teleop WebSocket server runs on this port (see useAxolVRClient default).
const VR_WS_PORT = 8000

const store = createXRStore({
  handTracking: false,
  bodyTracking: true,
  controller: { model: false },
  // Resolve controller (and hand) input profiles from our own origin instead
  // of the default jsdelivr CDN. @react-three/xr always fetches
  // profilesList.json + <profile>/profile.json to map the gamepad layout (even
  // with the 3D model disabled), which hangs forever on an offline LAN. The
  // descriptors are bundled into /webxr-profiles/ at build time (see
  // app/scripts/sync-webxr-profiles.mjs). Must be an absolute URL — the loader
  // resolves paths against it with `new URL(...)`, which rejects bare paths —
  // so anchor it to the current origin to stay same-origin on any host.
  baseAssetPath: new URL("/webxr-profiles/", window.location.href).href,
  // Desktop-only: @react-three/xr falls back to its bundled iwer emulator when
  // there's no native WebXR (i.e. on localhost in a normal browser). Its
  // synthetic "room" environment renders each frame with an older three.js API
  // (material.onBuild) that our three version removed, which throws every frame
  // ("onBuild is not a function") and blanks the scene. Disabling it keeps the
  // emulated device + controller panels working over a transparent background
  // (fine for this passthrough app). Ignored entirely on a real headset, where
  // native WebXR is used and the emulator never activates.
  emulate: { syntheticEnvironment: false },
})

const L_ELBOW_JOINT = "left-arm-lower" as XRBodyJoint
const R_ELBOW_JOINT = "right-arm-lower" as XRBodyJoint

const AXIS_LEN = 0.1
const SHAFT_R = 0.004
const TIP_R = 0.009
const TIP_LEN = 0.025
const DOT_RADIUS = 0.014

const AXES: { color: string; rotation: [number, number, number] }[] = [
  { color: "#FF0000", rotation: [0, 0, -Math.PI / 2] }, // X — red
  { color: "#00FF00", rotation: [0, 0, 0] }, // Y — green
  { color: "#0000FF", rotation: [Math.PI / 2, 0, 0] }, // Z — blue
]

function Arrow({ color, rotation }: { color: string; rotation: [number, number, number] }) {
  const shaftLen = AXIS_LEN - TIP_LEN
  return (
    <group rotation={rotation}>
      <mesh position={[0, shaftLen / 2, 0]}>
        <cylinderGeometry args={[SHAFT_R, SHAFT_R, shaftLen, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
      <mesh position={[0, shaftLen + TIP_LEN / 2, 0]}>
        <coneGeometry args={[TIP_R, TIP_LEN, 8]} />
        <meshBasicMaterial color={color} />
      </mesh>
    </group>
  )
}

function AxesMarker({ groupRef }: { groupRef: React.RefObject<THREE.Group | null> }) {
  return (
    <group ref={groupRef} visible={false}>
      {AXES.map((a) => (
        <Arrow key={a.color} color={a.color} rotation={a.rotation} />
      ))}
      <mesh>
        <sphereGeometry args={[DOT_RADIUS, 10, 10]} />
        <meshBasicMaterial color="#FFFF00" />
      </mesh>
    </group>
  )
}

function PoseVisualizer() {
  const { gl } = useThree()
  const leftRef = useRef<THREE.Group>(null)
  const rightRef = useRef<THREE.Group>(null)
  const lElbowRef = useRef<THREE.Group>(null)
  const rElbowRef = useRef<THREE.Group>(null)

  useFrame(() => {
    const session = gl.xr.getSession()
    if (!session) return
    const frame = gl.xr.getFrame()
    const refSpace = gl.xr.getReferenceSpace()
    if (!frame || !refSpace) return

    function applyPose(group: THREE.Group | null, space: XRSpace | null | undefined) {
      if (!group) return
      if (!space) {
        group.visible = false
        return
      }
      const pose = frame.getPose(space, refSpace!)
      if (!pose) {
        group.visible = false
        return
      }
      const { position: p, orientation: o } = pose.transform
      group.position.set(p.x, p.y, p.z)
      group.quaternion.set(o.x, o.y, o.z, o.w)
      group.visible = true
    }

    function applyPosition(group: THREE.Group | null, space: XRSpace | null | undefined) {
      if (!group) return
      if (!space) {
        group.visible = false
        return
      }
      const pose = frame.getPose(space, refSpace!)
      if (!pose) {
        group.visible = false
        return
      }
      const { position: p } = pose.transform
      group.position.set(p.x, p.y, p.z)
      group.visible = true
    }

    const leftSource = Array.from(session.inputSources).find(
      (s: XRInputSource) => s.handedness === "left"
    )
    const rightSource = Array.from(session.inputSources).find(
      (s: XRInputSource) => s.handedness === "right"
    )

    applyPose(leftRef.current, leftSource?.targetRaySpace ?? null)
    applyPose(rightRef.current, rightSource?.targetRaySpace ?? null)

    const body = (frame as XRFrame & { body?: XRBody }).body
    applyPosition(lElbowRef.current, body?.get(L_ELBOW_JOINT))
    applyPosition(rElbowRef.current, body?.get(R_ELBOW_JOINT))
  })

  return (
    <>
      <AxesMarker groupRef={leftRef} />
      <AxesMarker groupRef={rightRef} />
      <group ref={lElbowRef} visible={false}>
        <mesh>
          <sphereGeometry args={[DOT_RADIUS, 10, 10]} />
          <meshBasicMaterial color="#FFFF00" />
        </mesh>
      </group>
      <group ref={rElbowRef} visible={false}>
        <mesh>
          <sphereGeometry args={[DOT_RADIUS, 10, 10]} />
          <meshBasicMaterial color="#FFFF00" />
        </mesh>
      </group>
    </>
  )
}

// Cameras streamed from the robot, shown immersively over passthrough.
//
// The screens behave like TVs: they are anchored in the world where the
// operator was looking when the session started, so the head can move freely
// while the frames stay put. Pointing a controller at a screen and holding the
// rear trigger grabs it — move the controller to reposition, release to drop.
// Positions are remembered per view mode. Because the trigger doubles as the
// gripper control, grabbing is only allowed while robot tracking is
// disengaged — the teleop server broadcasts its engage toggle over the
// WebSocket (`{"type": "tracking"}`), covering grips, X/reset, and saving.
// Clicking the right thumbstick re-anchors all screens to the current gaze
// and resets dragged positions.
//
// View modes (the right thumbstick is a latched 4-way picker — flick once, no
// holding; flicking the *active* direction returns to the default):
//   - "default":  passthrough with the two wrist cams as bottom-corner PiPs
//                 (left wrist → bottom-left, right wrist → bottom-right)
//   - "overhead": fullscreen overhead (per-eye stereo when both eyes stream)
//   - "left":     fullscreen left wrist
//   - "right":    fullscreen right wrist
//   - "split":    left + right wrists side-by-side, fullscreen
// Stick directions map to where each camera roughly sits: up → overhead,
// left → left wrist, right → right wrist, down → split.
type ViewMode = "default" | "overhead" | "left" | "right" | "split"

// Which draggable screen a plane belongs to: the fullscreen/stereo plane
// ("main") or one of the two wrist-cam planes ("a" = left, "b" = right).
type GrabSlot = "main" | "a" | "b"

const FEED_DISTANCE = 1 // metres from the anchor point to the screens
const FEED_HEIGHT = 1.05 // fullscreen plane height in metres (width from aspect)
// Drop the feed slightly so its centre lands on the operator's natural gaze
// rather than sitting high in view.
const FEED_Y = -0.175
const STICK_DEADZONE = 0.6
// Bottom-corner picture-in-picture wrist cams shown in the default view.
const PIP_WIDTH = 0.5 // metres (height derives from aspect)
const PIP_X = 0.38 // horizontal offset of each corner PiP
const PIP_Y = -0.44 // vertical offset (lower corners)
// Side-by-side wrist cams in the split view.
const SPLIT_WIDTH = 0.92 // metres per pane
const SPLIT_X = 0.48 // horizontal offset of each pane from centre

// Scratch objects for the per-frame grab raycast (avoid allocations).
const _raycaster = new THREE.Raycaster()
_raycaster.layers.enableAll() // the stereo eye planes live on layers 1/2
const _rayMatrix = new THREE.Matrix4()
const _rayOrigin = new THREE.Vector3()
const _rayDir = new THREE.Vector3()
const _grabTarget = new THREE.Vector3()
const _yawFwd = new THREE.Vector3()
const _yAxis = new THREE.Vector3(0, 1, 0)

function ImmersiveCameraFeed({ wsRef }: { wsRef: RefObject<WebSocket | null> }) {
  const { gl } = useThree()
  const session = useXR((s) => s.session)
  // Only negotiate video while the headset is presenting. `available` is null
  // until known, false when the server reports no video — used to decide
  // whether to keep showing the loading spinner.
  const { streams, available } = useAxolVideo(wsRef, session != null)

  const groupRef = useRef<THREE.Group>(null)
  const meshRef = useRef<THREE.Mesh>(null)
  const matRef = useRef<THREE.MeshBasicMaterial>(null)
  // Per-eye planes for a stereo overhead: left on layer 1 (left lens only),
  // right on layer 2 (right lens only). Unused for mono feeds.
  const leftMeshRef = useRef<THREE.Mesh>(null)
  const leftMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const rightMeshRef = useRef<THREE.Mesh>(null)
  const rightMatRef = useRef<THREE.MeshBasicMaterial>(null)
  // Two shared planes (layer 0) reused for the two wrist cams in both the split
  // view (side-by-side) and the default view (bottom-corner PiPs); the two modes
  // never render at once. A = left wrist, B = right wrist.
  const dualAMeshRef = useRef<THREE.Mesh>(null)
  const dualAMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const dualBMeshRef = useRef<THREE.Mesh>(null)
  const dualBMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const spinnerRef = useRef<THREE.Group>(null)
  const spinnerMeshRef = useRef<THREE.Mesh>(null)
  const videosRef = useRef<Record<string, HTMLVideoElement>>({})
  const texturesRef = useRef<Record<string, THREE.VideoTexture>>({})
  // The current view mode; only changes on a thumbstick flick (edge), so the
  // operator doesn't have to hold the stick.
  const viewModeRef = useRef<ViewMode>("default")
  // Whether the right stick was deflected last frame, for rising-edge detection.
  const rightStickActiveRef = useRef(false)
  // Whether the screen group has been world-anchored for this XR session.
  const anchoredRef = useRef(false)
  // Active screen grab (trigger held while pointing at a plane), if any.
  const grabRef = useRef<{
    hand: "left" | "right"
    slot: GrabSlot
    distance: number
    mode: ViewMode
  } | null>(null)
  // User-dragged position offsets, keyed `${mode}:${slot}` so each view mode
  // remembers its own arrangement.
  const dragOffsetsRef = useRef<Record<string, THREE.Vector3>>({})
  // Per-hand trigger state last frame, for rising-edge grab detection.
  const triggerPrevRef = useRef({ left: false, right: false })
  // Whether robot tracking is engaged (the server owns this toggle and pushes
  // it over the WebSocket). Screen grabbing is blocked while engaged, since the
  // trigger drives the gripper then.
  const robotEngagedRef = useAxolTracking(wsRef)
  // Right-thumbstick click last frame, for rising-edge re-anchor detection.
  const stickClickPrevRef = useRef(false)

  // Wrap each incoming MediaStream in a <video> + VideoTexture, and tear down
  // textures whose stream has gone away.
  useEffect(() => {
    const videos = videosRef.current
    const textures = texturesRef.current
    for (const [name, stream] of Object.entries(streams)) {
      let video = videos[name]
      if (!video) {
        video = document.createElement("video")
        video.muted = true
        video.autoplay = true
        video.playsInline = true
        videos[name] = video
      }
      if (video.srcObject !== stream) {
        video.srcObject = stream
        void video.play().catch(() => {})
      }
      if (!textures[name]) {
        const tex = new THREE.VideoTexture(video)
        tex.colorSpace = THREE.SRGBColorSpace
        textures[name] = tex
      }
    }
    for (const name of Object.keys(textures)) {
      if (streams[name]) continue
      textures[name].dispose()
      delete textures[name]
      const video = videos[name]
      if (video) {
        video.srcObject = null
        delete videos[name]
      }
    }
  }, [streams])

  // Release GPU textures / video elements when the feed unmounts.
  useEffect(() => {
    const videos = videosRef.current
    const textures = texturesRef.current
    return () => {
      for (const tex of Object.values(textures)) tex.dispose()
      for (const video of Object.values(videos)) video.srcObject = null
    }
  }, [])

  // Confine the stereo eye planes to their lens via three.js layers: an object
  // on layer 1 renders to the left eye only, layer 2 to the right eye only.
  useEffect(() => {
    leftMeshRef.current?.layers.set(1)
    rightMeshRef.current?.layers.set(2)
  }, [])

  useFrame((_state, _delta, frame) => {
    const group = groupRef.current
    const mesh = meshRef.current
    const mat = matRef.current
    const spinner = spinnerRef.current
    const leftMesh = leftMeshRef.current
    const rightMesh = rightMeshRef.current
    const leftMat = leftMatRef.current
    const rightMat = rightMatRef.current
    const aMesh = dualAMeshRef.current
    const aMat = dualAMatRef.current
    const bMesh = dualBMeshRef.current
    const bMat = dualBMatRef.current
    if (!group || !mesh || !mat || !spinner) return
    if (!leftMesh || !rightMesh || !leftMat || !rightMat) return
    if (!aMesh || !aMat || !bMesh || !bMat) return

    const presenting = gl.xr.isPresenting
    const cam = gl.xr.getCamera()
    const textures = texturesRef.current

    // A camera is "available" once the relay has offered a track for it (its
    // texture exists). The relay only offers the cameras actually being streamed,
    // so we use this to (a) refuse to switch to a view whose camera isn't
    // streamed and (b) only show the connecting spinner for cameras we're
    // genuinely waiting on — never ones that will never arrive (e.g. wrist views
    // when only the overhead is streamed).
    const has = (name: string) => !!textures[name]
    const modeAvailable = (m: ViewMode): boolean => {
      if (m === "overhead") return has("overhead_left") || has("overhead_right") || has("overhead")
      if (m === "left") return has("left_arm")
      if (m === "right") return has("right_arm")
      if (m === "split") return has("left_arm") || has("right_arm")
      return true // "default" passthrough needs no camera
    }

    // World-anchor the screen group once per session: place it at the head
    // with a yaw-only orientation. After that the screens stay put like TVs —
    // the operator can look around freely. The loading spinner stays
    // head-locked so it's always seen.
    if (presenting && !anchoredRef.current) {
      anchoredRef.current = true
      grabRef.current = null
      dragOffsetsRef.current = {}
      group.position.copy(cam.position)
      _yawFwd.set(0, 0, -1).applyQuaternion(cam.quaternion)
      group.quaternion.setFromAxisAngle(_yAxis, Math.atan2(-_yawFwd.x, -_yawFwd.z))
      group.updateMatrixWorld(true)
    }
    if (!presenting) anchoredRef.current = false
    if (presenting) {
      spinner.position.copy(cam.position)
      spinner.quaternion.copy(cam.quaternion)
    }

    // Right thumbstick = latched 4-way view picker. axes[2]/[0] = X, axes[3]/[1]
    // = Y (up is negative). The dominant axis picks a direction → target mode;
    // only a rising edge (neutral → deflected) acts, so the view stays put when
    // the stick recentres. Flicking the active direction returns to "default".
    const xrSession = gl.xr.getSession()
    const sources = xrSession ? Array.from(xrSession.inputSources) : []
    const right = sources.find((s) => s.handedness === "right")
    const axes = right?.gamepad?.axes
    const sx = axes?.[2] ?? axes?.[0] ?? 0
    const sy = axes?.[3] ?? axes?.[1] ?? 0
    const stickActive = Math.max(Math.abs(sx), Math.abs(sy)) > STICK_DEADZONE
    if (stickActive && !rightStickActiveRef.current) {
      let target: ViewMode
      if (Math.abs(sy) > Math.abs(sx)) target = sy < 0 ? "overhead" : "split"
      else target = sx < 0 ? "left" : "right"
      // Ignore a flick toward a camera that isn't being streamed, so the operator
      // can't land on a view that would just sit on "Connecting camera…".
      // "default" (passthrough) is always reachable.
      if (modeAvailable(target)) {
        viewModeRef.current = viewModeRef.current === target ? "default" : target
        grabRef.current = null
      }
    }
    rightStickActiveRef.current = stickActive

    // Clicking the right thumbstick (buttons[3]) re-anchors the screens to the
    // current gaze and resets any dragged positions.
    const stickClicked = right?.gamepad?.buttons?.[3]?.pressed ?? false
    if (stickClicked && !stickClickPrevRef.current) {
      anchoredRef.current = false
      grabRef.current = null
      dragOffsetsRef.current = {}
    }
    stickClickPrevRef.current = stickClicked

    const mode = viewModeRef.current

    // A texture is usable only once its <video> has decoded real frames.
    const liveTex = (name: string) => {
      const t = textures[name]
      const v = t?.image as HTMLVideoElement | undefined
      return t && v && v.videoWidth ? t : undefined
    }

    // Which cams a mode needs to be considered "live" (vs showing the spinner).
    let live: boolean
    if (mode === "overhead") live = !!(liveTex("overhead_left") ?? liveTex("overhead"))
    else if (mode === "left") live = !!liveTex("left_arm")
    else if (mode === "right") live = !!liveTex("right_arm")
    else live = !!(liveTex("left_arm") ?? liveTex("right_arm")) // default, split

    group.visible = presenting && live
    // Spinner only while genuinely waiting on a camera that IS being streamed
    // (its track exists but hasn't decoded its first frame yet). Never for the
    // default passthrough view, and never for a camera that isn't streamed at
    // all — so streaming just the overhead no longer leaves the wrist views
    // stuck on "Connecting camera…".
    const expecting = mode !== "default" && modeAvailable(mode)
    spinner.visible = presenting && expecting && !live && available !== false
    if (spinner.visible && spinnerMeshRef.current) {
      spinnerMeshRef.current.rotation.z -= 0.12
    }

    // Hide every plane up front; each mode re-enables only what it draws.
    mesh.visible = false
    leftMesh.visible = false
    rightMesh.visible = false
    aMesh.visible = false
    bMesh.visible = false
    if (!group.visible) return

    // Place a plane sized to a target height (width from the video aspect).
    const fitHeight = (
      m: THREE.Mesh,
      mt: THREE.MeshBasicMaterial,
      t: THREE.VideoTexture,
      height: number,
      x: number,
      y: number
    ) => {
      const v = t.image as HTMLVideoElement | undefined
      const aspect = v && v.videoWidth ? v.videoWidth / v.videoHeight : 16 / 9
      if (mt.map !== t) {
        mt.map = t
        mt.needsUpdate = true
      }
      m.scale.set(height * aspect, height, 1)
      m.position.set(x, y, -FEED_DISTANCE)
      m.visible = true
    }
    // Place a plane sized to a target width (height from the video aspect).
    const fitWidth = (
      m: THREE.Mesh,
      mt: THREE.MeshBasicMaterial,
      t: THREE.VideoTexture,
      width: number,
      x: number,
      y: number
    ) => {
      const v = t.image as HTMLVideoElement | undefined
      const aspect = v && v.videoWidth ? v.videoWidth / v.videoHeight : 16 / 9
      fitHeight(m, mt, t, width / aspect, x, y)
      m.scale.x = width
    }

    if (mode === "overhead") {
      const oL = liveTex("overhead_left")
      const oR = liveTex("overhead_right")
      if (oL && oR) {
        // True stereo: each eye sees its own image (layer 1 left, layer 2 right).
        fitHeight(leftMesh, leftMat, oL, FEED_HEIGHT, 0, FEED_Y)
        fitHeight(rightMesh, rightMat, oR, FEED_HEIGHT, 0, FEED_Y)
        const eyes = (cam as THREE.ArrayCamera).cameras
        if (eyes && eyes.length >= 2) {
          eyes[0].layers.enable(1)
          eyes[1].layers.enable(2)
        }
      } else {
        const o = oL ?? liveTex("overhead")
        if (o) fitHeight(mesh, mat, o, FEED_HEIGHT, 0, FEED_Y)
      }
    } else if (mode === "left") {
      const t = liveTex("left_arm")
      if (t) fitHeight(mesh, mat, t, FEED_HEIGHT, 0, FEED_Y)
    } else if (mode === "right") {
      const t = liveTex("right_arm")
      if (t) fitHeight(mesh, mat, t, FEED_HEIGHT, 0, FEED_Y)
    } else if (mode === "split") {
      const l = liveTex("left_arm")
      const r = liveTex("right_arm")
      if (l) fitWidth(aMesh, aMat, l, SPLIT_WIDTH, -SPLIT_X, FEED_Y)
      if (r) fitWidth(bMesh, bMat, r, SPLIT_WIDTH, SPLIT_X, FEED_Y)
    } else {
      // default: passthrough with the wrist cams pinned to the bottom corners.
      const l = liveTex("left_arm")
      const r = liveTex("right_arm")
      if (l) fitWidth(aMesh, aMat, l, PIP_WIDTH, -PIP_X, PIP_Y)
      if (r) fitWidth(bMesh, bMat, r, PIP_WIDTH, PIP_X, PIP_Y)
    }

    // The layout above set each visible plane's *base* (group-local) position;
    // snapshot them before drag offsets are applied. The stereo eye planes are
    // coincident, so "main" covers both.
    const bases: Partial<Record<GrabSlot, THREE.Vector3>> = {}
    if (mesh.visible) bases.main = mesh.position.clone()
    if (leftMesh.visible) bases.main = leftMesh.position.clone()
    if (aMesh.visible) bases.a = aMesh.position.clone()
    if (bMesh.visible) bases.b = bMesh.position.clone()

    // Build the controller's pointing ray (origin + forward) in world space.
    const refSpace = gl.xr.getReferenceSpace()
    const computeRay = (src: XRInputSource): boolean => {
      if (!frame || !refSpace || !src.targetRaySpace) return false
      const pose = frame.getPose(src.targetRaySpace, refSpace)
      if (!pose) return false
      _rayMatrix.fromArray(pose.transform.matrix)
      _rayOrigin.setFromMatrixPosition(_rayMatrix)
      const e = _rayMatrix.elements
      _rayDir.set(-e[8], -e[9], -e[10]).normalize() // ray forward = -Z
      return true
    }

    // Grab handling: a rising-edge trigger press while pointing at a screen
    // grabs it at the hit distance; while held, the screen follows the ray
    // (offset from its layout base is remembered per `${mode}:${slot}`).
    // Disabled while robot tracking is engaged — the trigger drives the
    // gripper then, and a grab would fight the teleop.
    if (robotEngagedRef.current) grabRef.current = null
    for (const src of sources) {
      const hand = src.handedness
      if (hand !== "left" && hand !== "right") continue
      const pressed = src.gamepad?.buttons?.[0]?.pressed ?? false
      const wasPressed = triggerPrevRef.current[hand]
      triggerPrevRef.current[hand] = pressed
      if (robotEngagedRef.current) continue

      if (!pressed) {
        if (grabRef.current?.hand === hand) grabRef.current = null
        continue
      }
      if (pressed && !wasPressed && !grabRef.current && computeRay(src)) {
        const candidates: [THREE.Mesh, GrabSlot][] = []
        if (mesh.visible) candidates.push([mesh, "main"])
        if (leftMesh.visible) candidates.push([leftMesh, "main"])
        if (aMesh.visible) candidates.push([aMesh, "a"])
        if (bMesh.visible) candidates.push([bMesh, "b"])
        _raycaster.set(_rayOrigin, _rayDir)
        const hits = _raycaster.intersectObjects(
          candidates.map((c) => c[0]),
          false
        )
        const hit = hits[0]
        if (hit) {
          const slot = candidates.find((c) => c[0] === hit.object)?.[1]
          if (slot) grabRef.current = { hand, slot, distance: hit.distance, mode }
        }
      }
      const grab = grabRef.current
      if (grab && grab.hand === hand && grab.mode === mode && computeRay(src)) {
        const base = bases[grab.slot]
        if (base) {
          _grabTarget.copy(_rayOrigin).addScaledVector(_rayDir, grab.distance)
          group.worldToLocal(_grabTarget)
          dragOffsetsRef.current[`${mode}:${grab.slot}`] = _grabTarget.clone().sub(base)
        }
      }
    }

    // Apply any remembered drag offset, then turn each screen to face the
    // operator's head (a world-anchored panel viewed edge-on is useless).
    const orient = (m: THREE.Mesh, slot: GrabSlot) => {
      if (!m.visible) return
      const off = dragOffsetsRef.current[`${mode}:${slot}`]
      if (off) m.position.add(off)
      m.lookAt(cam.position)
    }
    orient(mesh, "main")
    orient(leftMesh, "main")
    orient(rightMesh, "main")
    orient(aMesh, "a")
    orient(bMesh, "b")
  })

  return (
    <>
      <group ref={groupRef} visible={false}>
        <mesh ref={meshRef} position={[0, FEED_Y, -FEED_DISTANCE]} renderOrder={1}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial ref={matRef} toneMapped={false} depthTest={false} depthWrite={false} />
        </mesh>
        <mesh
          ref={leftMeshRef}
          position={[0, FEED_Y, -FEED_DISTANCE]}
          renderOrder={1}
          visible={false}
        >
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial
            ref={leftMatRef}
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
        <mesh
          ref={rightMeshRef}
          position={[0, FEED_Y, -FEED_DISTANCE]}
          renderOrder={1}
          visible={false}
        >
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial
            ref={rightMatRef}
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
        {/* Shared wrist-cam planes: split panes or default-view corner PiPs. */}
        <mesh ref={dualAMeshRef} renderOrder={1} visible={false}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial
            ref={dualAMatRef}
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
        <mesh ref={dualBMeshRef} renderOrder={1} visible={false}>
          <planeGeometry args={[1, 1]} />
          <meshBasicMaterial
            ref={dualBMatRef}
            toneMapped={false}
            depthTest={false}
            depthWrite={false}
          />
        </mesh>
      </group>
      <group ref={spinnerRef} visible={false}>
        {/* Spinning arc (a torus with a gap) shown while the cameras connect. */}
        <mesh ref={spinnerMeshRef} position={[0, 0, -FEED_DISTANCE]} renderOrder={2}>
          <torusGeometry args={[0.11, 0.014, 16, 48, Math.PI * 1.5]} />
          <meshBasicMaterial color="#eff483" toneMapped={false} depthTest={false} />
        </mesh>
        <HudText
          position={[0, -0.22, -FEED_DISTANCE]}
          fontSize={0.045}
          fontWeight="bold"
          color="white"
          anchorX="center"
          anchorY="top"
          renderOrder={2}
          material-depthTest={false}
          {...hudBg}
        >
          Connecting camera…
        </HudText>
      </group>
    </>
  )
}

const hudBg = { backgroundColor: "#000000", backgroundOpacity: 0.5, padding: 0.006 } as object

function XRHud({ children }: { children: ReactNode }) {
  const session = useXR((s) => s.session)
  const groupRef = useRef<THREE.Group>(null)

  useFrame(({ gl }) => {
    if (!groupRef.current) return
    groupRef.current.visible = gl.xr.isPresenting
    if (!gl.xr.isPresenting) return
    const activeCam = gl.xr.getCamera()
    groupRef.current.position.copy(activeCam.position)
    groupRef.current.quaternion.copy(activeCam.quaternion)
  })

  if (!session) return null

  return (
    <group ref={groupRef} visible={false}>
      {children}
    </group>
  )
}

function ExitButton() {
  const [hovered, setHovered] = useState(false)

  return (
    <HudText
      position={[-0.2, 0.1, -0.5]}
      fontSize={0.02}
      fontWeight="bold"
      color={hovered ? "yellow" : "white"}
      anchorX="left"
      anchorY="top"
      renderOrder={999}
      material-depthTest={false}
      {...hudBg}
      onPointerOver={() => setHovered(true)}
      onPointerOut={() => setHovered(false)}
      onClick={() => store.getState().session?.end()}
    >
      Exit
    </HudText>
  )
}

const STATUS_DISPLAY: Partial<Record<AxolState | "pending", { color: string; label: string }>> = {
  pending: { color: "yellow", label: "• Starting…" },
  [AxolState.Error]: { color: "#f87171", label: "• Error" },
  [AxolState.Recording]: { color: "red", label: "• Recording" },
  [AxolState.Saving]: { color: "orange", label: "• Saving…" },
  [AxolState.DataCollection]: { color: "blue", label: "• Data Collection" },
}

function StateDisplay({
  state,
  isRecordingPending,
}: {
  state: AxolState
  isRecordingPending: boolean
}) {
  const displayState: AxolState | "pending" = isRecordingPending ? "pending" : state
  const { color, label } = STATUS_DISPLAY[displayState] ?? { color: "white", label: "• Teleop" }

  return (
    <HudText
      position={[0.2, 0.1, -0.5]}
      fontSize={0.02}
      fontWeight="bold"
      color={color}
      anchorX="right"
      anchorY="top"
      renderOrder={999}
      material-depthTest={false}
      {...hudBg}
    >
      {label}
    </HudText>
  )
}

function HelpPanel({ onDismiss }: { onDismiss: () => void }) {
  const W = 0.44
  const H = 0.133
  const col = 0.11

  return (
    <group position={[0, -0.038, 0]}>
      {/* Large dismiss plane behind everything */}
      <mesh position={[0, 0, -0.002]} renderOrder={996} onClick={onDismiss}>
        <planeGeometry args={[2, 2]} />
        <meshBasicMaterial transparent opacity={0} depthTest={false} side={THREE.DoubleSide} />
      </mesh>
      {/* Panel background */}
      <mesh position={[0, -H / 2, -0.001]} renderOrder={998} onClick={(e) => e.stopPropagation()}>
        <planeGeometry args={[W, H]} />
        <meshBasicMaterial
          color="black"
          transparent
          opacity={0.97}
          depthTest={false}
          side={THREE.DoubleSide}
        />
      </mesh>
      {/* Vertical divider */}
      <mesh position={[0, -H / 2, 0]} renderOrder={999}>
        <planeGeometry args={[0.002, H]} />
        <meshBasicMaterial color="white" depthTest={false} side={THREE.DoubleSide} />
      </mesh>
      {/* LEFT header */}
      <HudText
        position={[-col, -0.004, 0]}
        fontSize={0.013}
        color="white"
        fontWeight="bold"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
      >
        LEFT
      </HudText>
      {/* RIGHT header */}
      <HudText
        position={[col, -0.004, 0]}
        fontSize={0.013}
        color="white"
        fontWeight="bold"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
      >
        RIGHT
      </HudText>
      {/* Left buttons */}
      <HudText
        position={[-col, -0.022, 0]}
        fontSize={0.013}
        color="white"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
        lineHeight={1.6}
      >
        {`[Y]  Exit VR\n[X]  Reset Pose`}
      </HudText>
      {/* Right buttons */}
      <HudText
        position={[col, -0.022, 0]}
        fontSize={0.013}
        color="white"
        anchorX="center"
        anchorY="top"
        renderOrder={1000}
        material-depthTest={false}
        lineHeight={1.6}
      >
        {`[B]  Toggle Mode\n[A]  Start / Stop Rec\n[Stick]  Switch View\n[Trigger]  Move Screen\n[Stick Click]  Reset Screens`}
      </HudText>
    </group>
  )
}

function HelpIcon() {
  const [open, setOpen] = useState(false)

  return (
    <group position={[0, 0.1, -0.5]}>
      <HudText
        fontSize={0.02}
        fontWeight="bold"
        color={open ? "yellow" : "white"}
        anchorX="center"
        anchorY="top"
        renderOrder={999}
        material-depthTest={false}
        {...hudBg}
        onClick={() => setOpen((v) => !v)}
      >
        ?
      </HudText>
      {open && <HelpPanel onDismiss={() => setOpen(false)} />}
    </group>
  )
}

function CountdownDisplay({ recordingPendingAt }: { recordingPendingAt: number | null }) {
  const [count, setCount] = useState(3)
  const prevCountRef = useRef(3)

  useFrame(() => {
    if (recordingPendingAt === null) return
    const remaining = Math.ceil((3000 - (Date.now() - recordingPendingAt)) / 1000)
    const clamped = Math.max(1, Math.min(3, remaining))
    if (clamped !== prevCountRef.current) {
      prevCountRef.current = clamped
      setCount(clamped)
    }
  })

  if (recordingPendingAt === null) return null

  return (
    <HudText
      position={[0, 0, -0.5]}
      fontSize={0.1}
      fontWeight="bold"
      color="white"
      anchorX="center"
      anchorY="middle"
      renderOrder={999}
      material-depthTest={false}
    >
      {String(count)}
    </HudText>
  )
}

function ControlHints({ title, rows }: { title: string; rows: [string, string][] }) {
  return (
    <div className="rounded-lg border border-white/10 bg-white/[0.02] p-3">
      <div className="mb-1.5 font-mono text-[0.65rem] tracking-widest text-white/40 uppercase">
        {title}
      </div>
      <div className="flex flex-col gap-1">
        {rows.map(([key, label]) => (
          <div key={key} className="flex items-center gap-2">
            <kbd className="flex h-5 min-w-5 items-center justify-center rounded border border-white/15 bg-white/[0.06] px-1 font-mono text-[0.65rem] whitespace-nowrap text-white/70">
              {key}
            </kbd>
            <span className="text-white/60">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}

function ConnectionStatus({ status }: { status: AxolConnectionStatus }) {
  const meta =
    status === AxolConnectionStatus.Open
      ? { dot: "bg-emerald-400", ring: "bg-emerald-400/40", label: "Connected" }
      : status === AxolConnectionStatus.Connecting
        ? { dot: "bg-amber-400", ring: "bg-amber-400/40", label: "Connecting…" }
        : status === AxolConnectionStatus.Failed
          ? { dot: "bg-red-400", ring: "bg-red-400/40", label: "Connection failed" }
          : { dot: "bg-white/40", ring: "bg-white/10", label: "Not connected" }

  return (
    <div className="flex items-center justify-center gap-2 text-sm text-white/60">
      <span className="relative flex size-2.5">
        {status === AxolConnectionStatus.Connecting && (
          <span
            className={cn(
              "absolute inline-flex h-full w-full animate-ping rounded-full",
              meta.ring
            )}
          />
        )}
        <span className={cn("relative inline-flex size-2.5 rounded-full", meta.dot)} />
      </span>
      {meta.label}
    </div>
  )
}

export default function App() {
  const [hostname, setHostname] = useState(() => localStorage.getItem("wsHostname") ?? "")
  const [usbPoses, setUsbPoses] = useState(() => localStorage.getItem("usbPoses") === "1")
  const [vrState, setVrState] = useState<AxolState>(AxolState.Teleop)
  const [recordingPendingAt, setRecordingPendingAt] = useState<number | null>(null)
  const { status, connect, disconnect, wsRef } = useAxolVRClient(hostname)
  // Controller poses can ride a wired USB `adb reverse` tunnel (localhost) to
  // avoid WiFi latency; camera video keeps using the LAN host above. The pose
  // socket comes up once the main connection is open and the operator opts in.
  const { poseWsRef, status: poseStatus } = useAxolPoseSocket(
    usbPoses && status === AxolConnectionStatus.Open
  )

  const handleConnect = () => {
    localStorage.setItem("wsHostname", hostname)
    connect()
  }

  const handleUsbToggle = (next: boolean) => {
    setUsbPoses(next)
    localStorage.setItem("usbPoses", next ? "1" : "0")
  }

  return (
    <>
      <div className="pointer-events-none fixed inset-0 z-10 flex flex-col bg-[#121212]/70 backdrop-blur-sm">
        <div className="pointer-events-auto">
          <SiteNav current="vr" />
        </div>
        <div className="flex flex-1 items-center justify-center p-6">
          <Card className="pointer-events-auto w-full max-w-sm gap-6">
            <div className="flex flex-col items-center gap-3 text-center">
              <img src="/almond.svg" alt="Almond" className="h-12 w-12" />
              <div>
                <h1 className="font-heading text-2xl font-bold tracking-tight">Almond Axol</h1>
                <p className="text-sm text-white/40">VR Teleoperation</p>
              </div>
            </div>

            <ConnectionStatus status={status} />

            {status === AxolConnectionStatus.Open ? (
              <div className="flex flex-col gap-2">
                <Button
                  size="lg"
                  className="w-full"
                  onClick={() => store.enterAR()}
                  disabled={usbPoses && poseStatus !== AxolConnectionStatus.Open}
                >
                  <Headset />
                  Enter VR
                </Button>
                <Button variant="ghost" className="w-full" onClick={disconnect}>
                  Disconnect
                </Button>
                {usbPoses && (
                  <div className="flex flex-col gap-2">
                    <p className="text-center text-xs text-white/40">
                      Quest over USB:{" "}
                      <span
                        className={cn(
                          "font-medium",
                          poseStatus === AxolConnectionStatus.Open
                            ? "text-emerald-400"
                            : "text-amber-400"
                        )}
                      >
                        {poseStatus === AxolConnectionStatus.Open
                          ? "controller over cable"
                          : "waiting for USB link…"}
                      </span>
                    </p>
                    {poseStatus !== AxolConnectionStatus.Open && (
                      <Button
                        variant="outline"
                        size="sm"
                        className="w-full"
                        onClick={() => authorizeCert(`https://localhost:${VR_WS_PORT}`)}
                      >
                        <ShieldCheck />
                        Authorize USB certificate
                      </Button>
                    )}
                  </div>
                )}
              </div>
            ) : status === AxolConnectionStatus.Connecting ? (
              <Button variant="secondary" className="w-full" onClick={disconnect}>
                <Loader2 className="animate-spin" />
                Cancel
              </Button>
            ) : (
              <form
                onSubmit={(e) => {
                  e.preventDefault()
                  handleConnect()
                }}
                className="flex flex-col gap-2"
              >
                <label
                  htmlFor="vr-host"
                  className="text-xs font-medium tracking-widest text-white/40 uppercase"
                >
                  Axol Host Address
                </label>
                <Input
                  id="vr-host"
                  type="text"
                  value={hostname}
                  onChange={(e) => setHostname(e.target.value)}
                  placeholder="axol-host.local"
                />
                <label className="flex items-center gap-2 rounded-lg border border-white/10 bg-white/5 p-3 text-sm text-white/80">
                  <input
                    type="checkbox"
                    checked={usbPoses}
                    onChange={(e) => handleUsbToggle(e.target.checked)}
                    className="size-4 shrink-0 accent-white"
                  />
                  Quest over USB
                </label>
                <Button type="submit" className="w-full" disabled={!hostname.trim()}>
                  Connect
                </Button>
              </form>
            )}

            {status === AxolConnectionStatus.Open && (
              <div className="grid grid-cols-2 gap-3 text-left text-xs">
                <ControlHints
                  title="Left"
                  rows={[
                    ["Y", "Exit VR"],
                    ["X", "Reset pose"],
                  ]}
                />
                <ControlHints
                  title="Right"
                  rows={[
                    ["B", "Toggle mode"],
                    ["A", "Start / stop rec"],
                    ["Stick", "Switch view"],
                    ["Stick click", "Reset screens"],
                    ["Trigger", "Move screen"],
                  ]}
                />
              </div>
            )}

            {status === AxolConnectionStatus.Failed && (
              <div className="flex flex-col gap-2">
                <p className="rounded-lg border border-red-400/25 bg-red-400/10 p-3 text-xs text-red-300">
                  Could not connect to <span className="font-mono">{hostname || "the server"}</span>
                  . Check that <span className="font-mono">axol teleop</span> is running, then
                  authorize its self-signed certificate below.
                </p>
                {hostname.trim() && (
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={() =>
                      authorizeCert(`https://${hostname.trim()}:${VR_WS_PORT}`).then(handleConnect)
                    }
                  >
                    <ShieldCheck />
                    Authorize certificate
                  </Button>
                )}
              </div>
            )}
          </Card>
        </div>
      </div>

      <Canvas>
        {/* Boundary so in-canvas suspense (drei <Text> preloading its font)
            never bubbles up to the route-level Suspense — without it, the
            whole page flashes back to the loading spinner shortly after
            first paint. The text is only visible in-headset, so an empty
            fallback is fine. */}
        <Suspense fallback={null}>
          <XR store={store}>
            <AxolVRClient
              wsRef={wsRef}
              poseWsRef={poseWsRef}
              usbOnly={usbPoses}
              onStateChange={setVrState}
              onPendingRecording={setRecordingPendingAt}
              onExit={() => store.getState().session?.end()}
            />
            <ImmersiveCameraFeed wsRef={wsRef} />
            <XRHud>
              <ExitButton />
              <HelpIcon />
              <StateDisplay state={vrState} isRecordingPending={recordingPendingAt !== null} />
              <CountdownDisplay recordingPendingAt={recordingPendingAt} />
            </XRHud>
            <PoseVisualizer />
          </XR>
        </Suspense>
      </Canvas>
    </>
  )
}
