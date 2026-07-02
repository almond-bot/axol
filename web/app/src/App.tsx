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
  type AxolMode,
  AxolVRClient,
  AxolState,
  axolHttpsOrigin,
  useAxolControlChannel,
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
// All three feeds are shown at once: the overhead in the centre (per-eye stereo
// when both eyes stream) with the two wrist cams as bottom-corner picture-in-
// picture panes (left wrist → bottom-left, right wrist → bottom-right). There
// is no view picker — the operator tailors the layout directly instead:
//   - Move a screen: point a controller at it and hold the rear trigger, then
//     move the controller; release to drop. Each screen remembers its spot.
//   - Resize a screen: grab the *same* screen with both controllers' triggers
//     and move your hands apart (bigger) or together (smaller).
//   - Reset: click the right thumbstick to re-anchor every screen to the
//     current gaze and clear all moves + resizes.
//
// The screens behave like TVs: they are world-anchored where the operator was
// looking when the session started, so the head can move freely while the
// frames stay put. Because the trigger doubles as the gripper control, grabbing
// is only allowed while robot tracking is disengaged — the teleop server
// broadcasts its engage toggle over the WebSocket (`{"type": "tracking"}`),
// covering grips, X/reset, and saving.

// Which draggable screen a plane belongs to: the overhead (its mono plane or
// its two stereo eye planes, both "overhead") or a wrist-cam plane.
type GrabSlot = "overhead" | "left" | "right"

const FEED_DISTANCE = 1 // metres from the anchor point to the screens
const FEED_HEIGHT = 1.05 // overhead plane height in metres (width from aspect)
// Drop the feed slightly so its centre lands on the operator's natural gaze
// rather than sitting high in view.
const FEED_Y = -0.175
// Bottom-corner picture-in-picture wrist cams flanking the overhead.
const PIP_WIDTH = 0.5 // metres (height derives from aspect)
const PIP_X = 0.72 // horizontal offset of each corner PiP
const PIP_Y = -0.52 // vertical offset (lower corners)
// Per-screen resize limits, relative to each screen's default size.
const MIN_SCALE = 0.3
const MAX_SCALE = 4

// Scratch objects for the per-frame grab raycast (avoid allocations).
const _raycaster = new THREE.Raycaster()
_raycaster.layers.enableAll() // the stereo eye planes live on layers 1/2
const _rayMatrix = new THREE.Matrix4()
const _grabTarget = new THREE.Vector3()
const _yawFwd = new THREE.Vector3()
const _yAxis = new THREE.Vector3(0, 1, 0)
// Per-hand controller ray (origin + forward), refreshed each frame so both the
// single-hand drag and the two-hand resize can read either hand's pointing ray.
const _handRay = {
  left: { origin: new THREE.Vector3(), dir: new THREE.Vector3(), valid: false },
  right: { origin: new THREE.Vector3(), dir: new THREE.Vector3(), valid: false },
}

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
  // The two wrist-cam planes (layer 0): A = left wrist, B = right wrist.
  const dualAMeshRef = useRef<THREE.Mesh>(null)
  const dualAMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const dualBMeshRef = useRef<THREE.Mesh>(null)
  const dualBMatRef = useRef<THREE.MeshBasicMaterial>(null)
  const spinnerRef = useRef<THREE.Group>(null)
  const spinnerMeshRef = useRef<THREE.Mesh>(null)
  const videosRef = useRef<Record<string, HTMLVideoElement>>({})
  const texturesRef = useRef<Record<string, THREE.VideoTexture>>({})
  // Cameras that have decoded at least one real frame. Used to keep the last
  // good frame on screen through brief dropouts instead of cutting to
  // passthrough (see `liveTex`).
  const shownRef = useRef<Record<string, boolean>>({})
  // Whether the screen group has been world-anchored for this XR session.
  const anchoredRef = useRef(false)
  // Per-hand active grab (trigger held while pointing at a plane), if any.
  // When both hands hold the same slot we resize it instead of dragging.
  const grabsRef = useRef<{
    left: { slot: GrabSlot; distance: number } | null
    right: { slot: GrabSlot; distance: number } | null
  }>({ left: null, right: null })
  // Active two-hand resize: the slot being scaled plus the hand separation and
  // screen scale captured when the second hand grabbed on.
  const resizeRef = useRef<{ slot: GrabSlot; startSep: number; startScale: number } | null>(
    null
  )
  // User-dragged position offsets, keyed by slot (group-local).
  const dragOffsetsRef = useRef<Record<string, THREE.Vector3>>({})
  // Per-slot size multipliers (default 1), driven by the two-hand resize.
  const scalesRef = useRef<Record<string, number>>({})
  // Per-hand trigger state last frame, for rising-edge grab detection.
  const triggerPrevRef = useRef({ left: false, right: false })
  // Whether robot tracking is engaged (the server owns this toggle and pushes
  // it over the WebSocket). Screen grabbing is blocked while engaged, since the
  // trigger drives the gripper then.
  const robotEngagedRef = useAxolTracking(wsRef)
  // Right-thumbstick click last frame, for rising-edge re-anchor detection.
  const stickClickPrevRef = useRef(false)

  // Wrap each incoming MediaStream in a <video> + VideoTexture.
  //
  // We deliberately do NOT tear down a camera's <video>/texture when its stream
  // momentarily disappears from `streams`. A `THREE.VideoTexture` keeps the last
  // decoded frame uploaded on the GPU, so as long as we keep the texture alive
  // and the plane visible, a brief WebRTC blip (a transient `failed`/`closed`,
  // a socket swap, or a re-offer — all of which clear `streams` upstream) shows
  // the frozen last frame instead of cutting to passthrough. When the stream
  // returns we just re-point the existing <video> at the new MediaStream, so the
  // texture keeps streaming into the same object with no flash. Textures are
  // only released on unmount (below).
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
  }, [streams])

  // Release GPU textures / video elements when the feed unmounts.
  useEffect(() => {
    const videos = videosRef.current
    const textures = texturesRef.current
    return () => {
      for (const tex of Object.values(textures)) tex.dispose()
      for (const video of Object.values(videos)) video.srcObject = null
      shownRef.current = {}
    }
  }, [])

  // Fully release the cameras when the XR session ends. This component stays
  // mounted across sessions, and we intentionally keep the last frame through
  // *in-session* dropouts (see the `[streams]` effect), but a session end is a
  // clean teardown: without this, the sticky textures/videos/shownRef would
  // survive and a later re-entry would treat last session's frozen GPU frames
  // as live (hiding the connecting spinner) until fresh tracks arrive.
  useEffect(() => {
    if (session) return
    const videos = videosRef.current
    const textures = texturesRef.current
    for (const name of Object.keys(textures)) {
      textures[name].dispose()
      delete textures[name]
    }
    for (const name of Object.keys(videos)) {
      videos[name].srcObject = null
      delete videos[name]
    }
    shownRef.current = {}
  }, [session])

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

    // A texture is usable once its <video> has decoded at least one real frame.
    // Once that's happened we keep returning it (the texture still holds that
    // last frame on the GPU) even if `videoWidth` momentarily reads 0 during a
    // stall or stream reconnect — that's what stops the feed from flickering to
    // passthrough and back. It only becomes unusable again if the texture itself
    // is gone (i.e. on unmount).
    const liveTex = (name: string) => {
      const t = textures[name]
      if (!t) return undefined
      const v = t.image as HTMLVideoElement | undefined
      if (v && v.videoWidth) {
        shownRef.current[name] = true
        return t
      }
      return shownRef.current[name] ? t : undefined
    }

    // World-anchor the screen group once per session: place it at the head
    // with a yaw-only orientation. After that the screens stay put like TVs —
    // the operator can look around freely. The loading spinner stays
    // head-locked so it's always seen.
    if (presenting && !anchoredRef.current) {
      anchoredRef.current = true
      grabsRef.current = { left: null, right: null }
      resizeRef.current = null
      dragOffsetsRef.current = {}
      scalesRef.current = {}
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

    const xrSession = gl.xr.getSession()
    const sources = xrSession ? Array.from(xrSession.inputSources) : []
    const right = sources.find((s) => s.handedness === "right")

    // Clicking the right thumbstick (buttons[3]) re-anchors the screens to the
    // current gaze and clears every move + resize.
    const stickClicked = right?.gamepad?.buttons?.[3]?.pressed ?? false
    if (stickClicked && !stickClickPrevRef.current) {
      anchoredRef.current = false
      grabsRef.current = { left: null, right: null }
      resizeRef.current = null
      dragOffsetsRef.current = {}
      scalesRef.current = {}
    }
    stickClickPrevRef.current = stickClicked

    // All available feeds are shown at once: the overhead centred (true per-eye
    // stereo when both eyes stream, else a single mono plane) with the wrist
    // cams as bottom-corner PiPs.
    const oL = liveTex("overhead_left")
    const oR = liveTex("overhead_right")
    const overheadMono = oL && oR ? undefined : (oL ?? liveTex("overhead"))
    const leftTex = liveTex("left_arm")
    const rightTex = liveTex("right_arm")
    const anyLive = !!(oL || overheadMono || leftTex || rightTex)

    group.visible = presenting && anyLive

    // "Connecting cameras…" is a global indicator: show it while video is still
    // being negotiated (available === null), or while any camera that IS being
    // streamed hasn't produced its first frame yet. Hidden once every streamed
    // camera is live, and entirely when the server reports no video
    // (available === false — nothing is streaming).
    const streamed = Object.keys(textures)
    const allLive = streamed.length > 0 && streamed.every((n) => !!liveTex(n))
    const connecting = available === null || (available === true && !allLive)
    spinner.visible = presenting && connecting
    if (spinner.visible && spinnerMeshRef.current) {
      spinnerMeshRef.current.rotation.z -= 0.12
    }

    // Hide every plane up front; the layout below re-enables what it draws.
    mesh.visible = false
    leftMesh.visible = false
    rightMesh.visible = false
    aMesh.visible = false
    bMesh.visible = false
    if (!group.visible) return

    // Place a plane sized to a target height (width from the video aspect),
    // times the slot's user resize factor.
    const fitHeight = (
      m: THREE.Mesh,
      mt: THREE.MeshBasicMaterial,
      t: THREE.VideoTexture,
      height: number,
      x: number,
      y: number,
      scale: number
    ) => {
      const v = t.image as HTMLVideoElement | undefined
      const aspect = v && v.videoWidth ? v.videoWidth / v.videoHeight : 16 / 9
      if (mt.map !== t) {
        mt.map = t
        mt.needsUpdate = true
      }
      const h = height * scale
      m.scale.set(h * aspect, h, 1)
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
      y: number,
      scale: number
    ) => {
      const v = t.image as HTMLVideoElement | undefined
      const aspect = v && v.videoWidth ? v.videoWidth / v.videoHeight : 16 / 9
      fitHeight(m, mt, t, width / aspect, x, y, scale)
      m.scale.x = width * scale
    }

    const scaleOf = (slot: GrabSlot) => scalesRef.current[slot] ?? 1

    // Overhead in the centre.
    if (oL && oR) {
      // True stereo: each eye sees its own image (layer 1 left, layer 2 right).
      fitHeight(leftMesh, leftMat, oL, FEED_HEIGHT, 0, FEED_Y, scaleOf("overhead"))
      fitHeight(rightMesh, rightMat, oR, FEED_HEIGHT, 0, FEED_Y, scaleOf("overhead"))
      const eyes = (cam as THREE.ArrayCamera).cameras
      if (eyes && eyes.length >= 2) {
        eyes[0].layers.enable(1)
        eyes[1].layers.enable(2)
      }
    } else if (overheadMono) {
      fitHeight(mesh, mat, overheadMono, FEED_HEIGHT, 0, FEED_Y, scaleOf("overhead"))
    }
    // Wrist cams as bottom-corner PiPs (left → bottom-left, right → bottom-right).
    if (leftTex) fitWidth(aMesh, aMat, leftTex, PIP_WIDTH, -PIP_X, PIP_Y, scaleOf("left"))
    if (rightTex) fitWidth(bMesh, bMat, rightTex, PIP_WIDTH, PIP_X, PIP_Y, scaleOf("right"))

    // The layout above set each visible plane's *base* (group-local) position;
    // snapshot them before drag offsets are applied. The stereo eye planes are
    // coincident, so "overhead" covers both.
    const bases: Partial<Record<GrabSlot, THREE.Vector3>> = {}
    if (mesh.visible) bases.overhead = mesh.position.clone()
    if (leftMesh.visible) bases.overhead = leftMesh.position.clone()
    if (aMesh.visible) bases.left = aMesh.position.clone()
    if (bMesh.visible) bases.right = bMesh.position.clone()

    // Refresh each hand's pointing ray (origin + forward) in world space, so
    // both the single-hand drag and the two-hand resize can read either hand.
    const refSpace = gl.xr.getReferenceSpace()
    const computeRay = (hand: "left" | "right") => {
      const r = _handRay[hand]
      r.valid = false
      const src = sources.find((s) => s.handedness === hand)
      if (!src || !frame || !refSpace || !src.targetRaySpace) return
      const pose = frame.getPose(src.targetRaySpace, refSpace)
      if (!pose) return
      _rayMatrix.fromArray(pose.transform.matrix)
      r.origin.setFromMatrixPosition(_rayMatrix)
      const e = _rayMatrix.elements
      r.dir.set(-e[8], -e[9], -e[10]).normalize() // ray forward = -Z
      r.valid = true
    }
    computeRay("left")
    computeRay("right")

    // Grab handling. Pointing at a screen and pressing the trigger grabs it at
    // the hit distance. One hand grabbing drags the screen along its ray; both
    // hands grabbing the *same* screen resize it by their separation. Disabled
    // while robot tracking is engaged — the trigger drives the gripper then, and
    // a grab would fight the teleop.
    const grabs = grabsRef.current
    if (robotEngagedRef.current) {
      grabs.left = null
      grabs.right = null
      resizeRef.current = null
    }
    for (const hand of ["left", "right"] as const) {
      const src = sources.find((s) => s.handedness === hand)
      const pressed = src?.gamepad?.buttons?.[0]?.pressed ?? false
      const wasPressed = triggerPrevRef.current[hand]
      triggerPrevRef.current[hand] = pressed
      if (robotEngagedRef.current) continue
      if (!pressed) {
        grabs[hand] = null
        continue
      }
      if (!wasPressed && !grabs[hand] && _handRay[hand].valid) {
        const candidates: [THREE.Mesh, GrabSlot][] = []
        if (mesh.visible) candidates.push([mesh, "overhead"])
        if (leftMesh.visible) candidates.push([leftMesh, "overhead"])
        if (aMesh.visible) candidates.push([aMesh, "left"])
        if (bMesh.visible) candidates.push([bMesh, "right"])
        _raycaster.set(_handRay[hand].origin, _handRay[hand].dir)
        const hits = _raycaster.intersectObjects(
          candidates.map((c) => c[0]),
          false
        )
        const hit = hits[0]
        if (hit) {
          const slot = candidates.find((c) => c[0] === hit.object)?.[1]
          if (slot) grabs[hand] = { slot, distance: hit.distance }
        }
      }
    }

    // Two hands on the same screen → resize it by how far apart they are now
    // relative to when the second hand grabbed on. Otherwise each grabbing hand
    // drags its screen along the ray at the captured distance.
    if (
      grabs.left &&
      grabs.right &&
      grabs.left.slot === grabs.right.slot &&
      _handRay.left.valid &&
      _handRay.right.valid
    ) {
      const slot = grabs.left.slot
      const sep = _handRay.left.origin.distanceTo(_handRay.right.origin)
      const rz = resizeRef.current
      if (!rz || rz.slot !== slot) {
        resizeRef.current = { slot, startSep: sep, startScale: scaleOf(slot) }
      } else if (rz.startSep > 1e-3) {
        const next = rz.startScale * (sep / rz.startSep)
        scalesRef.current[slot] = Math.min(MAX_SCALE, Math.max(MIN_SCALE, next))
      }
    } else {
      resizeRef.current = null
      for (const hand of ["left", "right"] as const) {
        const grab = grabs[hand]
        const ray = _handRay[hand]
        if (!grab || !ray.valid) continue
        const base = bases[grab.slot]
        if (!base) continue
        _grabTarget.copy(ray.origin).addScaledVector(ray.dir, grab.distance)
        group.worldToLocal(_grabTarget)
        dragOffsetsRef.current[grab.slot] = _grabTarget.clone().sub(base)
      }
    }

    // Apply any remembered drag offset, then turn each screen to face the
    // operator's head (a world-anchored panel viewed edge-on is useless).
    const orient = (m: THREE.Mesh, slot: GrabSlot) => {
      if (!m.visible) return
      const off = dragOffsetsRef.current[slot]
      if (off) m.position.add(off)
      m.lookAt(cam.position)
    }
    orient(mesh, "overhead")
    orient(leftMesh, "overhead")
    orient(rightMesh, "overhead")
    orient(aMesh, "left")
    orient(bMesh, "right")
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
        {/* Wrist-cam planes shown as bottom-corner PiPs (A = left, B = right). */}
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
          Connecting cameras…
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

function HelpPanel({ onDismiss, mode }: { onDismiss: () => void; mode: AxolMode | null }) {
  const W = 0.44
  const H = 0.133
  const col = 0.11
  // Recording only exists in data collection; teleop drops the [A] hint.
  const rightRows =
    mode === "teleop"
      ? "[Trigger]  Move Screen\n[2× Trigger]  Resize\n[Stick Click]  Reset Screens"
      : "[A]  Start / Stop Rec\n[Trigger]  Move Screen\n[2× Trigger]  Resize\n[Stick Click]  Reset Screens"

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
        {rightRows}
      </HudText>
    </group>
  )
}

function HelpIcon({ mode }: { mode: AxolMode | null }) {
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
      {open && <HelpPanel onDismiss={() => setOpen(false)} mode={mode} />}
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
  // Operating mode the server locked us to (null until it announces one on
  // connect). Drives which HUD/hint controls are shown.
  const [vrMode, setVrMode] = useState<AxolMode | null>(null)
  const { status, connect, disconnect, wsRef } = useAxolVRClient(hostname)
  // Controller poses can ride a wired USB `adb reverse` tunnel (localhost) to
  // avoid WiFi latency; camera video keeps using the LAN host above. The pose
  // socket comes up once the main connection is open and the operator opts in.
  const { poseWsRef, status: poseStatus } = useAxolPoseSocket(
    usbPoses && status === AxolConnectionStatus.Open
  )
  // Low-latency WebRTC pose data channel — negotiated once the teleop
  // connection is up (independent of cameras / presenting). AxolVRClient prefers
  // it over the main WebSocket, which stays as the fallback.
  const { poseChannelRef } = useAxolControlChannel(wsRef, status === AxolConnectionStatus.Open)

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
                <Button size="lg" className="w-full" onClick={() => store.enterAR()}>
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
                          : poseStatus === AxolConnectionStatus.Connecting
                            ? "connecting USB link… (on WiFi)"
                            : "WiFi fallback — USB link down"}
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
                    ...(vrMode === "teleop"
                      ? []
                      : ([["A", "Start / stop rec"]] as [string, string][])),
                    ["Trigger", "Move screen"],
                    ["2× Trigger", "Resize screen"],
                    ["Stick click", "Reset screens"],
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
                      authorizeCert(axolHttpsOrigin(hostname, VR_WS_PORT)).then(handleConnect)
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
              poseChannelRef={poseChannelRef}
              onStateChange={setVrState}
              onPendingRecording={setRecordingPendingAt}
              onMode={setVrMode}
              onExit={() => store.getState().session?.end()}
            />
            <ImmersiveCameraFeed wsRef={wsRef} />
            <XRHud>
              <ExitButton />
              <HelpIcon mode={vrMode} />
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
