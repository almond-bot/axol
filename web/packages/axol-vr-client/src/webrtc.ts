// Cap on how long to wait for ICE gathering before signaling the SDP. Host
// candidates gather almost instantly; a TURN allocation is a quick round-trip.
// The cap prevents a stalled TURN server from hanging negotiation forever (we
// then signal whatever candidates we have).
const ICE_GATHER_TIMEOUT_MS = 3000

/**
 * Resolve once the peer connection has finished gathering ICE candidates, so a
 * non-trickle offer/answer carries them all (crucially the TURN relay
 * candidate). Falls back after `ICE_GATHER_TIMEOUT_MS` if gathering stalls.
 *
 * Shared by the video (`useAxolVideo`) and control-channel
 * (`useAxolControlChannel`) negotiations, both of which signal SDP over the
 * teleop WebSocket without trickle ICE.
 */
export function waitForIceGathering(pc: RTCPeerConnection): Promise<void> {
  if (pc.iceGatheringState === "complete") return Promise.resolve()
  return new Promise((resolve) => {
    const finish = () => {
      pc.removeEventListener("icegatheringstatechange", onChange)
      clearTimeout(timer)
      resolve()
    }
    const onChange = () => {
      if (pc.iceGatheringState === "complete") finish()
    }
    const timer = setTimeout(finish, ICE_GATHER_TIMEOUT_MS)
    pc.addEventListener("icegatheringstatechange", onChange)
  })
}
