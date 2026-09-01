type MutableCurrent<T> = { current: T }

export function videoPeerNeedsRetry(state: string): boolean {
  return state === "disconnected" || state === "failed" || state === "closed"
}

/**
 * Retire one peer only if it still belongs to the current signaling socket.
 *
 * Clearing the peer reference is the single-flight claim: a second terminal
 * callback for the same peer, or a delayed callback after socket replacement,
 * cannot schedule another retry or tear down its replacement.
 */
export function retireCurrentVideoPeer<Socket, Peer>(
  socketRef: MutableCurrent<Socket | null>,
  peerRef: MutableCurrent<Peer | null>,
  socket: Socket,
  peer: Peer
): boolean {
  if (socketRef.current !== socket || peerRef.current !== peer) return false
  peerRef.current = null
  return true
}
