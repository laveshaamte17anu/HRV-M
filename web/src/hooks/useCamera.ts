import { useCallback, useEffect, useMemo, useRef, useState } from 'react'

export type CameraState =
  | { status: 'idle' }
  | { status: 'requesting' }
  | { status: 'running'; stream: MediaStream }
  | { status: 'error'; error: string }

export interface UseCameraResult {
  state: CameraState
  devices: MediaDeviceInfo[]
  deviceId: string | null
  stream: MediaStream | null
  videoRef: React.RefObject<HTMLVideoElement | null>
  attachVideo: (node: HTMLVideoElement | null) => void
  start: (deviceId?: string) => Promise<MediaStream>
  stop: () => void
  setDeviceId: (id: string) => void
}

export function useCamera(): UseCameraResult {
  const [state, setState] = useState<CameraState>({ status: 'idle' })
  const [devices, setDevices] = useState<MediaDeviceInfo[]>([])
  const [deviceId, setDeviceIdState] = useState<string | null>(null)
  const deviceIdRef = useRef<string | null>(null)
  const videoRef = useRef<HTMLVideoElement | null>(null)
  const streamRef = useRef<MediaStream | null>(null)

  const attachVideo = useCallback((node: HTMLVideoElement | null) => {
    videoRef.current = node
    if (node && streamRef.current) {
      if (node.srcObject !== streamRef.current) {
        node.srcObject = streamRef.current
      }
      node.play().catch(() => {})
    }
  }, [])

  // Continuously sync videoRef with active stream on any re-render or layout change
  useEffect(() => {
    const video = videoRef.current
    const stream = streamRef.current
    if (video && stream && video.srcObject !== stream) {
      video.srcObject = stream
      video.play().catch(() => {})
    }
  })

  const stop = useCallback(() => {
    if (streamRef.current) {
      streamRef.current.getTracks().forEach((t) => t.stop())
      streamRef.current = null
    }
    if (videoRef.current) {
      videoRef.current.srcObject = null
    }
    setState({ status: 'idle' })
  }, [])

  const setDeviceId = useCallback((id: string) => {
    deviceIdRef.current = id
    setDeviceIdState(id)
  }, [])

  const start = useCallback(
    async (preferredId?: string): Promise<MediaStream> => {
      const targetId = preferredId || deviceIdRef.current

      // If we already have an active stream with live tracks, reuse it unless a different device was requested
      if (
        streamRef.current &&
        streamRef.current.active &&
        streamRef.current.getVideoTracks().some((t) => t.readyState === 'live')
      ) {
        if (!preferredId || deviceIdRef.current === preferredId) {
          if (videoRef.current && videoRef.current.srcObject !== streamRef.current) {
            videoRef.current.srcObject = streamRef.current
            await videoRef.current.play().catch(() => {})
          }
          setState({ status: 'running', stream: streamRef.current })
          return streamRef.current
        }
      }

      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }

      setState({ status: 'requesting' })

      try {
        if (typeof navigator === 'undefined' || !navigator.mediaDevices?.getUserMedia) {
          throw new Error(
            'Camera access is not supported by your browser in this context (requires HTTPS or localhost).'
          )
        }

        const constraints: MediaStreamConstraints = {
          video: targetId
            ? { deviceId: { exact: targetId }, width: { ideal: 1280 }, height: { ideal: 720 } }
            : { facingMode: 'user', width: { ideal: 1280 }, height: { ideal: 720 } },
          audio: false,
        }

        const stream = await navigator.mediaDevices.getUserMedia(constraints)
        streamRef.current = stream

        if (videoRef.current) {
          videoRef.current.srcObject = stream
          await videoRef.current.play().catch((err) => {
            console.warn('Video element play() was paused or deferred:', err)
          })
        }

        const all = await navigator.mediaDevices.enumerateDevices().catch(() => [])
        const cams = all.filter((d) => d.kind === 'videoinput')
        setDevices(cams)

        const activeTrackSettings = stream.getVideoTracks()[0]?.getSettings()
        if (activeTrackSettings?.deviceId) {
          deviceIdRef.current = activeTrackSettings.deviceId
          setDeviceIdState(activeTrackSettings.deviceId)
        }

        setState({ status: 'running', stream })
        return stream
      } catch (err) {
        const message =
          err instanceof Error
            ? err.message
            : 'Camera access failed. Please ensure camera permissions are allowed in your browser.'
        setState({ status: 'error', error: message })
        throw err
      }
    },
    []
  )

  useEffect(() => {
    return () => {
      if (streamRef.current) {
        streamRef.current.getTracks().forEach((t) => t.stop())
        streamRef.current = null
      }
    }
  }, [])

  return useMemo(
    () => ({
      state,
      devices,
      deviceId,
      stream: streamRef.current,
      videoRef,
      attachVideo,
      start,
      stop,
      setDeviceId,
    }),
    [state, devices, deviceId, attachVideo, start, stop, setDeviceId]
  )
}
