import { useEffect, useRef, useState } from 'react'
import { FaceLandmarker, FilesetResolver } from '@mediapipe/tasks-vision'

export type LandmarkerState =
  | { status: 'loading' }
  | { status: 'ready'; landmarker: FaceLandmarker }
  | { status: 'error'; error: string }

// Load MediaPipe Tasks Vision once per app lifetime. The WASM bundle is
// fetched from jsdelivr (~500KB, cached after first hit) and the face
// landmarker model is served from the app's own /models/ path so we
// don't depend on Google's CDN.
export function useFaceLandmarker() {
  const [state, setState] = useState<LandmarkerState>({ status: 'loading' })
  const mountedRef = useRef(true)

  useEffect(() => {
    mountedRef.current = true
    let created: FaceLandmarker | null = null
    ;(async () => {
      try {
        const vision = await FilesetResolver.forVisionTasks(
          'https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.34/wasm'
        )
        try {
          created = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`,
              delegate: 'GPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          })
        } catch (gpuErr) {
          console.warn('MediaPipe GPU delegate failed, falling back to CPU:', gpuErr)
          created = await FaceLandmarker.createFromOptions(vision, {
            baseOptions: {
              modelAssetPath: `${import.meta.env.BASE_URL}models/face_landmarker.task`,
              delegate: 'CPU',
            },
            runningMode: 'VIDEO',
            numFaces: 1,
            outputFaceBlendshapes: false,
            outputFacialTransformationMatrixes: false,
          })
        }
        if (mountedRef.current) {
          setState({ status: 'ready', landmarker: created })
        } else {
          created.close()
        }
      } catch (err) {
        const message =
          err instanceof Error ? err.message : 'Face landmarker failed to load.'
        if (mountedRef.current) setState({ status: 'error', error: message })
      }
    })()

    return () => {
      mountedRef.current = false
      // If the model loaded after unmount, close it. Otherwise, close it now.
      setState((prev) => {
        if (prev.status === 'ready') prev.landmarker.close()
        return prev
      })
      if (created) {
        // Might double-close if the setter above already ran; MediaPipe is
        // idempotent on close, so safe.
        try {
          created.close()
        } catch {
          // ignore
        }
      }
    }
  }, [])

  return state
}
