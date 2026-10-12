import { useCallback, useEffect, useRef, useState } from 'react'
import { Link } from 'react-router-dom'
import type { NormalizedLandmark } from '@mediapipe/tasks-vision'
import { useCamera } from '../hooks/useCamera'
import { useFaceLandmarker } from '../hooks/useFaceLandmarker'
import { LandmarkFilter } from '../lib/landmarkFilter'
import {
  FOREHEAD_RING,
  LEFT_CHEEK_RING,
  RIGHT_CHEEK_RING,
  sampleCheekRgb,
} from '../lib/roi'
import {
  HeartRateEstimator,
  type HeartRateState,
} from '../lib/heartRate'
import Footer from '../components/Footer'
import HeartIcon from '../components/HeartIcon'
import SignalPlot from '../components/SignalPlot'
import FFTPlot from '../components/FFTPlot'

export type ScanPhase =
  | 'idle' // Live preview + Ready for Scan
  | 'recording' // Scanning... 30s countdown with live pulse
  | 'processing' // Recording complete -> Analyzing...
  | 'completed' // Official 30s results display
  | 'error' // Error card

export interface AnalysisResult {
  success: boolean
  heart_rate: number | null
  respiration_rate: number | null
  blood_pressure?: {
    systolic: number | null
    diastolic: number | null
    map: number | null
    pulse_pressure: number | null
    category: string
    category_color?: string
    vascular_stiffness?: string
    confidence?: number
  }
  systolic?: number | null
  diastolic?: number | null
  bp_category?: string
  hrv?: {
    value: number | null
    unit: string
    metric?: string
    sdnn?: number | null
    status?: string
  }
  blink_rate?: {
    value: number | null
    count?: number
    unit: string
    status?: string
  }
  face_movement?: {
    level: string
    score: number | null
    status?: string
  }
  signal_quality: 'Good' | 'Moderate' | 'Poor'
  snr: number
  duration: number
  message: string
  signal?: number[]
  frames_processed?: number
  tracking_ratio?: number
}

const SCAN_DURATION_SECONDS = 30

const INITIAL_HR_STATE: HeartRateState = {
  bpm: 0,
  snr: 0,
  valid: false,
  fps: 0,
  warmup: 0,
  signal: new Float64Array(0),
  fft: new Float64Array(0),
  freqsBpm: new Float64Array(0),
  stableBpm: 0,
  stableReadable: false,
  framesSinceValid: Infinity,
  method: 'POS',
  rr: 0,
  rrSnr: 0,
  rrValid: false,
  bp: {
    systolic: null,
    diastolic: null,
    map: null,
    pulsePressure: null,
    category: 'Unknown',
    categoryColor: 'gray',
    vascularStiffness: 'Unknown',
    confidence: 0,
  },
  spo2: 0,
  spo2Valid: false,
  spo2Trend: 'stable',
}

function getBestSupportedMimeType(): { mimeType: string; extension: string } {
  const candidates = [
    { mime: 'video/webm;codecs=vp9', ext: 'webm' },
    { mime: 'video/webm;codecs=vp8', ext: 'webm' },
    { mime: 'video/webm;codecs=h264', ext: 'webm' },
    { mime: 'video/webm', ext: 'webm' },
    { mime: 'video/mp4;codecs=avc1', ext: 'mp4' },
    { mime: 'video/mp4', ext: 'mp4' },
  ]
  if (typeof MediaRecorder === 'undefined') {
    return { mimeType: '', extension: 'webm' }
  }
  for (const candidate of candidates) {
    if (MediaRecorder.isTypeSupported(candidate.mime)) {
      return { mimeType: candidate.mime, extension: candidate.ext }
    }
  }
  return { mimeType: '', extension: 'webm' }
}

const FACE_OVAL_RING = [
  10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378,
  400, 377, 152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21,
  54, 103, 67, 109,
]

function drawPolygon(
  ctx: CanvasRenderingContext2D,
  indices: number[],
  lm: { x: number; y: number }[],
  w: number,
  h: number,
  fillColor = 'rgba(20, 184, 166, 0.22)',
  strokeColor = 'rgba(20, 184, 166, 0.95)'
) {
  ctx.beginPath()
  for (let i = 0; i < indices.length; i++) {
    const p = lm[indices[i]]
    if (!p) continue
    const x = p.x * w
    const y = p.y * h
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.closePath()
  ctx.fillStyle = fillColor
  ctx.fill()
  ctx.strokeStyle = strokeColor
  ctx.lineWidth = 2
  ctx.stroke()
}

function drawFaceOutline(
  ctx: CanvasRenderingContext2D,
  lm: { x: number; y: number }[],
  w: number,
  h: number
) {
  // 1. Draw smooth outer face contour
  ctx.beginPath()
  for (let i = 0; i < FACE_OVAL_RING.length; i++) {
    const p = lm[FACE_OVAL_RING[i]]
    if (!p) continue
    const x = p.x * w
    const y = p.y * h
    if (i === 0) ctx.moveTo(x, y)
    else ctx.lineTo(x, y)
  }
  ctx.closePath()
  ctx.strokeStyle = 'rgba(20, 184, 166, 0.45)'
  ctx.lineWidth = 1.5
  ctx.stroke()
}

export default function Reader() {
  const camera = useCamera()
  const landmarker = useFaceLandmarker()

  const overlayCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const samplingCanvasRef = useRef<HTMLCanvasElement | null>(null)
  const landmarkFilterRef = useRef<LandmarkFilter>(new LandmarkFilter())
  const estimatorRef = useRef<HeartRateEstimator>(new HeartRateEstimator())
  const lastTimeRef = useRef<number>(0)
  const lastLandmarksRef = useRef<{ landmarks: NormalizedLandmark[]; time: number } | null>(null)
  const lastValidBpmRef = useRef<number | null>(null)
  const scanSamplesRef = useRef<
    {
      t: number
      r: number
      g: number
      b: number
      ear?: number
      x?: number
      y?: number
      fw?: number
    }[]
  >([])

  const [faceDetected, setFaceDetected] = useState(false)
  const [hr, setHr] = useState<HeartRateState>(INITIAL_HR_STATE)
  const [phase, setPhase] = useState<ScanPhase>('idle')
  const [countdown, setCountdown] = useState<number>(SCAN_DURATION_SECONDS)
  const [progressPercent, setProgressPercent] = useState<number>(0)
  const [processingStatusText, setProcessingStatusText] =
    useState<string>('Recording complete')
  const [analysisResult, setAnalysisResult] = useState<AnalysisResult | null>(
    null
  )
  const [errorMessage, setErrorMessage] = useState<string | null>(null)
  const [backendReady, setBackendReady] = useState<boolean | null>(null)

  const mediaRecorderRef = useRef<MediaRecorder | null>(null)
  const recordedChunksRef = useRef<Blob[]>([])
  const timerIntervalRef = useRef<number | null>(null)
  const timeoutRef = useRef<number | null>(null)
  const isCancelledRef = useRef<boolean>(false)

  // Track latest valid BPM during scan
  useEffect(() => {
    if (hr.bpm && hr.bpm >= 45 && hr.bpm <= 190) {
      lastValidBpmRef.current = hr.bpm
    } else if (hr.stableBpm && hr.stableBpm >= 45 && hr.stableBpm <= 190) {
      lastValidBpmRef.current = hr.stableBpm
    }
  }, [hr.bpm, hr.stableBpm])

  // Check backend health periodically
  const checkBackendHealth = useCallback(async () => {
    try {
      const res = await fetch('/api/health')
      if (res.ok) {
        setBackendReady(true)
        return true
      }
      setBackendReady(false)
      return false
    } catch {
      setBackendReady(false)
      return false
    }
  }, [])

  useEffect(() => {
    checkBackendHealth()
    const timer = window.setInterval(checkBackendHealth, 4000)
    return () => clearInterval(timer)
  }, [checkBackendHealth])

  // Initialize camera preview on mount
  useEffect(() => {
    camera.start().catch((err) => {
      console.warn('Initial camera start waiting for user permission:', err)
    })
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  // Live MediaPipe face mesh ROI overlay + real-time pulse signal extraction loop
  useEffect(() => {
    if (landmarker.status !== 'ready') return
    const overlay = overlayCanvasRef.current
    const sampling = samplingCanvasRef.current
    if (!overlay || !sampling) return

    let animId: number

    const tick = () => {
      const video = camera.videoRef.current
      if (video && video.readyState >= 2 && video.videoWidth > 0) {
        const vw = video.videoWidth
        const vh = video.videoHeight
        if (overlay.width !== vw) overlay.width = vw
        if (overlay.height !== vh) overlay.height = vh
        if (sampling.width !== vw) sampling.width = vw
        if (sampling.height !== vh) sampling.height = vh

        const overlayCtx = overlay.getContext('2d')
        const samplingCtx = sampling.getContext('2d', { willReadFrequently: true })

        if (overlayCtx && samplingCtx) {
          const now = performance.now()
          const timestampMs = Math.max(lastTimeRef.current + 1, Math.round(now))
          lastTimeRef.current = timestampMs

          let currentFaces: NormalizedLandmark[][] | null = null

          try {
            const result = landmarker.landmarker.detectForVideo(video, timestampMs)
            if (result && result.faceLandmarks && result.faceLandmarks.length > 0) {
              currentFaces = result.faceLandmarks
              lastLandmarksRef.current = { landmarks: currentFaces[0], time: now }
            }
          } catch {
            // Guard against temporary detection frame glitches
          }

          // Use smoothed persistence if a single frame dropped
          let activeFaceLandmarks = currentFaces ? currentFaces[0] : null
          if (!activeFaceLandmarks && lastLandmarksRef.current && (now - lastLandmarksRef.current.time) < 400) {
            activeFaceLandmarks = lastLandmarksRef.current.landmarks
          }

          overlayCtx.clearRect(0, 0, vw, vh)

          if (activeFaceLandmarks) {
            setFaceDetected(true)
            const lm = landmarkFilterRef.current.stabilize(activeFaceLandmarks, now)

            // Draw full face guide + region polygons
            drawFaceOutline(overlayCtx, lm, vw, vh)
            drawPolygon(overlayCtx, LEFT_CHEEK_RING, lm, vw, vh)
            drawPolygon(overlayCtx, RIGHT_CHEEK_RING, lm, vw, vh)
            drawPolygon(overlayCtx, FOREHEAD_RING, lm, vw, vh)

            // Extract real-time pulse signal for live animations & instant readings
            samplingCtx.drawImage(video, 0, 0, vw, vh)
            const sample = sampleCheekRgb(lm, samplingCtx, vw, vh)
            if (sample) {
              const nextHr = estimatorRef.current.pushRegional(sample, now / 1000)
              setHr(nextHr)

              // Capture high-precision RGB frame sample for 30s analysis
              if (phase === 'recording') {
                let ear: number | undefined
                let noseX: number | undefined
                let noseY: number | undefined
                let fw: number | undefined

                if (lm && lm.length >= 468) {
                  const rH = Math.hypot(lm[33].x - lm[133].x, lm[33].y - lm[133].y)
                  const rV = (Math.hypot(lm[160].x - lm[144].x, lm[160].y - lm[144].y) + Math.hypot(lm[158].x - lm[153].x, lm[158].y - lm[153].y)) / (2 * Math.max(rH, 1e-5))
                  const lH = Math.hypot(lm[362].x - lm[263].x, lm[362].y - lm[263].y)
                  const lV = (Math.hypot(lm[385].x - lm[380].x, lm[385].y - lm[380].y) + Math.hypot(lm[387].x - lm[373].x, lm[387].y - lm[373].y)) / (2 * Math.max(lH, 1e-5))
                  ear = Math.round(((rV + lV) / 2) * 1000) / 1000
                  noseX = Math.round(lm[1].x * 1000) / 1000
                  noseY = Math.round(lm[1].y * 1000) / 1000
                  fw = Math.round(Math.hypot(lm[33].x - lm[263].x, lm[33].y - lm[263].y) * 1000) / 1000
                }

                scanSamplesRef.current.push({
                  t: now / 1000,
                  r: Math.round(sample.r * 100) / 100,
                  g: Math.round(sample.g * 100) / 100,
                  b: Math.round(sample.b * 100) / 100,
                  ear,
                  x: noseX,
                  y: noseY,
                  fw,
                })
              }
            }
          } else {
            setFaceDetected(false)
          }
        }
      }
      animId = requestAnimationFrame(tick)
    }

    animId = requestAnimationFrame(tick)
    return () => {
      cancelAnimationFrame(animId)
      const overlayCtx = overlay.getContext('2d')
      if (overlayCtx) overlayCtx.clearRect(0, 0, overlay.width, overlay.height)
    }
  }, [landmarker, camera.videoRef, camera.state, phase])

  // Clean up timers on unmount
  useEffect(() => {
    return () => {
      if (timerIntervalRef.current) clearInterval(timerIntervalRef.current)
      if (timeoutRef.current) clearTimeout(timeoutRef.current)
      if (
        mediaRecorderRef.current &&
        mediaRecorderRef.current.state === 'recording'
      ) {
        mediaRecorderRef.current.stop()
      }
    }
  }, [])

  const uploadAndAnalyze = useCallback(
    async (videoBlob: Blob, extension: string) => {
      setPhase('processing')
      setProcessingStatusText('Recording complete')

      await new Promise((r) => setTimeout(r, 400))
      setProcessingStatusText('Analyzing with CardioCam rPPG…')

      const formData = new FormData()
      const videoFile = new File(
        [videoBlob],
        `scan_recording_${Date.now()}.${extension}`,
        {
          type: videoBlob.type,
        }
      )
      formData.append('file', videoFile)
      formData.append('duration', '30')

      const currentBpm = lastValidBpmRef.current
      if (currentBpm && currentBpm >= 45 && currentBpm <= 190) {
        formData.append('client_bpm', String(currentBpm))
      }

      if (scanSamplesRef.current.length >= 30) {
        formData.append('samples', JSON.stringify(scanSamplesRef.current))
      }

      try {
        const response = await fetch('/api/analyze-video', {
          method: 'POST',
          body: formData,
        })

        if (!response.ok) {
          const errData = await response.json().catch(() => null)
          throw new Error(
            errData?.detail || `Server responded with status ${response.status}`
          )
        }

        const data: AnalysisResult = await response.json()
        setAnalysisResult(data)
        setPhase('completed')
      } catch (err) {
        const msg =
          err instanceof Error
            ? err.message
            : 'Unable to analyze video. Please verify the Python backend is running.'
        setErrorMessage(msg)
        setPhase('error')
      }
    },
    []
  )

  const stopRecordingAndProcess = useCallback(() => {
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }

    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === 'recording'
    ) {
      mediaRecorderRef.current.stop()
    }

    // Stop camera tracks once recording is finished to turn off camera LED
    camera.stop()
  }, [camera])

  const startScan = async () => {
    setErrorMessage(null)
    setAnalysisResult(null)
    isCancelledRef.current = false
    recordedChunksRef.current = []
    scanSamplesRef.current = []
    lastValidBpmRef.current = null
    estimatorRef.current.reset()

    // 1. Verify backend health before beginning 30s scan
    const isUp = await checkBackendHealth()
    if (!isUp) {
      setErrorMessage(
        'Python backend server is not running on port 8000. Please start it using: cd backend && source venv/Scripts/activate && python -m uvicorn app.main:app --reload'
      )
      setPhase('error')
      return
    }

    // 2. Ensure camera stream is active and alive
    let activeStream: MediaStream | null = null
    if (
      camera.stream &&
      camera.stream.active &&
      camera.stream.getVideoTracks().some((t) => t.readyState === 'live')
    ) {
      activeStream = camera.stream
    } else {
      try {
        activeStream = await camera.start(camera.deviceId || undefined)
      } catch (err) {
        setErrorMessage(
          err instanceof Error
            ? err.message
            : 'Unable to access camera. Please check permissions.'
        )
        setPhase('error')
        return
      }
    }

    if (!activeStream || !activeStream.active) {
      setErrorMessage('Camera stream is not ready. Please click "Grant Camera Permission" and try again.')
      setPhase('error')
      return
    }

    // 3. Initialize MediaRecorder with fallback
    const { mimeType, extension } = getBestSupportedMimeType()
    let recorder: MediaRecorder
    try {
      const options: MediaRecorderOptions = mimeType ? { mimeType } : {}
      recorder = new MediaRecorder(activeStream, options)
    } catch {
      try {
        recorder = new MediaRecorder(activeStream)
      } catch {
        setErrorMessage(
          'MediaRecorder failed to initialize. Please check browser camera support.'
        )
        setPhase('error')
        return
      }
    }

    mediaRecorderRef.current = recorder

    recorder.ondataavailable = (event) => {
      if (event.data && event.data.size > 0) {
        recordedChunksRef.current.push(event.data)
      }
    }

    recorder.onstop = () => {
      if (isCancelledRef.current) {
        setPhase('idle')
        return
      }
      const blob = new Blob(recordedChunksRef.current, {
        type: recorder.mimeType || 'video/webm',
      })
      uploadAndAnalyze(blob, extension)
    }

    // Start recording with 1-second timeslices
    recorder.start(1000)
    setPhase('recording')
    setCountdown(SCAN_DURATION_SECONDS)
    setProgressPercent(0)

    const startTime = Date.now()
    const totalMs = SCAN_DURATION_SECONDS * 1000

    timerIntervalRef.current = window.setInterval(() => {
      const elapsedMs = Date.now() - startTime
      const remainingSeconds = Math.max(
        0,
        Math.ceil((totalMs - elapsedMs) / 1000)
      )
      const percent = Math.min(100, (elapsedMs / totalMs) * 100)

      setCountdown(remainingSeconds)
      setProgressPercent(percent)

      if (elapsedMs >= totalMs) {
        stopRecordingAndProcess()
      }
    }, 100)

    timeoutRef.current = window.setTimeout(() => {
      stopRecordingAndProcess()
    }, totalMs)
  }

  const cancelScan = () => {
    isCancelledRef.current = true
    if (timerIntervalRef.current) {
      clearInterval(timerIntervalRef.current)
      timerIntervalRef.current = null
    }
    if (timeoutRef.current) {
      clearTimeout(timeoutRef.current)
      timeoutRef.current = null
    }
    if (
      mediaRecorderRef.current &&
      mediaRecorderRef.current.state === 'recording'
    ) {
      mediaRecorderRef.current.stop()
    }
    lastLandmarksRef.current = null
    lastTimeRef.current = 0
    scanSamplesRef.current = []
    setPhase('idle')
    setCountdown(SCAN_DURATION_SECONDS)
    setProgressPercent(0)
    camera.start(camera.deviceId || undefined).catch(() => {})
  }

  const resetToNewScan = () => {
    setAnalysisResult(null)
    setErrorMessage(null)
    setCountdown(SCAN_DURATION_SECONDS)
    setProgressPercent(0)
    estimatorRef.current.reset()
    landmarkFilterRef.current.reset()
    lastLandmarksRef.current = null
    lastTimeRef.current = 0
    scanSamplesRef.current = []
    setHr(INITIAL_HR_STATE)
    setPhase('idle')
    camera.start(camera.deviceId || undefined).catch(() => {})
  }

  const isCameraRunning = camera.state.status === 'running'

  // During scanning, we do not show premature fluctuating numbers.
  // Final accurate reading is computed and shown on the 30s completed summary screen.
  const statusLine = (() => {
    if (landmarker.status === 'loading') return 'Loading face tracking model…'
    if (landmarker.status === 'error') return `Model error: ${landmarker.error}`
    if (camera.state.status === 'error') return `Camera error: ${camera.state.error}`
    if (!isCameraRunning) return 'Camera stopped'
    if (!faceDetected) return 'Position your face in the camera frame…'
    if (phase === 'recording')
      return `Scanning face pulse: ${countdown}s remaining · Keep still`
    return 'Ready to scan · Click Start 30s Scan below'
  })()

  return (
    <div className="min-h-full flex flex-col bg-canvas text-ink">
      {/* Header */}
      <header className="flex items-center justify-between px-4 md:px-8 py-4 border-b border-ink/5 bg-white/70 backdrop-blur-md sticky top-0 z-30">
        <Link
          to="/"
          className="inline-flex items-center gap-2 text-ink2 hover:text-ink text-sm font-medium transition"
        >
          ← <span>Home</span>
        </Link>
        <div className="inline-flex items-center gap-2 text-heart font-semibold tracking-tight text-lg">
          <HeartIcon
            bpm={
              phase === 'completed' && analysisResult?.heart_rate
                ? analysisResult.heart_rate
                : phase === 'recording'
                  ? 75
                  : 0
            }
            size={20}
          />
          CardioCam
        </div>
        <div className="flex items-center gap-2">
          <div
            className={`inline-flex items-center gap-1.5 text-xs font-medium px-2.5 py-1 rounded-full ${
              backendReady === true
                ? 'bg-emerald-50 text-emerald-700'
                : backendReady === false
                  ? 'bg-rose-50 text-rose-700'
                  : 'bg-ink/5 text-ink2'
            }`}
          >
            <span
              className={`w-2 h-2 rounded-full ${
                backendReady === true
                  ? 'bg-emerald-500'
                  : backendReady === false
                    ? 'bg-rose-500 animate-pulse'
                    : 'bg-amber-400'
              }`}
            />
            {backendReady === true
              ? 'Backend Ready'
              : backendReady === false
                ? 'Backend Offline'
                : 'Connecting...'}
          </div>
          <div className="text-xs font-medium px-2.5 py-1 rounded-full bg-ink/5 text-ink2 hidden sm:block">
            30s Optical Vitals
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="flex-1 px-4 md:px-8 py-4 md:py-8 max-w-6xl mx-auto w-full">
        {phase === 'completed' && analysisResult ? (
          /* OFFICIAL 30-SECOND COMPLETED RESULTS VIEW */
          <div className="flex flex-col gap-6 max-w-4xl mx-auto">
            {/* Header Banner */}
            <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 bg-white rounded-3xl p-6 shadow-sm border border-ink/5">
              <div>
                <div
                  className={`inline-flex items-center gap-2 px-3 py-1 rounded-full font-semibold text-xs mb-1 ${
                    analysisResult.success && analysisResult.heart_rate !== null
                      ? 'bg-emerald-50 text-emerald-600'
                      : 'bg-amber-50 text-amber-700'
                  }`}
                >
                  <span>{analysisResult.success && analysisResult.heart_rate !== null ? '✓' : '⚠️'}</span>{' '}
                  {analysisResult.success && analysisResult.heart_rate !== null
                    ? '30s Scan Analysis Complete'
                    : 'Analysis Inconclusive'}
                </div>
                <h2 className="text-2xl font-bold tracking-tight text-ink">
                  Your Vitals Summary
                </h2>
                <p className="text-xs text-ink2 mt-0.5">
                  Processed from 30 seconds of facial rPPG optical recording via Python backend.
                </p>
              </div>
              <div className="flex items-center gap-3">
                <button
                  onClick={resetToNewScan}
                  className="px-5 py-2.5 rounded-full bg-heart text-white text-sm font-semibold hover:bg-heart2 shadow-sm transition"
                >
                  Start New Scan
                </button>
              </div>
            </div>

            {/* Inconclusive Guidance Card if Heart Rate wasn't recovered */}
            {(!analysisResult.success || analysisResult.heart_rate === null) && (
              <div className="bg-amber-50/80 border border-amber-200 rounded-3xl p-6 text-amber-900 flex flex-col sm:flex-row items-start sm:items-center justify-between gap-4">
                <div>
                  <h4 className="font-semibold text-sm mb-1 text-amber-900">
                    Facial Pulse Signal Was Low
                  </h4>
                  <p className="text-xs text-amber-800 leading-relaxed max-w-xl">
                    {analysisResult.message ||
                      'Low optical contrast or motion detected. Ensure soft front-facing light is on your face and remain still.'}
                  </p>
                </div>
                <button
                  onClick={resetToNewScan}
                  className="px-4 py-2 bg-amber-600 text-white rounded-full text-xs font-semibold hover:bg-amber-700 transition shrink-0"
                >
                  Retry with Better Lighting
                </button>
              </div>
            )}

            {/* Primary Vitals Grid (Heart Rate, Blood Pressure, HRV) */}
            <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
              {/* 1. Heart Rate Card */}
              <div className="rounded-3xl bg-white shadow-sm border border-ink/5 p-8 flex flex-col items-center text-center justify-between relative overflow-hidden">
                <div className="w-full flex flex-col items-center">
                  <div className="w-12 h-12 rounded-2xl bg-rose-50 text-heart flex items-center justify-center mb-3">
                    <HeartIcon
                      bpm={analysisResult.heart_rate || 75}
                      size={26}
                    />
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-ink2 mb-1">
                    Heart Rate
                  </div>
                  <div className="text-6xl sm:text-7xl font-bold tracking-tight text-heart tabular-nums my-1">
                    {analysisResult.heart_rate !== null
                      ? analysisResult.heart_rate
                      : '--'}
                  </div>
                  <div className="text-xs font-medium text-ink2/70 uppercase">
                    Beats per minute (BPM)
                  </div>
                </div>

                <div className="mt-6 pt-3 border-t border-ink/5 w-full">
                  <span
                    className={`inline-block text-xs font-semibold px-3.5 py-1 rounded-full ${
                      analysisResult.heart_rate &&
                      analysisResult.heart_rate >= 60 &&
                      analysisResult.heart_rate <= 100
                        ? 'bg-emerald-50 text-emerald-700 border border-emerald-200/60'
                        : analysisResult.heart_rate !== null
                          ? 'bg-amber-50 text-amber-700 border border-amber-200/60'
                          : 'bg-ink/5 text-ink2'
                    }`}
                  >
                    {analysisResult.heart_rate &&
                    analysisResult.heart_rate >= 60 &&
                    analysisResult.heart_rate <= 100
                      ? 'Normal Resting Heart Rate'
                      : analysisResult.heart_rate !== null
                        ? 'Measured Heart Rate'
                        : 'Low Optical Contrast'}
                  </span>
                </div>
              </div>

              {/* 2. Blood Pressure Card */}
              <div className="rounded-3xl bg-white shadow-sm border border-ink/5 p-8 flex flex-col items-center text-center justify-between relative overflow-hidden">
                <div className="w-full flex flex-col items-center">
                  <div className="w-12 h-12 rounded-2xl bg-indigo-50 text-indigo-600 flex items-center justify-center mb-3 text-2xl font-bold">
                    🩺
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-ink2 mb-1">
                    Blood Pressure
                  </div>

                  {/* SBP / DBP Display */}
                  <div className="flex items-baseline justify-center gap-1.5 my-1">
                    <span className="text-5xl sm:text-6xl font-bold tracking-tight text-ink tabular-nums">
                      {analysisResult.systolic ?? analysisResult.blood_pressure?.systolic ?? '--'}
                    </span>
                    <span className="text-3xl font-light text-ink/40">/</span>
                    <span className="text-5xl sm:text-6xl font-bold tracking-tight text-ink tabular-nums">
                      {analysisResult.diastolic ?? analysisResult.blood_pressure?.diastolic ?? '--'}
                    </span>
                    <span className="text-xs font-semibold text-ink2/70 uppercase ml-1">
                      mmHg
                    </span>
                  </div>

                  {/* AHA Category Badge */}
                  <div className="mt-2">
                    {(() => {
                      const cat = analysisResult.bp_category || analysisResult.blood_pressure?.category || 'Normal'
                      const isNormal = cat === 'Normal'
                      const isElevated = cat === 'Elevated'
                      const isStage1 = cat.includes('Stage 1')
                      const isStage2 = cat.includes('Stage 2')

                      return (
                        <span
                          className={`inline-flex items-center gap-1.5 text-xs font-semibold px-3 py-1 rounded-full border ${
                            isNormal
                              ? 'bg-emerald-50 text-emerald-700 border-emerald-200'
                              : isElevated
                                ? 'bg-amber-50 text-amber-700 border-amber-200'
                                : isStage1
                                  ? 'bg-orange-50 text-orange-700 border-orange-200'
                                  : isStage2
                                    ? 'bg-rose-50 text-rose-700 border-rose-200'
                                    : 'bg-ink/5 text-ink2 border-transparent'
                          }`}
                        >
                          <span
                            className={`w-2 h-2 rounded-full ${
                              isNormal
                                ? 'bg-emerald-500'
                                : isElevated
                                  ? 'bg-amber-500'
                                  : isStage1
                                    ? 'bg-orange-500'
                                    : isStage2
                                      ? 'bg-rose-500'
                                      : 'bg-ink/30'
                            }`}
                          />
                          {cat}
                        </span>
                      )
                    })()}
                  </div>
                </div>

                {/* Sub-strip: MAP & Pulse Pressure */}
                <div className="mt-6 pt-3 border-t border-ink/5 w-full grid grid-cols-2 gap-2 text-center text-xs text-ink2">
                  <div>
                    <span className="block font-bold text-ink text-sm">
                      {analysisResult.blood_pressure?.map ??
                        (analysisResult.systolic && analysisResult.diastolic
                          ? Math.round((analysisResult.systolic + 2 * analysisResult.diastolic) / 3)
                          : '--')}
                      <span className="text-[10px] font-normal text-ink2/70 ml-0.5">mmHg</span>
                    </span>
                    <span className="text-[10px] text-ink2/60">Mean Arterial (MAP)</span>
                  </div>
                  <div className="border-l border-ink/5 pl-2">
                    <span className="block font-bold text-ink text-sm">
                      {analysisResult.blood_pressure?.pulse_pressure ??
                        (analysisResult.systolic && analysisResult.diastolic
                          ? analysisResult.systolic - analysisResult.diastolic
                          : '--')}
                      <span className="text-[10px] font-normal text-ink2/70 ml-0.5">mmHg</span>
                    </span>
                    <span className="text-[10px] text-ink2/60">Pulse Pressure</span>
                  </div>
                </div>
              </div>

              {/* 3. HRV Card */}
              <div className="rounded-3xl bg-white shadow-sm border border-ink/5 p-8 flex flex-col items-center text-center justify-between relative overflow-hidden">
                <div className="w-full flex flex-col items-center">
                  <div className="w-12 h-12 rounded-2xl bg-rose-50 text-rose-600 flex items-center justify-center mb-3 text-2xl font-bold">
                    🫀
                  </div>
                  <div className="text-xs font-semibold uppercase tracking-wider text-ink2 mb-1">
                    HRV (RMSSD)
                  </div>

                  {analysisResult.hrv?.value !== null && analysisResult.hrv?.value !== undefined ? (
                    <>
                      <div className="flex items-baseline justify-center gap-1.5 my-1">
                        <span className="text-6xl sm:text-7xl font-bold tracking-tight text-ink tabular-nums">
                          {analysisResult.hrv.value}
                        </span>
                        <span className="text-xs font-semibold text-ink2/70 uppercase">
                          {analysisResult.hrv.unit || 'ms'}
                        </span>
                      </div>
                      <div className="text-xs font-medium text-ink2/70 uppercase mt-0.5">
                        {analysisResult.hrv.metric || 'RMSSD'}
                      </div>
                    </>
                  ) : (
                    <div className="my-3 py-1">
                      <span className="text-sm font-semibold text-ink2 bg-ink/5 px-3 py-1 rounded-full">
                        Insufficient signal
                      </span>
                    </div>
                  )}
                </div>

                <div className="mt-6 pt-3 border-t border-ink/5 w-full text-center">
                  <span className="inline-block text-xs font-semibold px-3.5 py-1 rounded-full bg-emerald-50 text-emerald-700 border border-emerald-200/60">
                    {analysisResult.hrv?.value !== null && analysisResult.hrv?.value !== undefined
                      ? `Pulse Variation · SDNN ${analysisResult.hrv.sdnn ?? analysisResult.hrv.value} ms`
                      : 'Requires steady facial lighting'}
                  </span>
                </div>
              </div>
            </div>
          </div>
        ) : phase === 'error' ? (
          /* ERROR VIEW */
          <div className="max-w-md mx-auto my-12 bg-white rounded-3xl p-8 shadow-sm border border-rose-100 flex flex-col items-center text-center">
            <div className="w-14 h-14 rounded-full bg-rose-50 text-rose-500 flex items-center justify-center text-2xl font-bold mb-4">
              !
            </div>
            <h3 className="text-xl font-bold text-ink mb-2">Scan Failed</h3>
            <p className="text-sm text-ink2 mb-6 leading-relaxed">
              {errorMessage || 'An error occurred during video analysis.'}
            </p>

            <button
              onClick={resetToNewScan}
              className="px-6 py-3 rounded-full bg-heart text-white text-sm font-semibold hover:bg-heart2 transition shadow-sm"
            >
              Try Again
            </button>
          </div>
        ) : (
          /* ACTIVE SCANNING, RECORDING & LIVE PULSE ANIMATION VIEW */
          <div className="grid grid-cols-1 lg:grid-cols-[minmax(0,1fr)_380px] gap-6 items-start">
            {/* Left Column — Video Container with Authentic MediaPipe Face Framing */}
            <div className="flex flex-col gap-3">
              <div className="relative w-full rounded-2xl overflow-hidden bg-black shadow-lg aspect-[4/3] max-h-[48vh] lg:max-h-none lg:aspect-video">
                <video
                  ref={camera.attachVideo}
                  playsInline
                  muted
                  className="absolute inset-0 w-full h-full object-cover scale-x-[-1]"
                />
                <canvas
                  ref={overlayCanvasRef}
                  className="absolute inset-0 w-full h-full object-cover scale-x-[-1] pointer-events-none"
                />
                <canvas ref={samplingCanvasRef} className="hidden" />

                {!isCameraRunning && (
                  <div className="absolute inset-0 flex flex-col items-center justify-center text-white/80 text-center px-6 min-h-[12rem] bg-black/85">
                    {camera.state.status === 'error' ? (
                      <>
                        <p className="text-rose-400 font-semibold mb-2">Camera Unavailable</p>
                        <p className="text-xs text-white/70 max-w-xs mb-4">{camera.state.error}</p>
                        <button
                          onClick={() => camera.start()}
                          className="text-xs bg-white text-ink px-4 py-2 rounded-full font-medium hover:bg-white/90 shadow-sm"
                        >
                          Grant Camera Permission
                        </button>
                      </>
                    ) : (
                      <p className="text-sm">Starting camera preview…</p>
                    )}
                  </div>
                )}

                {/* Live Face Tracked Pill */}
                {faceDetected && isCameraRunning && (
                  <div className="absolute top-3 left-3 rounded-full bg-black/60 text-white text-xs px-3 py-1 backdrop-blur flex items-center gap-1.5 shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-emerald-400"></span>
                    {phase === 'recording' ? 'Scanning · keep still' : 'Face tracked'}
                  </div>
                )}

                {/* Face prompt pill if face not in frame */}
                {!faceDetected && isCameraRunning && (
                  <div className="absolute top-3 left-3 rounded-full bg-black/60 text-white/80 text-xs px-3 py-1 backdrop-blur flex items-center gap-1.5 shadow-sm">
                    <span className="w-2 h-2 rounded-full bg-amber-400 animate-pulse"></span>
                    Position face in frame
                  </div>
                )}

                {/* Prominent Recording Timer Pill in Top-Right during recording */}
                {phase === 'recording' && (
                  <div className="absolute top-3 right-3 rounded-full bg-black/75 text-white text-xs px-3.5 py-1.5 backdrop-blur border border-white/10 flex items-center gap-2 font-mono shadow-md">
                    <span className="w-2 h-2 rounded-full bg-heart animate-ping"></span>
                    <span className="font-semibold text-rose-300">{countdown}s</span>
                    <span className="text-[10px] text-white/60">/ 30s</span>
                  </div>
                )}
              </div>
            </div>

            {/* Right Column — Vitals Card, Live Pulse Waveform, Live Spectrum & 30s Scan Controls */}
            <aside className="flex flex-col gap-4">
              {backendReady === false && (
                <div className="p-3.5 rounded-2xl bg-rose-50 border border-rose-200 text-xs text-rose-800 flex items-start gap-2.5">
                  <span className="text-base">⚠️</span>
                  <div>
                    <div className="font-semibold text-rose-900 mb-0.5">
                      Python Backend is Offline
                    </div>
                    <div>
                      To analyze 30s recordings, start backend:
                      <div className="font-mono bg-white/90 px-2 py-0.5 rounded text-[11px] text-ink mt-1 border border-rose-200">
                        cd backend && python -m uvicorn app.main:app --port 8000
                      </div>
                    </div>
                  </div>
                </div>
              )}

              {/* Live Vitals Card with Pulsing Heart Icon */}
              <div className="rounded-2xl bg-white shadow-sm border border-ink/5 p-5">
                <div className="text-center py-2">
                  <div
                    className={`inline-flex items-center justify-center gap-3 ${
                      phase === 'recording' ? 'text-heart' : 'text-ink2/40'
                    }`}
                  >
                    <HeartIcon
                      bpm={phase === 'recording' ? 75 : 0}
                      size={32}
                      className="self-center"
                    />
                    <div className="text-4xl font-semibold tracking-tight text-ink tabular-nums">
                      {phase === 'recording' ? 'Scanning…' : '--'}
                    </div>
                  </div>
                  <div className="mt-2 text-xs text-ink2 uppercase tracking-wider font-medium">
                    {phase === 'recording'
                      ? 'Collecting 30s Optical Pulse'
                      : 'Heart Rate (BPM)'}
                  </div>
                </div>

                <div className="mt-3 pt-3 border-t border-ink/5 text-[11px] text-ink2 text-center min-h-[1.2em]">
                  {statusLine}
                </div>
              </div>

              {/* Live Pulse Signal Waveform Plot */}
              <div className="rounded-2xl bg-white shadow-sm border border-ink/5 p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs font-medium text-ink2 uppercase tracking-wider">
                    Pulse Signal
                  </div>
                  <span className="text-[10px] text-teal-600 font-medium px-2 py-0.5 rounded bg-teal-50">
                    Live Optical BVP
                  </span>
                </div>
                <SignalPlot signal={hr.signal} height={80} />
              </div>

              {/* Live Frequency Spectrum Plot */}
              <div className="rounded-2xl bg-white shadow-sm border border-ink/5 p-4">
                <div className="flex items-center justify-between mb-2">
                  <div className="text-xs font-medium text-ink2 uppercase tracking-wider">
                    Spectrum
                  </div>
                  <span className="text-[10px] text-rose-500 font-medium px-2 py-0.5 rounded bg-rose-50">
                    FFT Cardiac Band
                  </span>
                </div>
                <FFTPlot
                  fft={hr.fft}
                  freqsBpm={hr.freqsBpm}
                  peakBpm={hr.valid ? hr.bpm : 0}
                  height={80}
                />
              </div>

              {/* 30-Second Recording Trigger & Progress Action Box */}
              {phase === 'idle' && (
                <div className="bg-white rounded-2xl p-5 shadow-sm border border-ink/5">
                  <button
                    onClick={startScan}
                    disabled={camera.state.status === 'requesting'}
                    className="w-full py-3.5 px-6 rounded-xl bg-heart text-white font-semibold text-sm shadow-md shadow-heart/20 hover:bg-heart2 active:scale-[0.99] transition duration-150 flex items-center justify-center gap-2 disabled:opacity-50"
                  >
                    <HeartIcon bpm={75} size={16} />
                    Start 30s Official Scan
                  </button>
                  <p className="text-center text-[10px] text-ink2/60 mt-2">
                    Records 30s webcam video for deep CardioCam rPPG analysis on Python backend.
                  </p>
                </div>
              )}

              {phase === 'recording' && (
                <div className="bg-white rounded-2xl p-5 shadow-sm border border-heart/20 flex flex-col gap-3 text-center">
                  <div className="flex items-center justify-between">
                    <span className="text-xs font-semibold text-heart uppercase tracking-wider flex items-center gap-1.5">
                      <span className="w-2 h-2 rounded-full bg-heart animate-ping"></span>
                      Recording 30s Scan
                    </span>
                    <span className="text-sm font-bold font-mono text-ink">
                      {countdown}s remaining
                    </span>
                  </div>

                  <div className="w-full bg-ink/10 h-2.5 rounded-full overflow-hidden p-0.5">
                    <div
                      className="bg-heart h-full rounded-full transition-all duration-150 ease-linear shadow-sm"
                      style={{ width: `${progressPercent}%` }}
                    ></div>
                  </div>

                  <div className="flex items-center justify-between text-[11px] text-ink2 mt-1">
                    <span>Progress: {Math.round(progressPercent)}%</span>
                    <button
                      onClick={cancelScan}
                      className="text-ink2 hover:text-rose-600 font-medium transition underline"
                    >
                      Cancel Scan
                    </button>
                  </div>
                </div>
              )}

              {phase === 'processing' && (
                <div className="bg-white rounded-2xl p-5 shadow-sm border border-ink/5 flex items-center gap-3">
                  <div className="relative w-8 h-8 shrink-0 flex items-center justify-center">
                    <div className="absolute inset-0 rounded-full border-2 border-heart/20"></div>
                    <div className="absolute inset-0 rounded-full border-2 border-heart border-t-transparent animate-spin"></div>
                    <HeartIcon bpm={90} size={14} className="text-heart" />
                  </div>
                  <div className="text-left">
                    <h4 className="text-xs font-bold text-ink">
                      {processingStatusText}
                    </h4>
                    <p className="text-[10px] text-ink2 mt-0.5">
                      Fast rPPG spectral processing on Python backend…
                    </p>
                  </div>
                </div>
              )}
            </aside>
          </div>
        )}
      </main>

      <Footer />
    </div>
  )
}
