import { useEffect, useRef } from 'react'

interface Props {
  fft: Float64Array
  freqsBpm: Float64Array
  /** Current peak BPM, so we can mark it. */
  peakBpm: number
  height?: number
}

// FFT magnitude plot — same hand-rolled canvas approach as SignalPlot.
// A filled area plot reads better for a spectrum than a line, and we
// draw a small vertical marker at the current argmax so users can see
// the detector's decision rather than just the final BPM number.
export default function FFTPlot({ fft, freqsBpm, peakBpm, height = 120 }: Props) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null)

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const rect = canvas.getBoundingClientRect()
    const dpr = window.devicePixelRatio || 1
    canvas.width = rect.width * dpr
    canvas.height = rect.height * dpr
    const ctx = canvas.getContext('2d')
    if (!ctx) return
    ctx.scale(dpr, dpr)
    ctx.clearRect(0, 0, rect.width, rect.height)

    if (fft.length < 2 || freqsBpm.length < 2) {
      drawPlaceholder(ctx, rect.width, rect.height)
      return
    }

    // Axis bounds — pin to the 50–180 BPM band we actually analyze.
    const xMin = freqsBpm[0]
    const xMax = freqsBpm[freqsBpm.length - 1]
    let peakVal = 1e-9
    for (let i = 0; i < fft.length; i++) if (fft[i] > peakVal) peakVal = fft[i]
    const topPad = 6
    const h = rect.height
    const w = rect.width
    const toX = (bpm: number) => ((bpm - xMin) / (xMax - xMin)) * w
    const toY = (v: number) => h - (v / peakVal) * (h - topPad)

    // Filled area under the spectrum.
    ctx.fillStyle = 'rgba(244, 63, 94, 0.15)'
    ctx.beginPath()
    ctx.moveTo(toX(freqsBpm[0]), h)
    for (let i = 0; i < fft.length; i++) {
      ctx.lineTo(toX(freqsBpm[i]), toY(fft[i]))
    }
    ctx.lineTo(toX(freqsBpm[freqsBpm.length - 1]), h)
    ctx.closePath()
    ctx.fill()

    // Outline trace on top.
    ctx.strokeStyle = 'rgba(244, 63, 94, 0.9)'
    ctx.lineWidth = 1.5
    ctx.beginPath()
    for (let i = 0; i < fft.length; i++) {
      const x = toX(freqsBpm[i])
      const y = toY(fft[i])
      if (i === 0) ctx.moveTo(x, y)
      else ctx.lineTo(x, y)
    }
    ctx.stroke()

    // Vertical marker at the detected peak.
    if (peakBpm > 0) {
      ctx.strokeStyle = 'rgba(31, 41, 55, 0.55)'
      ctx.setLineDash([3, 3])
      ctx.beginPath()
      ctx.moveTo(toX(peakBpm), 0)
      ctx.lineTo(toX(peakBpm), h)
      ctx.stroke()
      ctx.setLineDash([])

      ctx.fillStyle = '#1F2937'
      ctx.font = '11px Inter, system-ui, sans-serif'
      ctx.textAlign = 'center'
      ctx.fillText(`${peakBpm.toFixed(1)} BPM`, toX(peakBpm), 12)
    }

    // Band x-axis ticks at 60/90/120/150.
    ctx.fillStyle = 'rgba(75, 85, 99, 0.5)'
    ctx.font = '10px Inter, system-ui, sans-serif'
    ctx.textAlign = 'center'
    for (const bpm of [60, 90, 120, 150]) {
      if (bpm < xMin || bpm > xMax) continue
      ctx.fillText(String(bpm), toX(bpm), h - 2)
    }
  }, [fft, freqsBpm, peakBpm])

  return (
    <canvas
      ref={canvasRef}
      style={{ height, width: '100%', display: 'block' }}
    />
  )
}

function drawPlaceholder(ctx: CanvasRenderingContext2D, w: number, h: number) {
  ctx.fillStyle = 'rgba(75, 85, 99, 0.35)'
  ctx.font = '12px Inter, system-ui, sans-serif'
  ctx.textAlign = 'center'
  ctx.textBaseline = 'middle'
  ctx.fillText('Waiting for signal…', w / 2, h / 2)
}
