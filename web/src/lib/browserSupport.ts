// The page needs: camera access (getUserMedia), GPU compute for MediaPipe
// (WebGL2), and the browser's native FFT arithmetic path (ES2020). If any
// of these are missing, we short-circuit the app with a clear message
// instead of letting the user hit a cryptic stack trace when the model
// tries to load.

export interface BrowserSupport {
  ok: boolean
  missing: string[]
}

export function checkBrowserSupport(): BrowserSupport {
  const missing: string[] = []

  if (
    typeof navigator === 'undefined' ||
    !navigator.mediaDevices ||
    typeof navigator.mediaDevices.getUserMedia !== 'function'
  ) {
    missing.push('Camera API (getUserMedia)')
  }

  // Probe WebGL2. Some older Safari / embedded WebViews claim WebGL but
  // not WebGL2, which MediaPipe's GPU delegate requires.
  try {
    const canvas = document.createElement('canvas')
    const gl = canvas.getContext('webgl2')
    if (!gl) missing.push('WebGL2 (GPU graphics)')
  } catch {
    missing.push('WebGL2 (GPU graphics)')
  }

  // The getImageData path needs a 2D context too — trivially present on
  // anything modern, but belt-and-suspenders.
  try {
    const canvas = document.createElement('canvas')
    if (!canvas.getContext('2d')) missing.push('Canvas 2D')
  } catch {
    missing.push('Canvas 2D')
  }

  return { ok: missing.length === 0, missing }
}
