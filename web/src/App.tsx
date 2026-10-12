import { useMemo } from 'react'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import Landing from './pages/Landing'
import Reader from './pages/Reader'
import UnsupportedBrowser from './components/UnsupportedBrowser'
import { checkBrowserSupport } from './lib/browserSupport'

// GitHub Pages serves the app at /cardiocam/, so the router basename must
// match. import.meta.env.BASE_URL is '/cardiocam/' in production builds
// (set by vite.config.ts) and '/' during `npm run dev`.
const basename = import.meta.env.BASE_URL.replace(/\/$/, '') || '/'

export default function App() {
  // Probe once on mount — no point re-checking per route.
  const support = useMemo(() => checkBrowserSupport(), [])
  if (!support.ok) {
    return <UnsupportedBrowser missing={support.missing} />
  }
  return (
    <BrowserRouter basename={basename}>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/measure" element={<Reader />} />
      </Routes>
    </BrowserRouter>
  )
}
