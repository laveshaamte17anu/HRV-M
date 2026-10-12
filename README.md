# HRV-M CardioCam

Remote photoplethysmography (rPPG) optical vitals measurement system.

## Workflow

1. **React Frontend**:
   - User positions face using live camera preview.
   - User clicks **"Start 30s Scan"**.
   - Browser records video for **exactly 30 seconds** via `MediaRecorder` with dynamic MIME type negotiation (`video/webm;codecs=vp9`, `vp8`, `h264`, `mp4`).
   - Countdown timer (30 → 0) and progress bar show scan progress.
   - **Zero network frames** are transmitted during recording.
   - At exactly 30 seconds, recording stops automatically and camera stream tracks are released.
2. **Video Upload**:
   - The completed 30s video blob is uploaded to `POST /api/analyze-video` using `multipart/form-data`.
3. **Python Backend**:
   - Saves video to temporary storage and decodes frames using OpenCV / PyAV.
   - Runs face tracking & cheek ROI extraction using MediaPipe Face Landmarker.
   - Extracts microvascular blood volume pulse (BVP) via POS (Plane-Orthogonal-to-Skin) and Green rPPG algorithms.
   - Applies detrending, Hamming windowing, zero-padded FFT, Butterworth bandpass filtering (48–180 BPM cardiac band, 11–30 brpm respiratory band).
   - Computes signal-to-noise ratio (SNR) and signal quality gating.
   - Deletes temporary video file and returns JSON vitals response.
4. **Results Display**:
   - React displays heart rate (BPM) with pulsing icon, respiration rate (brpm), signal quality badge, SNR, duration, and recovered BVP pulse waveform.

---

## Running the Application

### 1. Python Backend

```bash
cd backend
# On Windows (Git Bash):
source venv/Scripts/activate
# Or on Windows (PowerShell):
# .\venv\Scripts\Activate.ps1

python -m uvicorn app.main:app --reload
```

The API service runs on `http://127.0.0.1:8000`.

Endpoints:
- `GET /api/health` - Health check
- `POST /api/analyze-video` - Multipart video upload & rPPG analysis

### 2. React Web Frontend

```bash
cd web
npm install
npm run dev
```

Open `http://localhost:5173/` in your browser.
Vite proxies `/api` requests to `http://127.0.0.1:8000`.

---

## API Specification

### `POST /api/analyze-video`

**Request**:
- Content-Type: `multipart/form-data`
- Body: `file` (binary video recording, e.g. `.webm` or `.mp4`)

**Example Response**:
```json
{
  "success": true,
  "heart_rate": 78,
  "respiration_rate": 16,
  "signal_quality": "Good",
  "snr": 12.4,
  "duration": 30.0,
  "message": "Analysis completed successfully",
  "signal": [0.12, 0.45, -0.22, ...],
  "frames_processed": 900,
  "tracking_ratio": 0.98
}
```

---

## Project Structure

```
HRV-M/
├── backend/                  # Python FastAPI Backend Service
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py           # FastAPI endpoints (/api/health, /api/analyze-video)
│   │   ├── video_analyzer.py # 30s video rPPG extraction & DSP pipeline
│   │   ├── mediapipe_face.py # Face tracking & landmark alignment
│   │   ├── process.py        # CardioCam core processing
│   │   └── signal_processing.py
│   ├── models/
│   │   └── face_landmarker.task
│   ├── requirements.txt
│   └── venv/                 # Virtual environment
├── web/                      # React Frontend
│   ├── src/
│   │   ├── pages/
│   │   │   ├── Landing.tsx   # Entry page
│   │   │   └── Reader.tsx    # 30s scan, countdown, upload & results dashboard
│   │   ├── components/       # UI plots, heart icon, footer
│   │   └── hooks/            # useCamera
│   ├── package.json
│   └── vite.config.ts
├── api.py                    # Root API launcher
└── README.md
```

## License

MIT
