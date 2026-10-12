import os
import shutil
import tempfile
import logging
from typing import Optional
# pyrefly: ignore [missing-import]
from fastapi import FastAPI, UploadFile, File, HTTPException, Form
# pyrefly: ignore [missing-import]
from fastapi.middleware.cors import CORSMiddleware

try:
    from app.video_analyzer import VideoAnalyzer
except ImportError:
    from video_analyzer import VideoAnalyzer

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("main")

app = FastAPI(
    title="HRV-M CardioCam API",
    version="2.0.0",
    description="30-second video rPPG heart rate and respiratory vitals analysis service",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "*",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

analyzer = VideoAnalyzer()


@app.get("/api/health")
def health():
    return {
        "status": "ok",
        "service": "HRV-M CardioCam API",
        "version": "2.0.0",
    }


@app.get("/api")
def api_info():
    return {
        "name": "HRV-M CardioCam API",
        "version": "2.0.0",
        "endpoints": [
            "/api/health",
            "/api/analyze-video",
        ],
    }


@app.post("/api/analyze-video")
async def analyze_video(
    file: UploadFile = File(...),
    duration: Optional[float] = Form(None),
    client_bpm: Optional[float] = Form(None),
    samples: Optional[str] = Form(None),
):
    """
    Accepts a 30-second recorded video (multipart/form-data), runs CardioCam
    rPPG facial pulse extraction and DSP analysis, and returns final vitals.
    """
    if not file:
        raise HTTPException(status_code=400, detail="No video file provided.")

    filename = file.filename or "recording.webm"
    ext = os.path.splitext(filename)[1].lower()
    if not ext:
        ext = ".webm"

    valid_extensions = {".webm", ".mp4", ".mov", ".avi", ".mkv"}
    if ext not in valid_extensions:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported video format '{ext}'. Supported formats: {', '.join(valid_extensions)}",
        )

    temp_video_path = None
    try:
        # Save uploaded file stream to a temporary video file on disk
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as temp_file:
            temp_video_path = temp_file.name
            shutil.copyfileobj(file.file, temp_file)

        file_size = os.path.getsize(temp_video_path)
        logger.info(
            f"Received video '{filename}', size={file_size} bytes, "
            f"client_duration={duration}, client_bpm={client_bpm}, has_samples={bool(samples)}, saved to {temp_video_path}"
        )

        if file_size < 1024 and not samples:
            raise HTTPException(status_code=400, detail="Uploaded video file is empty or corrupted.")

        # Process the complete video using the CardioCam rPPG pipeline
        result = analyzer.process_video(
            temp_video_path,
            # pyrefly: ignore [unexpected-keyword]
            client_duration=duration,
            # pyrefly: ignore [unexpected-keyword]
            client_bpm=client_bpm,
            # pyrefly: ignore [unexpected-keyword]
            samples_json=samples,
        )
        logger.info(f"Analysis result: heart_rate={result.get('heart_rate')}, success={result.get('success')}")

        return result

    except HTTPException:
        raise
    except Exception as exc:
        logger.exception("Error during video analysis:")
        raise HTTPException(
            status_code=500,
            detail=f"Video analysis error: {str(exc)}",
        )
    finally:
        # Guarantee deletion of the temporary file after processing
        if temp_video_path and os.path.exists(temp_video_path):
            try:
                os.remove(temp_video_path)
                logger.info(f"Cleaned up temporary file: {temp_video_path}")
            except Exception as e:
                logger.warning(f"Failed to remove temporary file {temp_video_path}: {e}")
