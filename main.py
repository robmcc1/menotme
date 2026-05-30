import os
import sys

# Must use add_dll_directory - os.environ PATH doesn't work for DLL loading on Windows
# Load CUDA 12.6 only - 13.3 is incomplete (missing cublasLt64_13.dll)
cuda_paths = [
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8\bin",
    r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.5\bin",
]
for p in cuda_paths:
    if os.path.exists(p):
        os.add_dll_directory(p)
        os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
        print(f"✓ Added CUDA DLL directory: {p}")

# cuDNN DLLs (cudnn64_9.dll) - check common install locations
cudnn_paths = [
    r"C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.9\x64",
    r"C:\Program Files\NVIDIA\CUDNN\v9.22\bin\12.6\x64",
    r"C:\Program Files\NVIDIA\CUDNN\v9.0\bin\12.6",
    r"C:\Program Files\NVIDIA\CUDNN\v9.0\bin",
    r"C:\Program Files\NVIDIA\CUDNN\v9\bin",
]
for p in cudnn_paths:
    if os.path.exists(p):
        dlls = [f for f in os.listdir(p) if f.startswith('cudnn') and f.endswith('.dll')]
        if dlls:
            os.add_dll_directory(p)
            os.environ["PATH"] = p + ";" + os.environ.get("PATH", "")
            print(f"✓ Added cuDNN DLL directory: {p}")
            break  # Only load ONE cuDNN version

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import insightface
from PIL import Image
import io
import os
from pathlib import Path
import asyncio
import threading
import time

app = FastAPI()

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create uploads directory
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

# --- Model config ---
# Models compatible with insightface's get_model() API:
#   'inswapper_128.onnx'         - standard, 554MB (recommended)
#   'inswapper_128_fp16.onnx'    - half-precision, 277MB, slightly faster
# NOTE: reswapper/hyperswap models are ReActor-specific and NOT compatible here.
SWAPPER_MODEL = 'inswapper_128.onnx'

# Global state
target_faces = {}  # {filename: face_embedding}
current_target = None
frame_lock = threading.Lock()
process_lock = threading.Lock()
camera_lock = threading.Lock()

CAMERA_RESOLUTION_PRESETS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
current_camera_resolution = "480p"

# Frame buffers
raw_frame = None  # Latest captured frame
latest_frame = None  # Latest processed frame (shown to browser)
frame_timestamp = None

# Face tracking - detect every N frames, reuse result in between
DETECT_EVERY_N_FRAMES = 3  # Detect face every 3 frames
_last_detected_faces = []
_detect_frame_counter = 0

# Check available providers
try:
    import onnxruntime as ort
    available_providers = ort.get_available_providers()
    print(f"✓ ONNX providers: {available_providers}")
    
    if 'CUDAExecutionProvider' in available_providers:
        providers = ['CUDAExecutionProvider', 'CPUExecutionProvider']
        use_gpu = True
        print("✓ Using GPU (CUDA) for face detection")
    elif 'DmlExecutionProvider' in available_providers:
        providers = ['DmlExecutionProvider', 'CPUExecutionProvider']
        use_gpu = True
        print("✓ Using GPU (DirectML) for face detection")
    else:
        providers = ['CPUExecutionProvider']
        use_gpu = False
        print("⚠️  Using CPU only")
except Exception as e:
    providers = ['CPUExecutionProvider']
    use_gpu = False
    print(f"⚠️  ONNX check failed: {e}")

# Initialize InsightFace - use smaller det_size for speed
try:
    app_insightface = insightface.app.FaceAnalysis(
        name='buffalo_l',
        providers=providers
    )
    # 320x320 is 4x faster than 640x640 with minimal quality loss
    app_insightface.prepare(ctx_id=0 if use_gpu else -1, det_size=(320, 320))
    print("✓ InsightFace model loaded")
except Exception as e:
    print(f"❌ InsightFace load failed: {e}")
    app_insightface = None

# Load swapper model - CUDA now that cuDNN is installed
swapper = None
try:
    print(f"Loading face swapper model ({SWAPPER_MODEL})...")
    swapper = insightface.model_zoo.get_model(
        SWAPPER_MODEL,
        providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
    )
    print("✓ Face swapper model loaded (GPU/CUDA)")
except Exception as e:
    print(f"⚠️  Warning: Swapper model failed to load: {e}")
    print("   Attempting alternative loading method...")
    try:
        import urllib.request
        import os
        
        model_path = SWAPPER_MODEL
        if not os.path.exists(model_path):
            print(f"   Downloading {SWAPPER_MODEL} from mirror...")
            
            # Gourieff/ReActor is the only reliable public mirror for these models
            base_name = SWAPPER_MODEL
            urls = [
                f"https://huggingface.co/datasets/Gourieff/ReActor/resolve/main/models/{base_name}",
            ]
            
            downloaded = False
            for url in urls:
                try:
                    print(f"   Trying: {url.split('/')[2]}...")
                    urllib.request.urlretrieve(url, model_path, reporthook=lambda count, block_size, total_size: print(f"   Progress: {count*block_size/1024/1024:.1f}MB") if count % 50 == 0 else None)
                    print("   ✓ Download complete!")
                    downloaded = True
                    break
                except Exception as url_err:
                    print(f"   ✗ Failed: {str(url_err)[:50]}")
                    continue
            
            if not downloaded:
                raise Exception("All download sources failed")
        
        swapper = insightface.model_zoo.get_model(
            model_path,
            providers=['CPUExecutionProvider']
        )
        print("✓ Face swapper model loaded from file (CPU - stable)")
    except Exception as e2:
        print(f"❌ Failed to load swapper model: {e2}")
        print("   Face swapping will not work")
        print(f"   Manual fix: Download {SWAPPER_MODEL} from https://huggingface.co/datasets/Gourieff/ReActor/tree/main/models")
        print(f"   Save as: {SWAPPER_MODEL} in the project folder")

# Webcam capture
print("🎥 Attempting to open webcam...")
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)  # Use DirectShow backend on Windows
cap.set(cv2.CAP_PROP_FRAME_WIDTH, CAMERA_RESOLUTION_PRESETS[current_camera_resolution][0])
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, CAMERA_RESOLUTION_PRESETS[current_camera_resolution][1])
cap.set(cv2.CAP_PROP_FPS, 60)  # Request 60 FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Only keep latest frame, drop old ones

if not cap.isOpened():
    print("❌ Failed to open camera. Check Device Manager and Privacy Settings.")
    exit(1)

actual_fps = cap.get(cv2.CAP_PROP_FPS)
actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
print(f"✓ Webcam opened successfully ({actual_width}x{actual_height} @ {actual_fps:.0f} FPS)")
print("✓ Ready to process frames")


def _closest_preset_name(width, height):
    target_area = int(width) * int(height)
    return min(
        CAMERA_RESOLUTION_PRESETS.keys(),
        key=lambda name: abs(CAMERA_RESOLUTION_PRESETS[name][0] * CAMERA_RESOLUTION_PRESETS[name][1] - target_area)
    )


def set_camera_resolution(preset_name):
    """Set webcam capture resolution preset at runtime."""
    global current_camera_resolution

    if preset_name not in CAMERA_RESOLUTION_PRESETS:
        raise ValueError("Unsupported resolution preset")

    width, height = CAMERA_RESOLUTION_PRESETS[preset_name]
    with camera_lock:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        cap.set(cv2.CAP_PROP_FPS, 60)

        actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        actual_fps = cap.get(cv2.CAP_PROP_FPS)

        current_camera_resolution = _closest_preset_name(actual_width, actual_height)

    return actual_width, actual_height, actual_fps, current_camera_resolution

def extract_face_from_image(image_path):
    """Extract face embedding from an image file"""
    try:
        img = cv2.imread(image_path)
        if img is None:
            return None
        
        faces = app_insightface.get(img)
        if len(faces) > 0:
            return faces[0]  # Return first face
        return None
    except Exception as e:
        print(f"Error extracting face: {e}")
        return None

def swap_faces(frame, source_face, target_face):
    """Swap faces in frame using target face"""
    if swapper is None:
        return frame
    
    if source_face is None or target_face is None:
        return frame
    
    try:
        result = swapper.get(frame, source_face, target_face, paste_back=True)
        return result
    except Exception as e:
        print(f"Error swapping faces: {e}")
        return frame

def process_frame(frame):
    """Process a single frame with face swapping - optimized with face tracking"""
    global current_target, _last_detected_faces, _detect_frame_counter
    
    if frame is None or app_insightface is None:
        return frame
    
    if swapper is None or not current_target or current_target not in target_faces:
        return frame  # No target set - show raw feed
    
    try:
        _detect_frame_counter += 1
        
        # Only run full face detection every N frames
        if _detect_frame_counter % DETECT_EVERY_N_FRAMES == 0 or not _last_detected_faces:
            _last_detected_faces = app_insightface.get(frame)
        
        if _last_detected_faces:
            target_face = target_faces[current_target]
            return swap_faces(frame, _last_detected_faces[0], target_face)
        else:
            return None  # No face detected - caller will freeze on last good frame
    except Exception as e:
        return None  # On error, freeze too

def webcam_capture_thread():
    """Capture frames as fast as possible - NO PROCESSING"""
    global raw_frame
    
    while True:
        with camera_lock:
            ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue
        
        # Flip for mirror effect
        frame = cv2.flip(frame, 1)
        
        # Store raw frame - always latest, drop old ones
        with frame_lock:
            raw_frame = frame

def face_processing_thread():
    """Process frames with face swapping - runs as fast as GPU allows"""
    global latest_frame
    last_raw = None
    
    while True:
        # Get latest raw frame
        with frame_lock:
            current_raw = raw_frame
        
        if current_raw is None:
            time.sleep(0.005)
            continue
        
        # Skip if same frame (no new capture yet)
        if current_raw is last_raw:
            time.sleep(0.005)
            continue
        
        last_raw = current_raw
        frame_to_process = current_raw.copy()
        
        # Process
        processed = process_frame(frame_to_process)
        
        # Only update if we got a valid frame (None = freeze on last good frame)
        if processed is not None:
            with process_lock:
                latest_frame = processed
        
        # Don't sleep - capture as fast as possible
        # The lock prevents blocking the MJPEG encoder

# Start webcam capture thread (fast, no processing)
capture_worker = threading.Thread(target=webcam_capture_thread, daemon=True)
capture_worker.start()
print("✓ Webcam capture thread started")

# Start face processing thread (can be slow, won't block capture)
process_worker = threading.Thread(target=face_processing_thread, daemon=True)
process_worker.start()
print("✓ Face processing thread started")

def generate_mjpeg():
    """Generator for MJPEG stream"""
    frame_count = 0
    while True:
        with process_lock:
            if latest_frame is None:
                time.sleep(0.01)
                continue
            
            frame = latest_frame.copy()
        
        # Encode frame to JPEG with lower quality for speed
        ret, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        if ret:
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n'
                   b'Content-Length: ' + str(len(buffer)).encode() + b'\r\n\r\n'
                   + buffer.tobytes() + b'\r\n')
        
        frame_count += 1
        time.sleep(0.01)

@app.get("/video_feed")
async def video_feed():
    """MJPEG stream endpoint"""
    return StreamingResponse(
        generate_mjpeg(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

@app.get("/frame")
async def get_frame():
    """Get single frame as JPEG"""
    with process_lock:
        if latest_frame is None:
            # Return blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            ret, buffer = cv2.imencode('.jpg', blank, [cv2.IMWRITE_JPEG_QUALITY, 60])
        else:
            # Encode with lower quality for speed
            ret, buffer = cv2.imencode('.jpg', latest_frame, [cv2.IMWRITE_JPEG_QUALITY, 60])
    
    if ret:
        return StreamingResponse(io.BytesIO(buffer.tobytes()), media_type="image/jpeg")
    return {"error": "Failed to encode frame"}

@app.post("/upload_target")
async def upload_target(file: UploadFile = File(...)):
    """Upload a target face image"""
    try:
        contents = await file.read()
        
        # Save file
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Extract face
        face = extract_face_from_image(str(file_path))
        if face is None:
            os.remove(file_path)
            raise HTTPException(status_code=400, detail="No face detected in image")
        
        target_faces[file.filename] = face
        
        return {
            "status": "success",
            "filename": file.filename,
            "message": f"Face loaded from {file.filename}"
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.get("/targets")
async def list_targets():
    """List available target faces"""
    return {
        "targets": list(target_faces.keys()),
        "current": current_target
    }


@app.get("/camera_resolution")
async def get_camera_resolution():
    """Get current camera resolution state and presets."""
    with camera_lock:
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(cap.get(cv2.CAP_PROP_FPS))

    return {
        "current": current_camera_resolution,
        "available": list(CAMERA_RESOLUTION_PRESETS.keys()),
        "actual_width": width,
        "actual_height": height,
        "actual_fps": fps,
    }


@app.post("/camera_resolution/{preset_name}")
async def set_camera_resolution_endpoint(preset_name: str):
    """Set camera resolution using a preset name."""
    try:
        width, height, fps, resolved_preset = set_camera_resolution(preset_name)
        return {
            "status": "success",
            "requested": preset_name,
            "current": resolved_preset,
            "actual_width": width,
            "actual_height": height,
            "actual_fps": fps,
        }
    except ValueError:
        raise HTTPException(status_code=400, detail="Unsupported resolution preset")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to switch camera resolution: {e}")

@app.post("/set_target/{filename}")
async def set_target(filename: str):
    """Set the current target face"""
    global current_target
    
    if filename not in target_faces:
        raise HTTPException(status_code=404, detail="Target not found")
    
    current_target = filename
    return {"status": "success", "current_target": filename}

@app.delete("/target/{filename}")
async def delete_target(filename: str):
    """Delete a target face"""
    global current_target
    
    if filename not in target_faces:
        raise HTTPException(status_code=404, detail="Target not found")
    
    del target_faces[filename]
    
    file_path = UPLOAD_DIR / filename
    if file_path.exists():
        os.remove(file_path)
    
    if current_target == filename:
        current_target = None
    
    return {"status": "success"}

@app.on_event("shutdown")
async def shutdown():
    cap.release()
    print("Webcam released")

# Serve index.html on root
@app.get("/", response_class=FileResponse)
async def root():
    return "static/index.html"

# Serve static files as catch-all
@app.get("/{file_path:path}", response_class=FileResponse)
async def serve_static(file_path: str):
    import os
    full_path = f"static/{file_path}"
    if os.path.exists(full_path):
        return full_path
    return FileResponse("static/index.html")  # Fallback to index.html

if __name__ == "__main__":
    import uvicorn
    import logging
    
    # Disable access logging
    logging.getLogger("uvicorn.access").setLevel(logging.CRITICAL)
    
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
