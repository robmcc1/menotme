# 🎭 Face Swap Stream

Real-time AI face swapping with your webcam. Upload photos or animated character images and see them swap with your face in real-time on a webpage.

## Features

✨ **Real-time Face Swapping** - Live webcam feed with AI face replacement
🎙️ **Real-time Voice Changer (RVC)** - Mic-to-speaker conversion with local `.pth/.index` model
🎨 **Multiple Targets** - Upload multiple faces (photos, characters, etc.)
🌐 **Web Interface** - Simple, modern UI for control and streaming
🚀 **GPU Optimized** - Built for RTX 5080 (CUDA support)
📸 **Drag & Drop Upload** - Easy target face management
🎬 **No OBS Required** - Direct video stream on webpage

## Requirements

- **OS:** Windows 10/11
- **GPU:** NVIDIA GPU with CUDA support (RTX 5080 recommended)
- **Python:** 3.9+
- **Webcam:** Any webcam
- **Audio:** Microphone + speakers/headphones

## Setup

### Windows (Recommended)

**Option 1: Automatic Setup**
1. Download and extract the project
2. Double-click `run.bat`
3. Wait for dependencies to install (~10-20 mins first time)
4. Browser opens to http://localhost:8000

**Option 2: Manual Setup**
```bash
cd face-swap-stream
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

### Voice Model

The app now expects your RVC zip at:

`C:\Users\robmc\Downloads\ryanreynoldsvoice.zip`

On first voice start, it extracts the model into `voice_models/` and loads the first `.pth` + `.index` found.

### Python 3.12 Note

- Base app (video face swap) runs on Python 3.12.
- Voice conversion now runs as a separate sidecar worker process in its own Python runtime.
- This avoids camera regressions and keeps RVC dependencies isolated.

### Sidecar Worker Setup (True RVC)

Run once from project root:

```bat
sidecar\setup_worker.bat
```

This script:
- Installs Python 3.10 (if missing)
- Creates `venv310/`
- Installs worker deps from `sidecar/requirements-worker.txt`

Optional override if your worker Python lives elsewhere:

```bat
set RVC_WORKER_PYTHON=C:\path\to\worker\python.exe
```

When you click **Start Voice**, the main app auto-starts `sidecar/rvc_worker.py` and proxies all `/voice/*` calls to it.

### Ubuntu/WSL

```bash
cd face-swap-stream
python3 -m venv venv
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
python3 main.py
```

**Note:** WSL cannot access Windows webcam directly. Run on Windows Python instead.

### You Should See

```
✓ InsightFace model loaded (GPU)
✓ Face swapper model loaded
✓ Webcam thread started
INFO:     Uvicorn running on http://0.0.0.0:8000
```

### 3. Open in Browser

Navigate to: **http://localhost:8000**

You'll see your live webcam feed on the left with upload controls on the right.

## How to Use

### 1. Upload a Target Face

- Drag and drop an image into the upload area, OR
- Click to browse and select an image
- Images can be:
  - Real faces (photos)
  - Animated characters
  - Anyone/anything with a detectable face

### 2. Select Target

- Click "Select" on an uploaded target
- The status will show "Active"
- Your face will now swap with the target face in real-time

### 3. Manage Targets

- Multiple targets can be uploaded
- Switch between them by clicking "Select"
- Delete targets with the "Delete" button

### 4. Start Voice Changer

- In the **Voice Changer (RVC)** panel, set pitch/index/latency block size.
- Choose output routing:
  - Enable **Prefer virtual audio output** to route to VB-Cable/Voicemeeter when detected.
  - Or manually choose an output device from the dropdown.
- Click **Start Voice**.
- Speak into your mic and monitor output from your selected playback device.
- Click **Stop Voice** when done.

### 5. Use in Zoom/Discord/OBS

- Install a virtual audio device like **VB-CABLE** (or Voicemeeter).
- In this app, enable **Prefer virtual audio output** and start voice.
- In Zoom/Discord/OBS microphone settings, select the virtual cable output as input.

## Tips

💡 **Best Results:**
- Use clear, well-lit photos
- Single face in target image (it uses the first detected face)
- Good lighting on your webcam
- Head-on angles work best

💡 **Character/Animated Faces:**
- Convert characters to static images first (screenshot)
- Ensure face is clear and forward-facing
- Larger images work better than tiny ones

💡 **Performance:**
- GPU mode: ~30 FPS (smooth real-time)
- If slow, check CUDA is working: watch NVIDIA GPU usage in Task Manager
- Resolution is 1280x720 by default

## Troubleshooting

### "No face detected in image"
- The target image must have at least one clearly visible face
- Try a different angle or image
- Increase lighting in the photo

### Video stream not showing
- Check webcam is not in use by another app
- Try different webcam in Device Manager
- Restart browser and server

### Slow/Lagging Video
- Check GPU usage (should be high)
- Reduce webcam resolution in main.py if needed
- Close other GPU-intensive apps

### InsightFace models failing to load
- Ensure you have enough disk space (~2GB)
- Check internet connection (models download on first run)
- Try: `pip install --upgrade insightface`

## Complete Uninstall

**Everything is self-contained in the project folder** - nothing installs globally.

**To completely remove:**

Option 1: Double-click `cleanup.bat` (removes venv and temp files)

Option 2: Delete entire folder
```bash
rmdir /s C:\Users\robmc\OneDrive\Documents\development\face-swap-stream
```

That's it! No leftover files or registry entries.

## File Structure

```
face-swap-stream/
├── main.py              # FastAPI backend
├── requirements.txt     # Python dependencies
├── run.bat              # Windows launcher
├── cleanup.bat          # Full cleanup script
├── uploads/             # Saved target images
└── static/
    ├── index.html       # Web interface
    ├── style.css        # Styling
    └── script.js        # Frontend logic
```

## API Endpoints

- `GET /` - Main webpage
- `GET /video_feed` - MJPEG video stream
- `POST /upload_target` - Upload target face image
- `GET /targets` - List available targets
- `POST /set_target/{filename}` - Activate a target
- `DELETE /target/{filename}` - Delete a target
- `GET /voice/status` - Voice changer runtime status
- `GET /voice/devices` - Available audio input/output devices + recommended virtual output
- `POST /voice/start` - Start RVC mic->speaker conversion
- `POST /voice/stop` - Stop voice conversion

## Performance Notes

**GPU:** RTX 5080 with CUDA
- Model loading: ~5-10 seconds first run
- Per-frame processing: ~30-50ms
- FPS: ~30 (MJPEG stream)

**Models Used:**
- **buffalo_l** (InsightFace) - Face detection & embedding
- **inswapper_128** - Face swapping

## License

MIT License - Feel free to modify and use as you like!

## Notes

- This tool requires active face detection in both your webcam and target image
- Quality depends on image clarity and lighting
- Works best with head-on angles
- More training/higher resolution models available if you want to extend it

---

Made with ❤️ using InsightFace, FastAPI, and OpenCV

Happy swapping! 🎭
