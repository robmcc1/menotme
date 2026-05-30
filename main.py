import os
import sys
import queue
import tempfile
import zipfile
import json
import subprocess
import atexit
from urllib import request as urllib_request
from urllib import error as urllib_error

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

try:
    from pedalboard import Pedalboard, PitchShift, Compressor, Limiter
except Exception:
    Pedalboard = None
    PitchShift = None
    Compressor = None
    Limiter = None

app = FastAPI()


class RVCWorkerClient:
    """Controls a dedicated sidecar process for true RVC voice conversion."""

    def __init__(self):
        self.host = "127.0.0.1"
        self.port = int(os.environ.get("RVC_WORKER_PORT", "8011"))
        self.base_url = f"http://{self.host}:{self.port}"
        self.process = None
        self.default_python = str(Path("venv310") / "Scripts" / "python.exe")
        self.python_exe = os.environ.get("RVC_WORKER_PYTHON", self.default_python)
        self.worker_script = str(Path("sidecar") / "rvc_worker.py")

    def _request_json(self, method: str, path: str, payload: dict | None = None, timeout: float = 3.0):
        url = f"{self.base_url}{path}"
        data = None
        headers = {}
        if payload is not None:
            data = json.dumps(payload).encode("utf-8")
            headers["Content-Type"] = "application/json"

        req = urllib_request.Request(url=url, data=data, method=method, headers=headers)
        try:
            with urllib_request.urlopen(req, timeout=timeout) as resp:
                body = resp.read().decode("utf-8")
                return resp.status, json.loads(body) if body else {}
        except urllib_error.HTTPError as e:
            try:
                body = e.read().decode("utf-8")
                parsed = json.loads(body) if body else {}
            except Exception:
                parsed = {"detail": str(e)}
            return e.code, parsed
        except Exception as e:
            return 0, {"detail": str(e)}

    def is_healthy(self):
        status, _ = self._request_json("GET", "/health", timeout=1.0)
        return status == 200

    def ensure_running(self):
        if self.is_healthy():
            return

        python_path = Path(self.python_exe)
        if not python_path.exists():
            raise RuntimeError(
                "RVC worker runtime not found. Expected worker Python at "
                f"{python_path}. Run sidecar\\setup_worker.bat first."
            )

        script_path = Path(self.worker_script)
        if not script_path.exists():
            raise RuntimeError(f"RVC worker script missing: {script_path}")

        # Keep sidecar isolated from system cuDNN injections used by the main app.
        # Torch wheels ship their own CUDA/cuDNN runtime and can fail with DLL mismatch
        # if an incompatible global cuDNN path is inherited.
        worker_env = os.environ.copy()
        path_entries = worker_env.get("PATH", "").split(";")
        filtered_entries = [
            entry
            for entry in path_entries
            if entry and ("\\NVIDIA\\CUDNN\\" not in entry.upper())
        ]
        worker_env["PATH"] = ";".join(filtered_entries)

        if self.process is None or self.process.poll() is not None:
            self.process = subprocess.Popen(
                [str(python_path), str(script_path), "--host", self.host, "--port", str(self.port)],
                env=worker_env,
                creationflags=subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0,
            )

        # First boot can take longer while loading RVC/rmvpe models.
        for _ in range(300):
            if self.is_healthy():
                return
            time.sleep(0.2)

        raise RuntimeError("RVC worker did not become healthy in time")

    def stop_worker(self):
        status, _ = self._request_json("POST", "/shutdown", payload={}, timeout=1.5)
        if self.process is not None and self.process.poll() is None:
            try:
                self.process.terminate()
            except Exception:
                pass

    def get_status(self):
        status, data = self._request_json("GET", "/voice/status", timeout=2.0)
        if status == 200:
            return data
        return {
            "running": False,
            "status": data.get("detail", "worker unavailable"),
            "backend": "rvc",
            "worker_online": False,
        }

    def list_devices(self):
        self.ensure_running()
        status, data = self._request_json("GET", "/voice/devices", timeout=3.0)
        if status != 200:
            raise RuntimeError(data.get("detail", "Failed to fetch worker devices"))
        return data

    def start_voice(self, config: dict):
        self.ensure_running()
        status, data = self._request_json("POST", "/voice/start", payload=config, timeout=10.0)
        if status != 200:
            raise RuntimeError(data.get("detail", "Failed to start worker voice"))
        return data

    def stop_voice(self):
        status, data = self._request_json("POST", "/voice/stop", payload={}, timeout=4.0)
        if status != 200:
            raise RuntimeError(data.get("detail", "Failed to stop worker voice"))
        return data


rvc_worker_client = RVCWorkerClient()
atexit.register(rvc_worker_client.stop_worker)

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

VOICE_MODEL_DIR = Path("voice_models")
VOICE_MODEL_DIR.mkdir(exist_ok=True)

DEFAULT_VOICE_MODEL_ZIP = Path(r"C:\Users\robmc\Downloads\ryanreynoldsvoice.zip")

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
shutdown_event = threading.Event()

CAMERA_RESOLUTION_PRESETS = {
    "480p": (640, 480),
    "720p": (1280, 720),
    "1080p": (1920, 1080),
}
current_camera_resolution = "480p"
capture_worker = None
process_worker = None

# Frame buffers
raw_frame = None  # Latest captured frame
latest_frame = None  # Latest processed frame (shown to browser)
frame_timestamp = None


class VoiceChangerEngine:
    """Chunked near-real-time voice conversion using an RVC model."""

    VIRTUAL_DEVICE_KEYWORDS = [
        "cable",
        "vb-audio",
        "voicemeeter",
        "virtual",
        "loopback",
        "blackhole",
    ]

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.status_message = "idle"
        self.backend = "none"
        self.model_pth = None
        self.model_index = None
        self.device = "cpu:0"

        self.sample_rate = 24000
        self.block_seconds = 0.18
        self.input_device = None
        self.output_device = None

        self.pitch = 0
        self.index_rate = 0.5
        self.f0_method = "harvest"
        self.protect = 0.33

        self._input_queue = queue.Queue(maxsize=3)
        self._output_queue = queue.Queue(maxsize=8)
        self._worker_thread = None
        self._stop_event = threading.Event()

        self._sd = None
        self._sf = None
        self._rvc = None
        self._input_stream = None
        self._output_stream = None
        self._dsp_board = None
        self._dsp_pitch = 0.0

    def _is_virtual_device_name(self, name: str):
        lowered = (name or "").lower()
        return any(keyword in lowered for keyword in self.VIRTUAL_DEVICE_KEYWORDS)

    def _detect_best_device(self):
        try:
            import onnxruntime as ort
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return "cuda:0"
        except Exception:
            pass
        return "cpu:0"

    def _extract_voice_model(self, zip_path: Path):
        if not zip_path.exists():
            raise FileNotFoundError(f"Voice model zip not found: {zip_path}")

        model_folder = VOICE_MODEL_DIR / zip_path.stem
        model_folder.mkdir(parents=True, exist_ok=True)

        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(model_folder)

        pth_files = list(model_folder.rglob("*.pth"))
        if not pth_files:
            raise ValueError("No .pth file found in model zip")

        index_files = list(model_folder.rglob("*.index"))
        return pth_files[0], (index_files[0] if index_files else None)

    def _ensure_audio_runtime(self):
        if self._sd is None:
            import sounddevice as sd
            self._sd = sd
        if self._sf is None:
            import soundfile as sf
            self._sf = sf

    def _ensure_rvc_runtime(self):
        if self._rvc is None:
            try:
                from rvc_python.infer import RVCInference
            except Exception as e:
                raise RuntimeError(
                    "rvc-python backend is not installed in this environment. "
                    "On Windows + Python 3.12 this is often unsupported due to native dependency constraints."
                ) from e
            self._rvc = RVCInference(device=self.device)

    def _resolve_device_index(self, selected_device, kind: str):
        if selected_device is None or selected_device == "":
            return None

        devices = self._sd.query_devices()
        needed_channels_key = "max_input_channels" if kind == "input" else "max_output_channels"

        if isinstance(selected_device, str) and selected_device.strip().isdigit():
            selected_device = int(selected_device.strip())

        if isinstance(selected_device, int):
            if selected_device < 0 or selected_device >= len(devices):
                raise ValueError(f"{kind} device index out of range: {selected_device}")
            if int(devices[selected_device].get(needed_channels_key, 0)) <= 0:
                raise ValueError(f"Selected device is not a valid {kind} device: {selected_device}")
            return selected_device

        selected_name = str(selected_device).strip().lower()
        if not selected_name:
            return None

        for idx, dev in enumerate(devices):
            if int(dev.get(needed_channels_key, 0)) <= 0:
                continue
            if dev.get("name", "").lower() == selected_name:
                return idx

        for idx, dev in enumerate(devices):
            if int(dev.get(needed_channels_key, 0)) <= 0:
                continue
            if selected_name in dev.get("name", "").lower():
                return idx

        raise ValueError(f"Could not find {kind} device matching: {selected_device}")

    def _pick_virtual_output_device(self):
        devices = self._sd.query_devices()
        for idx, dev in enumerate(devices):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            if self._is_virtual_device_name(dev.get("name", "")):
                return idx
        return None

    def _device_name(self, device_index):
        if device_index is None:
            return None
        try:
            return self._sd.query_devices(device_index).get("name")
        except Exception:
            return None

    def list_audio_devices(self):
        self._ensure_audio_runtime()
        devices = self._sd.query_devices()
        default_in, default_out = self._sd.default.device

        inputs = []
        outputs = []
        for idx, dev in enumerate(devices):
            in_ch = int(dev.get("max_input_channels", 0))
            out_ch = int(dev.get("max_output_channels", 0))

            if in_ch > 0:
                inputs.append({
                    "index": idx,
                    "name": dev.get("name", f"Input {idx}"),
                    "channels": in_ch,
                    "is_default": idx == default_in,
                })

            if out_ch > 0:
                outputs.append({
                    "index": idx,
                    "name": dev.get("name", f"Output {idx}"),
                    "channels": out_ch,
                    "is_default": idx == default_out,
                    "is_virtual": self._is_virtual_device_name(dev.get("name", "")),
                })

        recommended_virtual = None
        for dev in outputs:
            if dev["is_virtual"]:
                recommended_virtual = dev
                break

        return {
            "inputs": inputs,
            "outputs": outputs,
            "recommended_virtual_output": recommended_virtual,
        }

    def _input_callback(self, indata, frames, time_info, status):
        if status:
            pass
        if not self.running:
            return
        mono = indata[:, 0].copy()
        try:
            self._input_queue.put_nowait(mono)
        except queue.Full:
            # Drop oldest audio if conversion lags to keep latency bounded.
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(mono)
            except Exception:
                pass

    def _output_callback(self, outdata, frames, time_info, status):
        if status:
            pass
        if not self.running:
            outdata[:] = 0
            return

        needed = frames
        chunks = []
        while needed > 0:
            if self._output_queue.empty():
                chunks.append(np.zeros(needed, dtype=np.float32))
                break

            chunk = self._output_queue.get()
            if len(chunk) <= needed:
                chunks.append(chunk)
                needed -= len(chunk)
            else:
                chunks.append(chunk[:needed])
                remainder = chunk[needed:]
                self._output_queue.put(remainder)
                needed = 0

        data = np.concatenate(chunks) if chunks else np.zeros(frames, dtype=np.float32)
        if len(data) < frames:
            data = np.pad(data, (0, frames - len(data)), mode="constant")
        outdata[:, 0] = data[:frames]

    def _resample_linear(self, audio, src_sr, dst_sr):
        if src_sr == dst_sr or len(audio) == 0:
            return audio
        duration = len(audio) / float(src_sr)
        dst_len = max(1, int(duration * dst_sr))
        src_x = np.linspace(0.0, 1.0, num=len(audio), endpoint=False)
        dst_x = np.linspace(0.0, 1.0, num=dst_len, endpoint=False)
        return np.interp(dst_x, src_x, audio).astype(np.float32)

    def _convert_chunk_dsp(self, audio):
        """Higher-quality DSP fallback when RVC backend is unavailable."""
        audio = np.asarray(audio, dtype=np.float32)
        if len(audio) == 0:
            return audio

        semitones = float(self.pitch)

        if Pedalboard is None:
            # Hard fallback if pedalboard is unavailable.
            return np.clip(audio, -1.0, 1.0)

        if self._dsp_board is None or abs(self._dsp_pitch - semitones) > 1e-6:
            self._dsp_board = Pedalboard([
                PitchShift(semitones=semitones),
                Compressor(threshold_db=-22.0, ratio=3.0, attack_ms=5.0, release_ms=80.0),
                Limiter(threshold_db=-1.0),
            ])
            self._dsp_pitch = semitones

        out = self._dsp_board(audio, self.sample_rate)
        out = np.asarray(out, dtype=np.float32)
        if out.ndim > 1:
            out = out[0]

        if len(out) != len(audio):
            if len(out) > len(audio):
                out = out[:len(audio)]
            else:
                out = np.pad(out, (0, len(audio) - len(out)), mode="constant")

        return np.clip(out, -1.0, 1.0)

    def _convert_worker(self):
        while not self._stop_event.is_set():
            try:
                audio = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            if self.backend == "rvc":
                with tempfile.TemporaryDirectory(prefix="rvc_rt_") as tmp:
                    in_wav = Path(tmp) / "in.wav"
                    out_wav = Path(tmp) / "out.wav"

                    self._sf.write(str(in_wav), audio, self.sample_rate)

                    try:
                        self._rvc.infer_file(str(in_wav), str(out_wav))
                    except Exception as e:
                        with self.lock:
                            self.status_message = f"rvc conversion error: {e}"
                        continue

                    try:
                        converted, out_sr = self._sf.read(str(out_wav), dtype="float32")
                    except Exception as e:
                        with self.lock:
                            self.status_message = f"rvc output read error: {e}"
                        continue

                    if isinstance(converted, np.ndarray) and converted.ndim > 1:
                        converted = converted[:, 0]

                    converted = self._resample_linear(np.asarray(converted, dtype=np.float32), out_sr, self.sample_rate)
            else:
                converted = self._convert_chunk_dsp(audio)

            chunk_size = max(1, int(self.sample_rate * 0.04))
            for start in range(0, len(converted), chunk_size):
                if self._stop_event.is_set():
                    break
                piece = converted[start:start + chunk_size]
                try:
                    self._output_queue.put(piece, timeout=0.2)
                except queue.Full:
                    try:
                        self._output_queue.get_nowait()
                        self._output_queue.put_nowait(piece)
                    except Exception:
                        pass

    def start(self, model_zip: Path = DEFAULT_VOICE_MODEL_ZIP, pitch: int = 0, index_rate: float = 0.5,
              f0_method: str = "harvest", protect: float = 0.33, sample_rate: int = 24000,
              block_seconds: float = 0.18, input_device=None, output_device=None,
              use_virtual_output: bool = False, backend: str = "auto"):
        with self.lock:
            if self.running:
                raise RuntimeError("Voice changer is already running")

            self.device = self._detect_best_device()
            self.sample_rate = max(8000, int(sample_rate))
            self.block_seconds = max(0.05, min(0.8, float(block_seconds)))
            self.pitch = int(pitch)
            self.index_rate = float(index_rate)
            self.f0_method = str(f0_method)
            self.protect = float(protect)

            self.backend = "none"
            requested_backend = str(backend or "auto").strip().lower()

            self._ensure_audio_runtime()

            self.input_device = self._resolve_device_index(input_device, kind="input")
            virtual_requested_but_missing = False

            if use_virtual_output and (output_device is None or output_device == "" or str(output_device).strip().lower() == "auto"):
                self.output_device = self._pick_virtual_output_device()
                if self.output_device is None:
                    # Graceful fallback: continue on system default output instead of hard-failing.
                    self.output_device = None
                    virtual_requested_but_missing = True
            else:
                self.output_device = self._resolve_device_index(output_device, kind="output")

            rvc_error = None
            if requested_backend in {"auto", "rvc"}:
                try:
                    self.model_pth, self.model_index = self._extract_voice_model(Path(model_zip))
                    self._ensure_rvc_runtime()
                    self._rvc.load_model(
                        str(self.model_pth),
                        version="v2",
                        index_path=str(self.model_index) if self.model_index else "",
                    )
                    self._rvc.set_params(
                        f0method=self.f0_method,
                        f0up_key=self.pitch,
                        index_rate=self.index_rate,
                        protect=self.protect,
                    )
                    self.backend = "rvc"
                except Exception as e:
                    rvc_error = str(e)
                    if requested_backend == "rvc":
                        raise RuntimeError(rvc_error)

            if self.backend == "none":
                self.backend = "dsp"
                self.model_pth = None
                self.model_index = None

            while not self._input_queue.empty():
                self._input_queue.get_nowait()
            while not self._output_queue.empty():
                self._output_queue.get_nowait()
            self._dsp_board = None
            self._dsp_pitch = float(self.pitch)

            block_size = max(128, int(self.sample_rate * self.block_seconds))
            out_block = max(128, int(self.sample_rate * 0.01))

            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._convert_worker, daemon=True)
            self._worker_thread.start()

            self._input_stream = self._sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=block_size,
                callback=self._input_callback,
                device=self.input_device,
            )
            self._output_stream = self._sd.OutputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=out_block,
                callback=self._output_callback,
                device=self.output_device,
            )

            self._input_stream.start()
            self._output_stream.start()
            self.running = True
            output_name = self._device_name(self.output_device) or "system default"
            if self.backend == "rvc":
                self.status_message = f"running rvc ({self.device}) -> {output_name}"
            elif rvc_error:
                self.status_message = f"running dsp fallback -> {output_name} (rvc unavailable)"
            else:
                self.status_message = f"running dsp -> {output_name}"

            if virtual_requested_but_missing:
                self.status_message += " (no virtual output found; using default output)"

    def stop(self):
        with self.lock:
            if not self.running:
                return

            self.running = False
            self._stop_event.set()

            try:
                if self._input_stream is not None:
                    self._input_stream.stop()
                    self._input_stream.close()
            except Exception:
                pass

            try:
                if self._output_stream is not None:
                    self._output_stream.stop()
                    self._output_stream.close()
            except Exception:
                pass

            self._input_stream = None
            self._output_stream = None

            if self._worker_thread is not None and self._worker_thread.is_alive():
                self._worker_thread.join(timeout=1.0)
            self._worker_thread = None

            self.status_message = "stopped"

    def status(self):
        output_name = None
        input_name = None
        if self._sd is not None:
            input_name = self._device_name(self.input_device)
            output_name = self._device_name(self.output_device)

        return {
            "running": self.running,
            "status": self.status_message,
            "backend": self.backend,
            "device": self.device,
            "sample_rate": self.sample_rate,
            "block_seconds": self.block_seconds,
            "model_pth": str(self.model_pth) if self.model_pth else None,
            "model_index": str(self.model_index) if self.model_index else None,
            "pitch": self.pitch,
            "index_rate": self.index_rate,
            "f0_method": self.f0_method,
            "protect": self.protect,
            "default_model_zip": str(DEFAULT_VOICE_MODEL_ZIP),
            "input_device_index": self.input_device,
            "input_device_name": input_name,
            "output_device_index": self.output_device,
            "output_device_name": output_name,
            "using_virtual_output": self._is_virtual_device_name(output_name or ""),
        }


voice_engine = VoiceChangerEngine()

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
    
    while not shutdown_event.is_set():
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
    
    while not shutdown_event.is_set():
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


@app.get("/voice/status")
async def voice_status():
    """Get current voice changer state."""
    return rvc_worker_client.get_status()


@app.get("/voice/devices")
async def voice_devices():
    """List available input/output devices and virtual output recommendation."""
    try:
        return rvc_worker_client.list_devices()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/voice/start")
async def voice_start(config: dict | None = None):
    """Start near-real-time voice conversion (mic -> speakers)."""
    config = config or {}
    try:
        cfg = {
            "model_zip": str(config.get("model_zip", str(DEFAULT_VOICE_MODEL_ZIP))),
            "pitch": int(config.get("pitch", 0)),
            "index_rate": float(config.get("index_rate", 0.06)),
            "f0_method": str(config.get("f0_method", "harvest")),
            "protect": float(config.get("protect", 0.80)),
            "sample_rate": int(config.get("sample_rate", 24000)),
            "block_seconds": float(config.get("block_seconds", 0.32)),
            "noise_gate_enabled": bool(config.get("noise_gate_enabled", False)),
            "noise_gate_threshold": float(config.get("noise_gate_threshold", 0.008)),
            "noise_gate_hold_ms": int(config.get("noise_gate_hold_ms", 140)),
            "noise_gate_release_ms": int(config.get("noise_gate_release_ms", 70)),
            "dry_mix": float(config.get("dry_mix", 0.42)),
            "input_device": config.get("input_device"),
            "output_device": config.get("output_device"),
            "use_virtual_output": bool(config.get("use_virtual_output", False)),
            "backend": "rvc",
        }
        started = rvc_worker_client.start_voice(cfg)
        return {"status": "success", "voice": started.get("voice", started)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/voice/stop")
async def voice_stop():
    """Stop voice conversion runtime."""
    try:
        stopped = rvc_worker_client.stop_voice()
        return {"status": "success", "voice": stopped.get("voice", stopped)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.on_event("shutdown")
async def shutdown():
    shutdown_event.set()

    voice_engine.stop()
    rvc_worker_client.stop_worker()

    if capture_worker is not None and capture_worker.is_alive():
        capture_worker.join(timeout=1.0)
    if process_worker is not None and process_worker.is_alive():
        process_worker.join(timeout=1.0)

    cap.release()
    print("Webcam released")

# Serve index.html on root
@app.get("/", response_class=FileResponse)
async def root():
    return "static/index.html"


@app.get("/audio", response_class=FileResponse)
async def audio_app():
    return "static/audio.html"

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
    
    try:
        uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
    except KeyboardInterrupt:
        print("\nCtrl+C received, shutting down...")
    finally:
        shutdown_event.set()
        try:
            cap.release()
        except Exception:
            pass
