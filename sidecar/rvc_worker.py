import argparse
import json
import os
import queue
import tempfile
import threading
import time
import zipfile
from pathlib import Path

# Force legacy torch.load behavior for trusted local checkpoints (fairseq/RVC).
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

import numpy as np
import sounddevice as sd
import soundfile as sf
import torch
from scipy.io import wavfile
from fastapi import FastAPI, HTTPException

# Torch 2.6 changed torch.load default to weights_only=True, which breaks
# fairseq checkpoints used by rvc-python unless explicitly disabled.
_orig_torch_load = torch.load
_orig_torch_serialization_load = torch.serialization.load


def _torch_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_load(*args, **kwargs)


def _torch_serialization_load_compat(*args, **kwargs):
    kwargs.setdefault("weights_only", False)
    return _orig_torch_serialization_load(*args, **kwargs)


torch.load = _torch_load_compat
torch.serialization.load = _torch_serialization_load_compat

from rvc_python.infer import RVCInference


def _infer_file_tuple_safe(self, input_path, output_path):
    """Compatibility patch for rvc-python vc_single tuple return values."""
    if not self.current_model:
        raise ValueError("Please load a model first.")

    model_info = self.models[self.current_model]
    file_index = model_info.get("index", "")

    wav_opt = self.vc.vc_single(
        sid=0,
        input_audio_path=input_path,
        f0_up_key=self.f0up_key,
        f0_method=self.f0method,
        file_index=file_index,
        index_rate=self.index_rate,
        filter_radius=self.filter_radius,
        resample_sr=self.resample_sr,
        rms_mix_rate=self.rms_mix_rate,
        protect=self.protect,
        f0_file="",
        file_index2="",
    )

    # rvc-python may return tuple metadata + waveform; keep only waveform.
    if isinstance(wav_opt, tuple):
        for item in reversed(wav_opt):
            if hasattr(item, "dtype"):
                wav_opt = item
                break

    if not hasattr(wav_opt, "dtype"):
        raise RuntimeError(f"Unexpected vc_single output type: {type(wav_opt)}")

    wavfile.write(output_path, self.vc.tgt_sr, wav_opt)
    return output_path


RVCInference.infer_file = _infer_file_tuple_safe

VOICE_MODEL_DIR = Path("voice_models")
VOICE_MODEL_DIR.mkdir(exist_ok=True)
DEFAULT_VOICE_MODEL_ZIP = Path(r"C:\Users\robmc\Downloads\ryanreynoldsvoice.zip")


def _select_rvc_device():
    """Choose inference device, preferring CUDA when available."""
    forced = os.environ.get("RVC_WORKER_DEVICE", "").strip().lower()
    cuda_available = torch.cuda.is_available()

    if forced:
        if forced.startswith("cuda"):
            return "cuda:0" if cuda_available else ""
        if forced.startswith("cpu"):
            return ""

    return "cuda:0" if cuda_available else ""


class WorkerVoiceEngine:
    VIRTUAL_DEVICE_KEYWORDS = ["cable", "vb-audio", "voicemeeter", "virtual", "loopback", "blackhole"]

    def __init__(self):
        self.lock = threading.Lock()
        self.running = False
        self.status_message = "idle"
        self.backend = "rvc"

        self.sample_rate = 24000
        self.block_seconds = 0.24
        self.input_device = None
        self.output_device = None

        self.pitch = 0
        self.index_rate = 0.06
        self.f0_method = "harvest"
        self.protect = 0.80
        self.dry_mix = 0.42
        self.device = _select_rvc_device()

        self.model_pth = None
        self.model_index = None

        self.noise_gate_enabled = False
        self.noise_gate_threshold = 0.012
        self.noise_gate_hold_ms = 140
        self.noise_gate_release_ms = 70
        self._noise_gate_hold_remaining = 0
        self._noise_gate_gain = 0.0

        self._input_queue = queue.Queue(maxsize=6)
        self._output_queue = queue.Queue(maxsize=96)
        self._worker_thread = None
        self._stop_event = threading.Event()
        self._input_stream = None
        self._output_stream = None
        self._pending_tail = None
        self._crossfade_seconds = 0.012
        self._last_output_value = 0.0
        self._underrun_events = 0

        self._rvc = None

    def _init_rvc_runtime(self, device: str):
        self._rvc = RVCInference(device=device)
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
        self.device = device

    def _is_virtual_device_name(self, name: str):
        lowered = (name or "").lower()
        return any(k in lowered for k in self.VIRTUAL_DEVICE_KEYWORDS)

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

    def _resolve_device_index(self, selected_device, kind: str):
        if selected_device is None or selected_device == "":
            return None

        devices = sd.query_devices()
        needed = "max_input_channels" if kind == "input" else "max_output_channels"

        if isinstance(selected_device, str) and selected_device.strip().isdigit():
            selected_device = int(selected_device.strip())

        if isinstance(selected_device, int):
            if selected_device < 0 or selected_device >= len(devices):
                raise ValueError(f"{kind} device index out of range: {selected_device}")
            if int(devices[selected_device].get(needed, 0)) <= 0:
                raise ValueError(f"Selected device is not a valid {kind} device: {selected_device}")
            return selected_device

        selected_name = str(selected_device).strip().lower()
        if not selected_name:
            return None

        for idx, dev in enumerate(devices):
            if int(dev.get(needed, 0)) <= 0:
                continue
            if dev.get("name", "").lower() == selected_name:
                return idx

        for idx, dev in enumerate(devices):
            if int(dev.get(needed, 0)) <= 0:
                continue
            if selected_name in dev.get("name", "").lower():
                return idx

        raise ValueError(f"Could not find {kind} device matching: {selected_device}")

    def _pick_virtual_output_device(self):
        devices = sd.query_devices()
        for idx, dev in enumerate(devices):
            if int(dev.get("max_output_channels", 0)) <= 0:
                continue
            if self._is_virtual_device_name(dev.get("name", "")):
                return idx
        return None

    def _device_name(self, idx):
        if idx is None:
            return None
        try:
            return sd.query_devices(idx).get("name")
        except Exception:
            return None

    def list_audio_devices(self):
        devices = sd.query_devices()
        default_in, default_out = sd.default.device

        inputs = []
        outputs = []
        for idx, dev in enumerate(devices):
            in_ch = int(dev.get("max_input_channels", 0))
            out_ch = int(dev.get("max_output_channels", 0))
            if in_ch > 0:
                inputs.append({"index": idx, "name": dev.get("name", f"Input {idx}"), "channels": in_ch, "is_default": idx == default_in})
            if out_ch > 0:
                outputs.append({
                    "index": idx,
                    "name": dev.get("name", f"Output {idx}"),
                    "channels": out_ch,
                    "is_default": idx == default_out,
                    "is_virtual": self._is_virtual_device_name(dev.get("name", "")),
                })

        recommended_virtual = next((d for d in outputs if d["is_virtual"]), None)
        return {"inputs": inputs, "outputs": outputs, "recommended_virtual_output": recommended_virtual}

    def _input_callback(self, indata, frames, time_info, status):
        if not self.running:
            return
        mono = indata[:, 0].copy()
        mono = self._apply_noise_gate(mono)
        try:
            self._input_queue.put_nowait(mono)
        except queue.Full:
            try:
                self._input_queue.get_nowait()
                self._input_queue.put_nowait(mono)
            except Exception:
                pass

    def _apply_noise_gate(self, mono: np.ndarray):
        if not self.noise_gate_enabled or mono.size == 0:
            return mono

        rms = float(np.sqrt(np.mean(np.square(mono, dtype=np.float32)) + 1e-12))
        is_open = rms >= self.noise_gate_threshold
        hold_frames = int(self.sample_rate * (self.noise_gate_hold_ms / 1000.0))
        release_frames = max(1, int(self.sample_rate * (self.noise_gate_release_ms / 1000.0)))

        if is_open:
            self._noise_gate_hold_remaining = hold_frames
            target_gain = 1.0
        else:
            if self._noise_gate_hold_remaining > 0:
                self._noise_gate_hold_remaining = max(0, self._noise_gate_hold_remaining - mono.size)
                target_gain = 1.0
            else:
                target_gain = 0.0

        step = min(1.0, mono.size / float(release_frames))
        if target_gain > self._noise_gate_gain:
            self._noise_gate_gain = target_gain
        else:
            self._noise_gate_gain += (target_gain - self._noise_gate_gain) * step

        return (mono * self._noise_gate_gain).astype(np.float32, copy=False)

    def _output_callback(self, outdata, frames, time_info, status):
        if not self.running:
            outdata[:] = 0
            return

        needed = frames
        chunks = []
        had_underrun = False
        while needed > 0:
            if self._output_queue.empty():
                had_underrun = True
                self._underrun_events += 1
                # Avoid hard zero-dropouts; fade from last sample to silence.
                tail = np.linspace(self._last_output_value, 0.0, num=needed, endpoint=False, dtype=np.float32)
                chunks.append(tail)
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
        if had_underrun and len(data) > 0:
            self._last_output_value = float(data[-1])
        elif len(data) > 0:
            # Light fade-in after recovery to avoid click at chunk boundaries.
            fade_n = min(len(data), max(1, int(self.sample_rate * 0.006)))
            fade = np.linspace(0.0, 1.0, num=fade_n, endpoint=True, dtype=np.float32)
            data[:fade_n] *= fade
            self._last_output_value = float(data[-1])

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

    def _worker_loop(self):
        while not self._stop_event.is_set():
            try:
                audio = self._input_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            with tempfile.TemporaryDirectory(prefix="rvc_sidecar_") as tmp:
                in_wav = Path(tmp) / "in.wav"
                out_wav = Path(tmp) / "out.wav"
                sf.write(str(in_wav), audio, self.sample_rate)

                try:
                    if self._rvc is None:
                        raise RuntimeError("RVC engine not initialized")
                    self._rvc.infer_file(str(in_wav), str(out_wav))
                    converted, out_sr = sf.read(str(out_wav), dtype="float32")
                except Exception as e:
                    self.status_message = f"rvc error: {e}"
                    # Keep audio flowing if inference fails for this chunk.
                    fallback = np.asarray(audio, dtype=np.float32)
                    if fallback.size > 0:
                        try:
                            self._output_queue.put(fallback, timeout=0.05)
                        except Exception:
                            pass
                    continue

                if isinstance(converted, np.ndarray) and converted.ndim > 1:
                    converted = converted[:, 0]
                converted = self._resample_linear(np.asarray(converted, dtype=np.float32), out_sr, self.sample_rate)

                # Blend in a small amount of dry mic to preserve consonants/intelligibility.
                src = np.asarray(audio, dtype=np.float32)
                if src.size > 0 and converted.size > 0 and self.dry_mix > 0.0:
                    if src.size != converted.size:
                        src = self._resample_linear(src, self.sample_rate, self.sample_rate)
                        if src.size != converted.size:
                            src_x = np.linspace(0.0, 1.0, num=src.size, endpoint=False)
                            dst_x = np.linspace(0.0, 1.0, num=converted.size, endpoint=False)
                            src = np.interp(dst_x, src_x, src).astype(np.float32)
                    wet = float(np.clip(1.0 - self.dry_mix, 0.0, 1.0))
                    dry = float(np.clip(self.dry_mix, 0.0, 1.0))
                    converted = np.clip((converted * wet) + (src * dry), -1.0, 1.0).astype(np.float32, copy=False)

                crossfade = max(16, int(self.sample_rate * self._crossfade_seconds))
                converted_len = int(converted.size)
                if converted_len > 0:
                    if self._pending_tail is None:
                        if converted_len > crossfade:
                            emit = converted[:-crossfade]
                            self._pending_tail = converted[-crossfade:].copy()
                        else:
                            self._pending_tail = converted.copy()
                            emit = np.zeros(0, dtype=np.float32)
                    else:
                        overlap = min(self._pending_tail.size, converted_len)
                        if overlap > 0:
                            fade_in = np.linspace(0.0, 1.0, num=overlap, endpoint=False, dtype=np.float32)
                            fade_out = 1.0 - fade_in
                            blended = (self._pending_tail[-overlap:] * fade_out) + (converted[:overlap] * fade_in)
                            remainder = converted[overlap:]
                            if remainder.size > crossfade:
                                emit = np.concatenate((blended, remainder[:-crossfade]))
                                self._pending_tail = remainder[-crossfade:].copy()
                            else:
                                emit = blended
                                self._pending_tail = remainder.copy()
                        else:
                            emit = converted
                            self._pending_tail = None
                else:
                    emit = converted

                chunk = max(1, int(self.sample_rate * 0.02))
                for start in range(0, len(emit), chunk):
                    if self._stop_event.is_set():
                        break
                    piece = emit[start:start + chunk]
                    try:
                        self._output_queue.put(piece, timeout=0.05)
                    except queue.Full:
                        try:
                            self._output_queue.get_nowait()
                            self._output_queue.put_nowait(piece)
                        except Exception:
                            pass

    def start(self, cfg: dict):
        with self.lock:
            if self.running:
                raise RuntimeError("Voice is already running")

            self.pitch = int(cfg.get("pitch", 0))
            self.index_rate = float(cfg.get("index_rate", 0.06))
            self.f0_method = str(cfg.get("f0_method", "harvest"))
            self.protect = float(cfg.get("protect", 0.80))
            self.sample_rate = max(8000, int(cfg.get("sample_rate", 24000)))
            self.block_seconds = max(0.08, min(1.2, float(cfg.get("block_seconds", 0.32))))
            self.noise_gate_enabled = bool(cfg.get("noise_gate_enabled", False))
            self.noise_gate_threshold = float(cfg.get("noise_gate_threshold", 0.008))
            self.noise_gate_hold_ms = int(cfg.get("noise_gate_hold_ms", 140))
            self.noise_gate_release_ms = int(cfg.get("noise_gate_release_ms", 70))
            self.dry_mix = float(np.clip(cfg.get("dry_mix", 0.42), 0.0, 0.65))
            self._noise_gate_hold_remaining = 0
            self._noise_gate_gain = 0.0
            self._pending_tail = None
            self._last_output_value = 0.0
            self._underrun_events = 0

            self.input_device = self._resolve_device_index(cfg.get("input_device"), "input")

            use_virtual = bool(cfg.get("use_virtual_output", False))
            output_device = cfg.get("output_device")
            if use_virtual and (output_device is None or str(output_device).strip().lower() in {"", "auto"}):
                self.output_device = self._pick_virtual_output_device()
                if self.output_device is None:
                    self.output_device = None
            else:
                self.output_device = self._resolve_device_index(output_device, "output")

            model_zip = Path(cfg.get("model_zip", str(DEFAULT_VOICE_MODEL_ZIP)))
            self.model_pth, self.model_index = self._extract_voice_model(model_zip)

            self.device = _select_rvc_device()
            if not self.device.startswith("cuda"):
                raise RuntimeError("GPU is required for voice worker, but CUDA is not available in venv310")
            self._init_rvc_runtime(self.device)

            while not self._input_queue.empty():
                self._input_queue.get_nowait()
            while not self._output_queue.empty():
                self._output_queue.get_nowait()

            self._stop_event.clear()
            self._worker_thread = threading.Thread(target=self._worker_loop, daemon=True)
            self._worker_thread.start()

            in_block = max(128, int(self.sample_rate * self.block_seconds))
            out_block = max(128, int(self.sample_rate * 0.02))

            self._input_stream = sd.InputStream(
                samplerate=self.sample_rate,
                channels=1,
                dtype="float32",
                blocksize=in_block,
                callback=self._input_callback,
                device=self.input_device,
            )
            self._output_stream = sd.OutputStream(
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
            self.status_message = f"running rvc -> {self._device_name(self.output_device) or 'system default'}"

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
        return {
            "running": self.running,
            "status": self.status_message,
            "backend": "rvc",
            "sample_rate": self.sample_rate,
            "block_seconds": self.block_seconds,
            "model_pth": str(self.model_pth) if self.model_pth else None,
            "model_index": str(self.model_index) if self.model_index else None,
            "pitch": self.pitch,
            "index_rate": self.index_rate,
            "f0_method": self.f0_method,
            "protect": self.protect,
            "dry_mix": self.dry_mix,
            "device": self.device,
            "noise_gate_enabled": self.noise_gate_enabled,
            "noise_gate_threshold": self.noise_gate_threshold,
            "noise_gate_hold_ms": self.noise_gate_hold_ms,
            "noise_gate_release_ms": self.noise_gate_release_ms,
            "input_device_index": self.input_device,
            "input_device_name": self._device_name(self.input_device),
            "output_device_index": self.output_device,
            "output_device_name": self._device_name(self.output_device),
            "using_virtual_output": self._is_virtual_device_name(self._device_name(self.output_device) or ""),
            "underrun_events": self._underrun_events,
            "worker_online": True,
        }


engine = WorkerVoiceEngine()
app = FastAPI()


@app.get("/health")
async def health():
    return {"ok": True}


@app.get("/voice/status")
async def voice_status():
    return engine.status()


@app.get("/voice/devices")
async def voice_devices():
    return engine.list_audio_devices()


@app.post("/voice/start")
async def voice_start(cfg: dict | None = None):
    cfg = cfg or {}
    try:
        if str(cfg.get("backend", "rvc")).lower() != "rvc":
            raise RuntimeError("Worker only supports backend=rvc")
        engine.start(cfg)
        return {"status": "success", "voice": engine.status()}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.post("/voice/stop")
async def voice_stop():
    try:
        engine.stop()
        return {"status": "success", "voice": engine.status()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/shutdown")
async def shutdown_worker():
    def _delayed_exit():
        time.sleep(0.25)
        os._exit(0)

    engine.stop()
    threading.Thread(target=_delayed_exit, daemon=True).start()
    return {"status": "shutting_down"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8011)
    args = parser.parse_args()

    import uvicorn

    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()
