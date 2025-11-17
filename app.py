import os
import tempfile
import setproctitle
import nemo.collections.asr as nemo_asr
import torch
import numpy as np
import soundfile as sf
import resampy
from flask import Flask, request, jsonify

# ==========================
#  SYSTEM LEVEL OPTIMIZATIONS
# ==========================

# Disable CUDA Graphs (fixes replay errors)
os.environ["NEMO_CONVERT_CUDA_GRAPHS"] = "0"

# Faster GPU kernels
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.benchmark = True

# Use RAM disk for temporary files (VERY FAST)
TMP_DIR = "/dev/shm"
if not os.path.exists(TMP_DIR):
    TMP_DIR = tempfile.gettempdir()  # fallback if shm unavailable


setproctitle.setproctitle("parakeet-tdt-0.6b-v2-stt")

# ==========================
#  CONFIG
# ==========================
MODEL_PATH = "/app/model/parakeet-tdt-0.6b-v2.nemo"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_FP16 = True  # Enable FP16 acceleration if GPU supports it

print(f"Loading model on: {DEVICE}")


# ==========================
#  LOAD MODEL ONCE
# ==========================
asr_model = nemo_asr.models.ASRModel.restore_from(
    restore_path=MODEL_PATH,
    map_location=DEVICE
)

# Move to GPU
asr_model = asr_model.to(DEVICE)

# FP16 acceleration
if USE_FP16 and DEVICE == "cuda":
    asr_model = asr_model.half()
    print("Model converted to FP16 mode for maximum speed.")

asr_model.eval()

print("Model Loaded Successfully!")


# ==========================
#  SAFE WARMUP (NO transcribe())
# ==========================
print("Running warmup inference...")

try:
    with torch.inference_mode():
        # Multiple lengths warmup (1s, 3s, 6s)
        warmup_lengths = [16000, 48000, 96000]

        for L in warmup_lengths:
            dummy = torch.randn(1, L).to(DEVICE)
            if USE_FP16 and DEVICE == "cuda":
                dummy = dummy.half()

            dummy_len = torch.tensor([L]).to(DEVICE)
            _ = asr_model.forward(
                input_signal=dummy,
                input_signal_length=dummy_len
            )

    print("Warmup complete. Model ready!")

except Exception as warm_err:
    print(f"Warmup skipped due to: {warm_err}")


# ==========================
#  FLASK APP
# ==========================
app = Flask(__name__)


def fast_load_audio(path):
    """Load + resample + mono convert faster than librosa."""
    audio, sr = sf.read(path)

    # Convert stereo → mono
    if isinstance(audio, np.ndarray) and audio.ndim > 1:
        audio = audio.mean(axis=1)

    # Resample if needed
    if sr != 16000:
        audio = resampy.resample(audio, sr, 16000)

    return audio.astype(np.float32)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing audio file"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    uploaded_path = None

    try:
        # Save in RAM disk → super fast
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir=TMP_DIR) as tmp:
            file.save(tmp.name)
            uploaded_path = tmp.name

        # Fast audio loading
        audio = fast_load_audio(uploaded_path)

        # Write mono + resampled audio to another temp file
        mono_path = uploaded_path + "_mono.wav"
        sf.write(mono_path, audio, 16000)

        # Run transcription
        result = asr_model.transcribe(
            [mono_path],
            batch_size=BATCH_SIZE
        )

        # Supports both Nemo "text" object or raw string
        first = result[0]
        text = getattr(first, "text", str(first)).strip()

        return jsonify({"transcription": text}), 200

    except Exception as e:
        return jsonify({
            "error": "Transcription failed",
            "details": str(e)
        }), 500

    finally:
        # ALWAYS clean files
        for p in [uploaded_path, uploaded_path + "_mono.wav" if uploaded_path else None]:
            try:
                if p and os.path.exists(p):
                    os.unlink(p)
            except:
                pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def index():
    return "🔥 Parakeet TDT 0.6B v2 — Ultra-Optimized STT API Running!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5023)