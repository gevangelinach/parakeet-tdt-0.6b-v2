import os
import tempfile
import setproctitle
import nemo.collections.asr as nemo_asr
import torch
import librosa
import soundfile as sf
from flask import Flask, request, jsonify

# === OPTIONAL BUT RECOMMENDED (Disable CUDA Graphs) ===
os.environ["NEMO_CONVERT_CUDA_GRAPHS"] = "0"

setproctitle.setproctitle("parakeet-tdt-0.6b-v2-stt")

# === CONFIG ===
MODEL_PATH = "/app/model/parakeet-tdt-0.6b-v2.nemo"
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...")

# === LOAD MODEL ===
asr_model = nemo_asr.models.ASRModel.restore_from(
    restore_path=MODEL_PATH,
    map_location=DEVICE
)
asr_model = asr_model.to(DEVICE)
asr_model.eval()
print("Model Loaded Successfully!")

# === SAFE WARMUP (NO transcribe() to avoid CUDA Graph issues) ===
print("Running safe warmup inference...")
try:
    with torch.inference_mode():
        dummy_audio = torch.randn(1, 16000).to(DEVICE)
        dummy_len = torch.tensor([16000]).to(DEVICE)

        # Use forward() instead of transcribe() to avoid CUDA graph replay errors
        _ = asr_model.forward(
            input_signal=dummy_audio,
            input_signal_length=dummy_len
        )

    print("Warmup complete. Model ready!")

except Exception as warmup_error:
    print(f"Warmup skipped due to: {warmup_error}")

# === FLASK APP ===
app = Flask(__name__)


@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing audio file in request"}), 400

    file = request.files["file"]
    if not file.filename:
        return jsonify({"error": "Empty filename"}), 400

    uploaded_path = None
    mono_path = None

    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            uploaded_path = tmp.name

        # Convert to 16kHz mono
        y, _ = librosa.load(uploaded_path, sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as mono_tmp:
            sf.write(mono_tmp.name, y, 16000)
            mono_path = mono_tmp.name

        # Perform transcription
        result = asr_model.transcribe(
            [mono_path],
            batch_size=BATCH_SIZE
        )

        # Robust extractor: supports .text or raw string output
        first = result[0]
        text = getattr(first, "text", str(first)).strip()

        return jsonify({"transcription": text}), 200

    except Exception as e:
        return jsonify({
            "error": "Transcription failed",
            "details": str(e)
        }), 500

    finally:
        # Always clean temporary files safely
        for path in [uploaded_path, mono_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def index():
    return "Parakeet TDT 0.6B v2 STT API is running!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5023)