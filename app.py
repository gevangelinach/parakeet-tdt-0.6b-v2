import os
import tempfile
import setproctitle
import nemo.collections.asr as nemo_asr
import torch
import librosa
import soundfile as sf
from flask import Flask, request, jsonify

setproctitle.setproctitle("parakeet-tdt-0.6b-v2-stt")

# === CONFIG ===
MODEL_PATH = "/app/model/parakeet-tdt-0.6b-v2.nemo"
BATCH_SIZE = 10
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

# === WARMUP ===
print("Running warmup inference...")
with torch.inference_mode():
    dummy_audio = torch.randn(1, 16000).to(DEVICE)
    dummy_len = torch.tensor([16000]).to(DEVICE)
    try:
        _ = asr_model.transcribe(
            audio=dummy_audio,
            length=dummy_len,
            partial_audio=True
        )
        print("Warmup complete. Model ready!")
    except Exception as warmup_error:
        print(f"Warmup failed: {warmup_error}")
        # Still allow app to run


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

        # Run transcription
        result = asr_model.transcribe(
            [mono_path],
            batch_size=BATCH_SIZE
        )

        # Robust: Handle both raw strings or objects with .text
        first = result[0]
        text = getattr(first, "text", str(first)).strip()

        return jsonify({"transcription": text}), 200

    except Exception as e:
        return jsonify({
            "error": "Transcription failed",
            "details": str(e)
        }), 500

    finally:
        # Clean all temporary files no matter what
        for path in [uploaded_path, mono_path]:
            try:
                if path and os.path.exists(path):
                    os.unlink(path)
            except Exception:
                pass  # Avoid crashing due to cleanup failure


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def index():
    return "Parakeet TDT 0.6B v2 STT API is running!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5023)