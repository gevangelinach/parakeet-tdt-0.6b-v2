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
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading model on {DEVICE}...")

# === LOAD MODEL ===
asr_model = nemo_asr.models.ASRModel.restore_from(restore_path=MODEL_PATH, map_location=DEVICE)
asr_model.to(DEVICE)
asr_model.eval()

print("Model Loaded!")

# === FLASK APP ===
app = Flask(__name__)

@app.route("/transcribe", methods=["POST"])
def transcribe():
    if "file" not in request.files:
        return jsonify({"error": "Missing file"}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "Empty file"}), 400

    uploaded_path = None
    mono_path = None
    try:
        # Save uploaded file
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
            file.save(tmp.name)
            uploaded_path = tmp.name

        # Convert to 16k mono
        y, _ = librosa.load(uploaded_path, sr=16000, mono=True)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as mono_tmp:
            sf.write(mono_tmp.name, y, 16000)
            mono_path = mono_tmp.name

        # Nemo transcribe
        text_list = asr_model.transcribe([mono_path])
        text = getattr(text_list[0], "text", str(text_list[0])).strip()

        return jsonify({"transcription": text}), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

    finally:
        # Clean temp files safely
        for path in [uploaded_path, mono_path]:
            if path and os.path.exists(path):
                os.unlink(path)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "healthy"}), 200


@app.route("/", methods=["GET"])
def index():
    return "Parakeet TDT 0.6B v2 STT API is running!", 200


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5023)
