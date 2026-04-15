import gradio as gr
import torch
import numpy as np
import librosa
from model import GenderCNN
import os
import soundfile as sf
import imageio_ffmpeg

# Ensure ffmpeg binary is in the system PATH for librosa/audioread to decode .webm files
os.environ["PATH"] += os.pathsep + os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe())

MIN_DURATION_SEC = 1.5
SAMPLE_RATE      = 16000
N_MELS           = 64
MAX_LENGTH       = 128
SILENCE_THRESH   = 0.01

def load_model(model_path="model.pth"):
    model = GenderCNN()
    if os.path.exists(model_path):
        model.load_state_dict(
            torch.load(model_path, map_location=torch.device('cpu'), weights_only=True)
        )
        model.eval()
        print("Model loaded successfully.")
    else:
        print(f"Warning: {model_path} not found. Using untrained model.")
    return model

model = load_model()

def preprocess_audio(file_path: str):
    """
    Load audio from file path, validate, and return clean float32 mono waveform.
    Returns (waveform, error_message) — one of them will be None.
    """
    try:
        y, sr = librosa.load(file_path, sr=SAMPLE_RATE, mono=True)
    except Exception as e:
        return None, f"Could not read audio file: {e}"

    # Guard: too short
    min_samples = int(MIN_DURATION_SEC * SAMPLE_RATE)
    if len(y) < min_samples:
        return None, "Audio too short. Please record at least 1.5 seconds of speech."

    # Guard: silence
    rms = np.sqrt(np.mean(y ** 2))
    if rms < SILENCE_THRESH:
        return None, "Audio appears silent. Please speak clearly into the microphone."

    # Normalize to [-1, 1]
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    return y, None


def audio_to_melspec(y: np.ndarray) -> torch.Tensor:
    melspec = librosa.feature.melspectrogram(
        y=y, sr=SAMPLE_RATE, n_mels=N_MELS, fmax=8000
    )
    melspec = librosa.power_to_db(melspec, ref=np.max)

    if melspec.shape[1] < MAX_LENGTH:
        pad_width = MAX_LENGTH - melspec.shape[1]
        melspec = np.pad(
            melspec,
            pad_width=((0, 0), (0, pad_width)),
            mode='constant',
            constant_values=melspec.min()
        )
    else:
        melspec = melspec[:, :MAX_LENGTH]

    # Normalize spectrogram to [0, 1]
    melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-8)

    return torch.tensor(melspec, dtype=torch.float32).unsqueeze(0).unsqueeze(0)


def predict_gender(audio_path):
    """
    Main prediction function.
    audio_path: filepath string from Gradio's type="filepath"
    Returns: (result_text, audio_path) — audio_path fed back to Audio output for playback
    """
    if audio_path is None:
        return "No audio received. Please record or upload a voice clip.", None

    # Preprocess
    y_clean, error = preprocess_audio(audio_path)
    if error:
        return f"{error}", audio_path  # still return audio so user can hear what was captured

    # Feature extraction + inference
    melspec_tensor = audio_to_melspec(y_clean)

    with torch.no_grad():
        outputs = model(melspec_tensor)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)
        male_prob   = probabilities[0][0].item()
        female_prob = probabilities[0][1].item()

    confidence = max(male_prob, female_prob)
    warning = "\nLow confidence — try speaking longer or louder." if confidence < 0.65 else ""

    if male_prob > female_prob:
        result = f"Male\nConfidence: {male_prob:.1%}{warning}"
    else:
        result = f"Female\nConfidence: {female_prob:.1%}{warning}"

    return result, audio_path  # return audio_path for playback


# ── Gradio UI ──────────────────────────────────────────────────────────────────
with gr.Blocks(title="🎤 Voice Gender Recognition") as interface:
    gr.Markdown("# 🎤 Voice Gender Recognition")
    gr.Markdown(
        "Record at least **1–2 seconds** of clear speech or upload a `.wav` / `.mp3` file. "
        "Your recording will play back so you can verify it was captured correctly."
    )

    with gr.Row():
        with gr.Column():
            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="filepath",          # ← key fix: saves to temp file, enables playback
                label="🎙️ Record or Upload Voice"
            )
            submit_btn = gr.Button("Predict Gender", variant="primary")

        with gr.Column():
            playback = gr.Audio(
                label="Playback — verify your recording",
                interactive=False         # read-only, just for listening
            )
            result_box = gr.Textbox(
                label="Prediction",
                lines=3,
                placeholder="Result will appear here..."
            )

    submit_btn.click(
        fn=predict_gender,
        inputs=audio_input,
        outputs=[result_box, playback]
    )

if __name__ == "__main__":
    interface.launch(server_name="0.0.0.0", server_port=7860, share=True)