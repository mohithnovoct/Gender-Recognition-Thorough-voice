# Voice Gender Recognition

This project is a Deep Learning-based Speech Recognition system built with PyTorch. It actively classifies the gender of a speaker (Male/Female) by analyzing Voice Audio data utilizing Mel Spectrogram representations and a Convolutional Neural Network (CNN).

It features a streamlined backend data processor and an interactive web interface built using Gradio, allowing real-time recordings from a microphone or dataset uploads!

## Features
- **Mel Spectrogram Extraction:** Audio tracks are loaded and dynamically converted into images (mel spectrograms). This creates a highly accurate 2D structural footprint of speech characteristics.
- **Deep Convolutional Neural Network (CNN):** A custom PyTorch model classifies the extracted spectrogram natively, without relying on external APIs.
- **Gradio Interactive App:** An out-of-the-box web UI allows users to easily capture audio from their microphone, visualize the playback, and instantly see the model's confidence prediction. 
- **Standalone Audio Handling:** Ships with an embedded ffmpeg decoder (`imageio-ffmpeg`) for maximum cross-platform browser support (.webm audio parsing included!). 

## Requirements
Python 3.8+ is recommended. All major dependencies can be found below:

- `torch` & `torchaudio`
- `numpy`
- `librosa`
- `soundfile`
- `gradio`
- `imageio-ffmpeg`

## Installation

1. Prepare a local environment (Recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

2. Install the necessary dependencies:
```bash
pip install torch torchaudio numpy librosa soundfile gradio imageio-ffmpeg
```

3. Ensure that your dataset is placed structurally. You should have `.wav` files inside the `data/male` and `data/female` directories. 
```
Speech Recognition/
│
├── data/
│   ├── male/
│   │   └── audio_1.wav
│   └── female/
│       └── audio_2.wav
│
├── dataset.py
├── model.py
├── train.py
└── app.py
```

## Usage

### 1. Training the Model
To populate your neural network weights (`model.pth`), execute the training script. This script processes your `data/` folder, compiles the dataset, and runs through the Deep Learning epochs.
```bash
python train.py
```
> The script will split the data 80-20 for validation and automatically save the `model.pth` file into your workspace upon completion.

### 2. Running the UI (Inference)
Once the model is successfully trained, launch the interactive user interface! 
```bash
python app.py
```
This deploys a local web server (usually at `http://localhost:7860`). Open that URL in any modern web browser to access the Microphone capability, and start testing!
