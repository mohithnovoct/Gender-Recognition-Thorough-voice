import os
import glob
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset, DataLoader

def extract_features(file_path, max_length=128):
    """
    Extract Mel-Spectrogram features from audio file.
    Returns array of shape (n_mels, max_length)
    """
    try:
        # Load audio
        y, sr = librosa.load(file_path, sr=16000, mono=True)
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        y, sr = np.zeros(16000), 16000

    # Normalize to [-1, 1]
    peak = np.abs(y).max()
    if peak > 0:
        y = y / peak

    # Extract Mel-spectrogram
    melspec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=64, fmax=8000)
    melspec = librosa.power_to_db(melspec, ref=np.max)
    
    # Pad or truncate to max_length
    if melspec.shape[1] < max_length:
        pad_width = max_length - melspec.shape[1]
        melspec = np.pad(melspec, pad_width=((0, 0), (0, pad_width)), mode='constant', constant_values=melspec.min())
    else:
        melspec = melspec[:, :max_length]

    # Normalize spectrogram to [0, 1]
    melspec = (melspec - melspec.min()) / (melspec.max() - melspec.min() + 1e-8)
        
    return melspec

class GenderDataset(Dataset):
    def __init__(self, data_dir, max_length=128):
        self.file_paths = []
        self.labels = []
        self.max_length = max_length
        
        # 0 for male, 1 for female
        classes = {'male': 0, 'female': 1}
        
        for cls_name, cls_label in classes.items():
            cls_dir = os.path.join(data_dir, cls_name)
            if not os.path.isdir(cls_dir):
                print(f"Directory not found: {cls_dir}")
                continue
            for fname in os.listdir(cls_dir):
                if fname.endswith('.wav'):
                    self.file_paths.append(os.path.join(cls_dir, fname))
                    self.labels.append(cls_label)
                    
    def __len__(self):
        return len(self.file_paths)
        
    def __getitem__(self, idx):
        file_path = self.file_paths[idx]
        label = self.labels[idx]
        
        melspec = extract_features(file_path, self.max_length)
        # Add channel dimension (1, n_mels, max_length)
        melspec = np.expand_dims(melspec, axis=0)
        
        return torch.tensor(melspec, dtype=torch.float32), torch.tensor(label, dtype=torch.long)
