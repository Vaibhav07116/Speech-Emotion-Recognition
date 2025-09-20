# In file: dataset.py

import os
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

def load_tess_data(data_path):
    """
    Scans the TESS dataset folder to create a DataFrame of file paths and emotions.
    """
    data = []
    emotion_folders = os.listdir(data_path)
    
    for folder in emotion_folders:
        try:
            # The emotion is the part after the underscore (e.g., OAF_angry -> angry)
            emotion = folder.split('_')[-1].lower()
            emotion_folder_path = os.path.join(data_path, folder)
            
            # Skip non-directory files if any
            if not os.path.isdir(emotion_folder_path):
                continue
            
            for file_name in os.listdir(emotion_folder_path):
                file_path = os.path.join(emotion_folder_path, file_name)
                data.append({"path": file_path, "emotion": emotion})
        except Exception as e:
            print(f"Error processing folder {folder}: {e}")
            
    return pd.DataFrame(data)

class TessDataset(Dataset):
    def __init__(self, data_frame, max_len=128):
        self.data_frame = data_frame
        
        # Create a mapping from emotion string to integer
        self.emotions = sorted(data_frame['emotion'].unique())
        self.emotion_to_int = {emotion: i for i, emotion in enumerate(self.emotions)}
        
        self.max_len = max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = self.data_frame.iloc[idx]['path']
        emotion_str = self.data_frame.iloc[idx]['emotion']
        
        # 1. Load audio and convert to spectrogram
        y, sr = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        # 2. Handle fixed length (padding or trimming)
        if log_mel_spectrogram.shape[1] > self.max_len:
            # Trim if longer
            log_mel_spectrogram = log_mel_spectrogram[:, :self.max_len]
        else:
            # Pad with zeros if shorter
            padding = self.max_len - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')

        # 3. Get the integer label for the emotion
        label = self.emotion_to_int[emotion_str]

        # Return as PyTorch tensors (transpose the spectrogram)
        return torch.tensor(log_mel_spectrogram.T, dtype=torch.float32), torch.tensor(label, dtype=torch.long)