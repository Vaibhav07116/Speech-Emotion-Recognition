import os
import pandas as pd
import torch
import librosa
import numpy as np
from torch.utils.data import Dataset

def load_all_data(ravdess_path, tess_path):
    """
    Scans both TESS and RAVDESS dataset folders to create a combined DataFrame.
    It harmonizes the emotion labels into a consistent set.
    """
    data = []
    
    # --- 1. Process RAVDESS Dataset ---
    if os.path.exists(ravdess_path):
        # Map RAVDESS emotion codes to labels. 'calm' is merged into 'neutral'.
        ravdess_emotion_map = {
            "01": "neutral", "02": "neutral", 
            "03": "happy",   "04": "sad",
            "05": "angry",   "06": "fear",
            "07": "disgust", "08": "surprise"
        }
        for actor_folder in os.listdir(ravdess_path):
            actor_path = os.path.join(ravdess_path, actor_folder)
            if os.path.isdir(actor_path):
                for file_name in os.listdir(actor_path):
                    parts = file_name.split('.')[0].split('-')
                    if len(parts) > 2:
                        emotion_code = parts[2]
                        if emotion_code in ravdess_emotion_map:
                            emotion = ravdess_emotion_map[emotion_code]
                            data.append({"path": os.path.join(actor_path, file_name), "emotion": emotion})

    # --- 2. Process TESS Dataset ---
    if os.path.exists(tess_path):
        for folder in os.listdir(tess_path):
            try:
                emotion = folder.split('_')[-1].lower()
                # Standardize 'ps' and 'surprised' to the single label 'surprise'
                if emotion == 'ps' or emotion == 'surprised':
                    emotion = 'surprise'
                
                emotion_folder_path = os.path.join(tess_path, folder)
                if os.path.isdir(emotion_folder_path):
                    for file_name in os.listdir(emotion_folder_path):
                        file_path = os.path.join(emotion_folder_path, file_name)
                        data.append({"path": file_path, "emotion": emotion})
            except Exception as e:
                print(f"Skipping TESS folder {folder} due to error: {e}")
            
    return pd.DataFrame(data)


class AudioDataset(Dataset):
    """A generic audio dataset class for spectrograms."""
    def __init__(self, data_frame, max_len=128):
        self.data_frame = data_frame
        self.emotions = sorted(data_frame['emotion'].unique())
        self.emotion_to_int = {emotion: i for i, emotion in enumerate(self.emotions)}
        self.max_len = max_len

    def __len__(self):
        return len(self.data_frame)

    def __getitem__(self, idx):
        audio_path = self.data_frame.iloc[idx]['path']
        emotion_str = self.data_frame.iloc[idx]['emotion']
        
        y, sr = librosa.load(audio_path, sr=None)
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        log_mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)
        
        if log_mel_spectrogram.shape[1] > self.max_len:
            log_mel_spectrogram = log_mel_spectrogram[:, :self.max_len]
        else:
            padding = self.max_len - log_mel_spectrogram.shape[1]
            log_mel_spectrogram = np.pad(log_mel_spectrogram, ((0, 0), (0, padding)), mode='constant')

        label = self.emotion_to_int[emotion_str]
        
        return torch.tensor(log_mel_spectrogram.T, dtype=torch.float32), torch.tensor(label, dtype=torch.long)