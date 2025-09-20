import torch
import librosa
import numpy as np
import os
from model import ResNetModel

# --- 1. CONFIGURATION ---
MODEL_PATH = "resnet_combined.pth"
TEST_AUDIO_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data\YAF_angry\YAF_bar_angry.wav"
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

# =======================================================
device = "cpu"

# --- 2. Load the trained model ---
model = ResNetModel(n_classes=len(EMOTION_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- 3. Prediction function ---
def predict_emotion(audio_path, model, max_len=128):
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    if log_mel_spec.shape[1] > max_len:
        log_mel_spec = log_mel_spec[:, :max_len]
    else:
        padding = max_len - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, padding)), mode='constant')
        
    spec_tensor = torch.tensor(log_mel_spec.T, dtype=torch.float32).unsqueeze(0).to(device)
    
    with torch.no_grad():
        output = model(spec_tensor)
        _, predicted_idx = torch.max(output, 1)
    
    predicted_emotion = EMOTION_LABELS[predicted_idx.item()]
    return predicted_emotion

# --- 4. Run the prediction ---
if __name__ == '__main__':
    if os.path.exists(TEST_AUDIO_PATH):
        predicted_emotion = predict_emotion(TEST_AUDIO_PATH, model)
        print("\n--- Prediction Result ---")
        print(f"The predicted emotion for the audio file is: {predicted_emotion}")
    else:
        print(f"\nError: The file '{TEST_AUDIO_PATH}' was not found.")