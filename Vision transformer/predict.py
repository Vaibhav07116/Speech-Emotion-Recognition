import torch
import librosa
import numpy as np
import os

# --- Import your model classes ---
from model import ViT

MODEL_TYPE = 'ViT'

# Enter the full path to the audio file you want to test
TEST_AUDIO_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data\YAF_angry\YAF_bar_angry.wav"

# Make sure this list matches the one your model was trained on!
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'surprised']

# ==============================================================================

device = "cpu"

# --- 2. Load the trained model based on the chosen type ---
print(f"Loading model: {MODEL_TYPE}...")

if MODEL_TYPE == 'ViT':
    MODEL_PATH = "vit_tess.pth"
    model = ViT(n_classes=len(EMOTION_LABELS))
else:
    raise ValueError("Invalid MODEL_TYPE chosen. Please choose 'Emoformer', 'CNN+LSTM', or 'ViT'.")

# Load the saved weights into the model
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set the model to evaluation mode
print("Model loaded successfully.")


# --- 3. Prediction function ---
def predict_emotion(audio_path, model, max_len=128):
    """
    Takes an audio file path and a trained model, and returns the predicted emotion.
    """
    # 1. Load audio and create spectrogram
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # 2. Pad or trim to the fixed length the model was trained on
    if log_mel_spec.shape[1] > max_len:
        log_mel_spec = log_mel_spec[:, :max_len]
    else:
        padding = max_len - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, padding)), mode='constant')
        
    # 3. Prepare the tensor for the model
    # Transpose to (seq_len, n_mels), add batch dimension, and move to device
    spec_tensor = torch.tensor(log_mel_spec.T, dtype=torch.float32).unsqueeze(0).to(device)
    
    # 4. Get prediction
    with torch.no_grad():
        output = model(spec_tensor)
        
    # 5. Find the winning class
    _, predicted_idx = torch.max(output, 1)
    
    # 6. Map the index back to the emotion label
    predicted_emotion = EMOTION_LABELS[predicted_idx.item()]
    
    return predicted_emotion


# --- 4. Run the prediction ---
if __name__ == '__main__':
    # Check if the specified audio file exists
    if os.path.exists(TEST_AUDIO_PATH):
        # Call the prediction function
        predicted_emotion = predict_emotion(TEST_AUDIO_PATH, model)
        print("\n--- Prediction Result ---")
        print(f"The predicted emotion for the audio file is: {predicted_emotion}")
    else:
        print(f"\nError: The file '{TEST_AUDIO_PATH}' was not found.")
        print("Please update the TEST_AUDIO_PATH variable at the top of the script.")