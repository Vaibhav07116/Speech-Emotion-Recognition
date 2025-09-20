import torch
import librosa
import numpy as np
import os

# Make sure your Emoformer class is in model.py
from model import Emoformer

# --- 1. CONFIGURATION: EDIT THESE VARIABLES ---
# Path to the model you trained on the combined dataset
MODEL_PATH = "emoformer_combined.pth"

# Path to the single audio file you want to test
TEST_AUDIO_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data\YAF_angry\YAF_bar_angry.wav"

# The harmonized list of 7 emotions for the combined dataset
EMOTION_LABELS = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise', 'surprised']
# =======================================================

device = "cpu"

# --- 2. Load the trained model ---
print(f"Loading model: {MODEL_PATH}...")

# Instantiate the model with the correct number of classes (7)
model = Emoformer(
    input_dim=128,
    dim=256,
    n_heads=8,
    n_blocks=4,
    ff_hidden_dim=1024,
    n_classes=len(EMOTION_LABELS)
)

# Load the saved weights
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set the model to evaluation mode
print("Model loaded successfully.")

# --- 3. Prediction function ---
def predict_emotion(audio_path, model, max_len=128):
    """
    Takes an audio file path and a trained model, and returns the predicted emotion.
    """
    # Load audio and create a Mel-spectrogram
    y, sr = librosa.load(audio_path, sr=None)
    mel_spec = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
    
    # Pad or trim to the fixed length the model was trained on
    if log_mel_spec.shape[1] > max_len:
        log_mel_spec = log_mel_spec[:, :max_len]
    else:
        padding = max_len - log_mel_spec.shape[1]
        log_mel_spec = np.pad(log_mel_spec, ((0, 0), (0, padding)), mode='constant')
        
    # Prepare the tensor for the model
    spec_tensor = torch.tensor(log_mel_spec.T, dtype=torch.float32).unsqueeze(0).to(device)
    
    # Get the prediction
    with torch.no_grad():
        output = model(spec_tensor)
        
    # Find the class with the highest score
    _, predicted_idx = torch.max(output, 1)
    
    # Map the index back to the emotion label
    predicted_emotion = EMOTION_LABELS[predicted_idx.item()]
    
    return predicted_emotion

# --- 4. Run the prediction ---
if __name__ == '__main__':
    # Check if the audio file exists before trying to process it
    if os.path.exists(TEST_AUDIO_PATH):
        # Call the prediction function
        predicted_emotion = predict_emotion(TEST_AUDIO_PATH, model)
        print("\n--- Prediction Result ---")
        print(f"The predicted emotion for the audio file is: {predicted_emotion}")
    else:
        print(f"\nError: The file '{TEST_AUDIO_PATH}' was not found.")
        print("Please update the TEST_AUDIO_PATH variable at the top of the script.")