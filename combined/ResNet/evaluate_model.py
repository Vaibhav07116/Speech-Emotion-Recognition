import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from tqdm import tqdm

from model import ResNetModel
from dataset import load_all_data, AudioDataset

# --- 1. Configuration ---
MODEL_PATH = "resnet_combined.pth"
RAVDESS_DATA_PATH = r"E:\Speech Emotion Recog\Dataset\ravdess"
TESS_DATA_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data"
OUTPUT_IMAGE_PATH = "resnet_combined_confusion_matrix.png"
device = "cpu"

# --- 2. Load Data ---
df = load_all_data(ravdess_path=RAVDESS_DATA_PATH, tess_path=TESS_DATA_PATH)
_, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
val_dataset = AudioDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
EMOTION_LABELS = val_dataset.emotions

# --- 3. Load the Trained Model ---
model = ResNetModel(n_classes=len(EMOTION_LABELS))
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# --- 4. Get All Predictions ---
all_preds, all_labels = [], []
with torch.no_grad():
    for specs, labels in tqdm(val_loader, desc="Evaluating"):
        specs, labels = specs.to(device), labels.to(device)
        outputs = model(specs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

# --- 5. Calculate and Display Metrics ---
accuracy = accuracy_score(all_labels, all_preds)
print("\n--- Overall Performance Metrics ---")
print(f"Accuracy: {accuracy * 100:.2f}%")
print("\n--- Detailed Classification Report ---")
report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0)
print(report)
cm = confusion_matrix(all_labels, all_preds)
plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.title('Confusion Matrix - ResNet Model')
plt.xlabel('Predicted Label'); plt.ylabel('True Label'); plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH)
print(f"\nConfusion matrix image saved to {OUTPUT_IMAGE_PATH}")