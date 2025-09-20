import torch
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

# --- Import your models and dataset functions ---
# Make sure all three model classes are in your model.py file
from model import ViT
from dataset import load_tess_data, TessDataset

# --- 1. CHOOSE WHICH MODEL TO EVALUATE ---
# We have set this to 'ViT' for you.
MODEL_TYPE = 'ViT'

# --- 2. Configuration (the script will choose the right files) ---
if MODEL_TYPE == 'ViT':
    MODEL_PATH = "vit_tess.pth"
    MODEL_CLASS = ViT
    OUTPUT_IMAGE_PATH = "vit_confusion_matrix.png"
else:
    raise ValueError("Invalid MODEL_TYPE. Choose 'Emoformer', 'CNN+LSTM', or 'ViT'.")

TESS_DATA_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data"
device = "cpu"

# --- 3. Load Data (we only need the validation set) ---
print(f"Loading validation data...")
df = load_tess_data(TESS_DATA_PATH)
_, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
val_dataset = TessDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
EMOTION_LABELS = val_dataset.emotions
print("Data loaded.")

# --- 4. Load the Trained Model ---
print(f"Loading trained model: {MODEL_PATH}")

# Instantiate the correct model class based on MODEL_TYPE
if MODEL_TYPE == 'Emoformer':
    model = MODEL_CLASS(input_dim=128, dim=256, n_heads=8, n_blocks=4, ff_hidden_dim=1024, n_classes=len(EMOTION_LABELS))
else: # CNN+LSTM and ViT both just need the number of classes
    model = MODEL_CLASS(n_classes=len(EMOTION_LABELS))

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval() # Set model to evaluation mode (very important!)
print("Model loaded.")

# --- 5. Get All Predictions from the Validation Set ---
print("Getting predictions from the model...")
all_preds = []
all_labels = []

with torch.no_grad(): # Disable gradient calculations for speed
    for specs, labels in val_loader:
        specs, labels = specs.to(device), labels.to(device)
        outputs = model(specs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print("Prediction complete.")

# --- 6. Calculate and Display All Metrics ---

# Overall Accuracy
accuracy = accuracy_score(all_labels, all_preds)
print("\n--- Overall Performance Metrics ---")
print(f"Accuracy: {accuracy * 100:.2f}%")

# Classification Report (Precision, Recall, F1-Score)
print("\n--- Detailed Classification Report ---")
report = classification_report(all_labels, all_preds, target_names=EMOTION_LABELS, zero_division=0)
print(report)

# Confusion Matrix
print("Generating confusion matrix...")
cm = confusion_matrix(all_labels, all_preds)

plt.figure(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=EMOTION_LABELS, yticklabels=EMOTION_LABELS)
plt.title(f'Confusion Matrix - {MODEL_TYPE}')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH)

print(f"\nConfusion matrix image saved to {OUTPUT_IMAGE_PATH}")