import torch
import numpy as np
import matplotlib.pyplot as plt
from itertools import cycle
from tqdm import tqdm

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from torch.utils.data import DataLoader

# --- Import all your models and the TESS-specific dataset functions ---
from model import ViT
from dataset import load_tess_data, TessDataset

# --- 1. CONFIGURATION: CHOOSE WHICH MODEL TO EVALUATE ---
MODEL_TYPE = 'ViT'

# --- 2. Paths and Parameters ---
# Set the correct paths based on the chosen model
if MODEL_TYPE == 'ViT':
    MODEL_PATH = "vit_tess.pth"
    MODEL_CLASS = ViT
else:
    raise ValueError("Invalid MODEL_TYPE selected.")

OUTPUT_IMAGE_PATH = f"roc_curves_{MODEL_TYPE.lower()}_tess.png"
# This path should point directly to your TESS dataset folder
TESS_DATA_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data"
device = "cpu"

# --- 3. Load TESS-only Data ---
print("Loading TESS validation data...")
df = load_tess_data(TESS_DATA_PATH)
_, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
val_dataset = TessDataset(val_df)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
EMOTION_LABELS = val_dataset.emotions
N_CLASSES = len(EMOTION_LABELS)

# --- 4. Load the Trained Model ---
print(f"Loading trained model: {MODEL_PATH}")
# Instantiate the correct model class
if MODEL_TYPE == 'Emoformer':
    model = MODEL_CLASS(input_dim=128, dim=256, n_heads=8, n_blocks=4, ff_hidden_dim=1024, n_classes=N_CLASSES)
else: # For ResNet, CNN+LSTM
    model = MODEL_CLASS(n_classes=N_CLASSES)

model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()
print("Model loaded.")

# --- 5. Get Prediction Scores (Probabilities) ---
print("Getting model predictions...")
all_labels = []
all_scores = []
with torch.no_grad():
    for specs, labels in tqdm(val_loader, desc="Evaluating"):
        specs = specs.to(device)
        outputs = model(specs)
        # Apply softmax to convert logits to probabilities
        scores = torch.nn.functional.softmax(outputs, dim=1)
        
        all_labels.extend(labels.numpy())
        all_scores.extend(scores.cpu().numpy())

all_labels = np.array(all_labels)
all_scores = np.array(all_scores)

# --- 6. Calculate ROC Curve and AUC for each class ---
print("Calculating ROC and AUC...")
# Binarize the labels for One-vs-Rest calculation
y_bin = label_binarize(all_labels, classes=range(N_CLASSES))

# Dictionaries to store ROC curve data
fpr = dict()
tpr = dict()
roc_auc = dict()

for i in range(N_CLASSES):
    fpr[i], tpr[i], _ = roc_curve(y_bin[:, i], all_scores[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

# --- 7. Plot All ROC Curves ---
print("Generating ROC plot...")
plt.figure(figsize=(12, 10))
colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'green', 'red', 'purple', 'brown'])

for i, color in zip(range(N_CLASSES), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=2,
             label=f'ROC curve for {EMOTION_LABELS[i]} (AUC = {roc_auc[i]:.2f})')

# Plot the random guess line
plt.plot([0, 1], [0, 1], 'k--', lw=2, label='Random Guess')

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (FPR)')
plt.ylabel('True Positive Rate (TPR)')
plt.title(f'Receiver Operating Characteristic (ROC) - {MODEL_TYPE} Model (TESS Only)')
plt.legend(loc="lower right")
plt.grid(True)
plt.tight_layout()
plt.savefig(OUTPUT_IMAGE_PATH)

print(f"\nROC curve plot saved to {OUTPUT_IMAGE_PATH}")
for i in range(N_CLASSES):
    print(f"AUC for {EMOTION_LABELS[i]}: {roc_auc[i]:.4f}")