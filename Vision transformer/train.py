import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

from model import ViT

from dataset import load_tess_data, TessDataset

# --- 1. Configuration ---
TESS_DATA_PATH = r"E:\Speech Emotion Recog\Dataset\TESS Toronto emotional speech set data"
BATCH_SIZE = 32
NUM_EPOCHS = 25
LEARNING_RATE = 0.0001

MODEL_SAVE_PATH = "vit_tess.pth"
GRAPH_SAVE_PATH = "vit_training_graphs.png"


# --- 2. Load Data ---
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using {device} device")
df = load_tess_data(TESS_DATA_PATH)
train_df, val_df = train_test_split(df, test_size=0.2, random_state=42, stratify=df['emotion'])
train_dataset = TessDataset(train_df)
val_dataset = TessDataset(val_df)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 3. Initialize Model ---
N_CLASSES = len(train_dataset.emotions)
model = ViT(n_classes=N_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE)

# --- 4. Training Loop with History ---
train_loss_history, val_loss_history, val_accuracy_history = [], [], []
print(f"Starting training for {MODEL_SAVE_PATH}...")
for epoch in range(NUM_EPOCHS):
    model.train()
    total_train_loss = 0
    for specs, labels in train_loader:
        specs, labels = specs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(specs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()
    avg_train_loss = total_train_loss / len(train_loader)

    model.eval()
    total_val_loss, correct_predictions, total_samples = 0, 0, 0
    with torch.no_grad():
        for specs, labels in val_loader:
            specs, labels = specs.to(device), labels.to(device)
            outputs = model(specs)
            loss = criterion(outputs, labels)
            total_val_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total_samples += labels.size(0)
            correct_predictions += (predicted == labels).sum().item()
    avg_val_loss = total_val_loss / len(val_loader)
    accuracy = (correct_predictions / total_samples) * 100
    
    train_loss_history.append(avg_train_loss)
    val_loss_history.append(avg_val_loss)
    val_accuracy_history.append(accuracy)
    print(f"Epoch [{epoch+1}/{NUM_EPOCHS}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Accuracy: {accuracy:.2f}%")

# --- 5. Save Model and Graphs ---
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"\nTraining complete. Model saved to {MODEL_SAVE_PATH}")

epochs_range = range(1, NUM_EPOCHS + 1)
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_history, label='Training Loss')
plt.plot(epochs_range, val_loss_history, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs'); plt.ylabel('Loss'); plt.legend()
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracy_history, label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs'); plt.ylabel('Accuracy (%)'); plt.legend()
plt.tight_layout()
plt.savefig(GRAPH_SAVE_PATH)
print(f"Training graphs saved to {GRAPH_SAVE_PATH}") 