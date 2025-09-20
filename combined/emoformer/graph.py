import matplotlib.pyplot as plt

# --- 1. Data Manually Copied from Your Screenshots ---
# I have extracted the loss and accuracy values from your terminal output.
train_loss_history = [
    1.6506, 1.0829, 0.7938, 0.6607, 0.6407, 0.6277, 0.5960, 0.5920, 0.5709, 0.5599,
    0.5322, 0.5079, 0.4846, 0.4768, 0.4569, 0.4544, 0.4544, 0.4357, 0.3935, 0.3635,
    0.3355, 0.3299, 0.2859, 0.2401, 0.2379
]

val_loss_history = [
    1.2673, 0.8228, 0.7167, 0.7337, 0.7377, 0.6636, 0.6219, 0.6431, 0.6465, 0.5657,
    0.5697, 0.5517, 0.5855, 0.5566, 0.5594, 0.5947, 0.5807, 0.5334, 0.5354, 0.5364,
    0.5068, 0.5209, 0.5815, 0.5405, 0.5994
]

val_accuracy_history = [
    47.41, 70.75, 74.65, 71.58, 73.82, 75.94, 74.76, 75.59, 76.65, 77.42,
    77.95, 79.60, 79.25, 79.13, 79.60, 78.89, 79.72, 80.28, 80.05, 79.95,
    81.01, 80.42, 81.72, 82.08, 82.31
]

NUM_EPOCHS = 25
GRAPH_SAVE_PATH = "emoformer_combined_graphs.png"

# --- 2. Code to Generate and Save the Plots ---
print(f"Generating training graphs from the provided data...")

epochs_range = range(1, NUM_EPOCHS + 1)

plt.figure(figsize=(12, 5))

# Plot for Loss
plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_loss_history, 'o-', label='Training Loss')
plt.plot(epochs_range, val_loss_history, 'o-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.grid(True)
plt.legend()

# Plot for Accuracy
plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_accuracy_history, 'o-', label='Validation Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(GRAPH_SAVE_PATH)

print(f"Training graphs saved successfully to {GRAPH_SAVE_PATH}")