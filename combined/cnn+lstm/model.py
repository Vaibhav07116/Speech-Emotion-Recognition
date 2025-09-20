import torch
import torch.nn as nn

class CnnLstmModel(nn.Module):
    def __init__(self, n_classes):
        super().__init__()
        
        # --- CNN "Eyes" Part ---
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(16),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(32),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        # --- LSTM "Ears" Part ---
        self.lstm = nn.LSTM(
            input_size=1024,  # Features from CNN (64 channels * 16 height)
            hidden_size=128,
            num_layers=2,
            batch_first=True,
            bidirectional=True
        )
        
        # --- Final Classifier ---
        self.classifier = nn.Linear(128 * 2, n_classes)

    def forward(self, x):
        # Input x has shape: (batch_size, seq_length, n_mels)
        x = x.unsqueeze(1)    # Shape becomes: (batch, 1, seq_len, n_mels)
        x = x.permute(0, 1, 3, 2) # Shape becomes: (batch, 1, n_mels, seq_len)
        
        x = self.cnn_layers(x) # Shape: (batch, 64, 16, 16)
        
        x = x.permute(0, 3, 1, 2) # Shape: (batch, 16, 64, 16)
        batch_size, seq_len, channels, height = x.shape
        x = x.reshape(batch_size, seq_len, channels * height) # Shape: (batch, 16, 1024)
        
        x, _ = self.lstm(x) # Shape: (batch, 16, 256)
        
        x = x[:, -1, :] # Shape: (batch, 256)
        
        x = self.classifier(x)
        return x

# You can also add your other model classes (Emoformer, ResNet, etc.) in this file