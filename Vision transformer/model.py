# In file: model.py

import torch
import torch.nn as nn

# You can keep your CnnLstmModel and Emoformer classes in this file too

class PatchEmbedding(nn.Module):
    """Cuts the image into patches and flattens them."""
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x: (batch, channels, height, width)
        x = self.proj(x)  # (batch, embed_dim, n_patches**0.5, n_patches**0.5)
        x = x.flatten(2)  # (batch, embed_dim, n_patches)
        x = x.transpose(1, 2)  # (batch, n_patches, embed_dim)
        return x

class ViT(nn.Module):
    """Vision Transformer Model."""
    def __init__(self, img_size=128, patch_size=16, in_channels=1, embed_dim=768, n_heads=12, n_layers=12, n_classes=8):
        super().__init__()
        
        # 1. Patch Embedding
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        
        # 2. CLS Token and Positional Embedding
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, 1 + self.patch_embed.n_patches, embed_dim))
        
        # 3. Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=n_heads, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        
        # 4. Classifier Head
        self.classifier = nn.Linear(embed_dim, n_classes)

    def forward(self, x):
        # Input x: (batch_size, seq_len, n_mels) which is (batch, 128, 128)
        # Add a channel dimension for the CNN in patch embedding
        x = x.unsqueeze(1) # (batch, 1, 128, 128)
        
        n_samples = x.shape[0]
        x = self.patch_embed(x) # (batch, n_patches, embed_dim)
        
        # Prepend the CLS token
        cls_token = self.cls_token.expand(n_samples, -1, -1)
        x = torch.cat((cls_token, x), dim=1) # (batch, 1 + n_patches, embed_dim)
        
        # Add positional embedding
        x = x + self.pos_embed
        
        # Pass through Transformer Encoder
        x = self.transformer_encoder(x)
        
        # Get the CLS token output for classification
        cls_output = x[:, 0]
        
        # Pass through classifier
        output = self.classifier(cls_output)
        return output