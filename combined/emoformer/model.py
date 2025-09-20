import torch
import torch.nn as nn

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Linear(hidden_dim, dim),
        )
    def forward(self, x):
        return self.net(x)

class EmoformerBlock(nn.Module):
    def __init__(self, dim, n_heads, ff_hidden_dim):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim=dim, num_heads=n_heads)
        self.feed_forward = FeedForward(dim, ff_hidden_dim)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        attn_output, _ = self.attention(x, x, x)
        x = self.norm1(x + attn_output)
        ff_output = self.feed_forward(x)
        x = self.norm2(x + ff_output)
        return x

class Emoformer(nn.Module):
    def __init__(self, input_dim, dim, n_heads, n_blocks, ff_hidden_dim, n_classes):
        super().__init__()
        self.patch_embed = nn.Linear(input_dim, dim)
        self.local_blocks = nn.ModuleList(
            [EmoformerBlock(dim, n_heads, ff_hidden_dim) for _ in range(n_blocks)]
        )
        self.classifier = nn.Sequential(
            nn.Linear(dim, ff_hidden_dim), nn.GELU(), nn.Linear(ff_hidden_dim, n_classes)
        )

    def forward(self, x):
        # x is a batch of spectrograms: (batch_size, seq_length, n_mels)
        x = self.patch_embed(x)
        # PyTorch attention expects (seq_length, batch_size, dim)
        x = x.permute(1, 0, 2)
        for block in self.local_blocks:
            x = block(x)
        # Global average pooling
        x = x.mean(dim=0)
        return self.classifier(x)