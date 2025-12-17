import torch
import torch.nn as nn

class PatchEmbed(nn.Module):
    def __init__(self, img_size=28, patch_size=7, in_chans=1, embed_dim=64):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                 # (B, embed_dim, H/ps, W/ps)
        x = x.flatten(2).transpose(1, 2) # (B, N, embed_dim)
        return x


class ViTTiny(nn.Module):
    def __init__(self, num_classes=47, embed_dim=64, depth=4, num_heads=4):
        super().__init__()

        self.patch_embed = PatchEmbed()
        num_patches = (28 // 7) ** 2

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=128,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.norm = nn.LayerNorm(embed_dim)
        self.fc = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed

        x = self.encoder(x)
        x = self.norm(x)

        cls_out = x[:, 0]
        return self.fc(cls_out)
