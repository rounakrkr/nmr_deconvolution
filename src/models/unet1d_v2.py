"""
MixNet V2 — U-Net with self-attention across the 16 bottleneck spectral positions.
The mixtures are stacked as input channels; no attention operates across mixtures (see V3).
"""
import torch
import torch.nn as nn
from src.models.blocks import ConvBlock, DownBlock, UpBlock


class BottleneckAttention(nn.Module):
    """
    Attention at bottleneck to capture cross-position correlations.
    After encoding, the bottleneck is (B, 512, 16).
    Attention lets each of the 16 positions look at all others —
    finding which positions have correlated patterns (= same compound!).
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, 
            dim_feedforward=1024, dropout=0.1,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
    
    def forward(self, x):
        # x: (B, C=512, L=16)
        x_t = x.permute(0, 2, 1)   # (B, 16, 512) — each position is a token
        x_t = self.transformer(x_t) # Attention across 16 positions
        x_t = self.norm(x_t)
        return x_t.permute(0, 2, 1) # (B, 512, 16)


class MixNetV2(nn.Module):
    """
    MixNet with bottleneck self-attention over spectral positions.
    
    Same U-Net backbone + Transformer attention at bottleneck.
    The attention lets the 16 coarse spectral positions exchange information.
    """
    
    def __init__(self, in_channels=20, out_channels=5, 
                 encoder_channels=[64, 128, 256, 256, 512]):
        super().__init__()
        
        # Encoder (same as MixNet V1)
        self.enc1 = DownBlock(in_channels, encoder_channels[0])
        self.enc2 = DownBlock(encoder_channels[0], encoder_channels[1])
        self.enc3 = DownBlock(encoder_channels[1], encoder_channels[2])
        self.enc4 = DownBlock(encoder_channels[2], encoder_channels[3])
        self.enc5 = DownBlock(encoder_channels[3], encoder_channels[4])
        
        # Bottleneck with ATTENTION (NEW!)
        self.bottleneck_conv = ConvBlock(encoder_channels[4], encoder_channels[4])
        self.bottleneck_attn = BottleneckAttention(
            d_model=encoder_channels[4], nhead=8, num_layers=2
        )
        
        # Decoder (same as MixNet V1)
        self.dec5 = UpBlock(encoder_channels[4], encoder_channels[4], encoder_channels[3])
        self.dec4 = UpBlock(encoder_channels[3], encoder_channels[3], encoder_channels[2])
        self.dec3 = UpBlock(encoder_channels[2], encoder_channels[2], encoder_channels[1])
        self.dec2 = UpBlock(encoder_channels[1], encoder_channels[1], encoder_channels[0])
        self.dec1 = UpBlock(encoder_channels[0], encoder_channels[0], encoder_channels[0])
        
        self.final_conv = nn.Conv1d(encoder_channels[0], out_channels, kernel_size=1)
        self.activation = nn.Softplus()
    
    def forward(self, x):
        # Encoder
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        s5, x = self.enc5(x)
        
        # Bottleneck + Attention
        x = self.bottleneck_conv(x)
        x = self.bottleneck_attn(x)  # <-- NEW: cross-position attention
        
        # Decoder
        x = self.dec5(x, s5)
        x = self.dec4(x, s4)
        x = self.dec3(x, s3)
        x = self.dec2(x, s2)
        x = self.dec1(x, s1)
        
        x = self.final_conv(x)
        x = self.activation(x)
        return x


if __name__ == "__main__":
    model = MixNetV2(in_channels=20, out_channels=5)
    params = sum(p.numel() for p in model.parameters())
    print(f"MixNet V2 parameters: {params:,}")
    
    x = torch.randn(1, 20, 16384)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
    
    # Compare with V1
    import sys
    sys.path.insert(0, '.')
    from unet1d import MixNet
    v1 = MixNet(in_channels=20, out_channels=5)
    v1_params = sum(p.numel() for p in v1.parameters())
    print(f"\nV1 params: {v1_params:,}")
    print(f"V2 params: {params:,}")
    print(f"Extra:     {params - v1_params:,} ({(params-v1_params)/v1_params*100:.1f}% more)")
