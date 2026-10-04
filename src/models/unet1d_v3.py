"""
MixNet V3 — True Cross-Mixture Architecture.

Sir's idea PROPERLY implemented:
  1. Each mixture SEPARATELY encoded (shared weights)
  2. Attention ACROSS mixtures to find co-varying peaks
  3. Decode to compound spectra
"""
import torch
import torch.nn as nn
from src.models.blocks import ConvBlock, DownBlock, UpBlock


class SharedEncoder(nn.Module):
    """Encodes ONE mixture spectrum. Same weights used for all 20."""
    def __init__(self, encoder_channels=[64, 128, 256, 256, 512]):
        super().__init__()
        self.enc1 = DownBlock(1, encoder_channels[0])
        self.enc2 = DownBlock(encoder_channels[0], encoder_channels[1])
        self.enc3 = DownBlock(encoder_channels[1], encoder_channels[2])
        self.enc4 = DownBlock(encoder_channels[2], encoder_channels[3])
        self.enc5 = DownBlock(encoder_channels[3], encoder_channels[4])
        self.bottleneck = ConvBlock(encoder_channels[4], encoder_channels[4])
    
    def forward(self, x):
        # x: (B, 1, 16384) — single mixture
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        s5, x = self.enc5(x)
        x = self.bottleneck(x)
        return x, [s1, s2, s3, s4, s5]


class CrossMixtureAttention(nn.Module):
    """
    The KEY innovation: look across all 20 mixtures.
    
    At each spectral position, compare intensities across mixtures.
    Positions that co-vary = same compound!
    """
    def __init__(self, d_model=512, nhead=8, num_layers=2, num_mixtures=20):
        super().__init__()
        # Attention across mixtures (20 tokens, each d_model dim)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead,
            dim_feedforward=1024, dropout=0.1,
            batch_first=True
        )
        self.mix_attn = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        self.norm = nn.LayerNorm(d_model)
        
    def forward(self, encoded_mixtures):
        # encoded_mixtures: (B, M=20, C=512, L=16)
        B, M, C, L = encoded_mixtures.shape
        
        # For each spatial position, attend across mixtures
        # Reshape: (B*L, M, C) — each position gets its own attention
        x = encoded_mixtures.permute(0, 3, 1, 2)  # (B, L, M, C)
        x = x.reshape(B * L, M, C)                  # (B*L, M, C)
        
        x = self.mix_attn(x)   # Attention across M=20 mixtures!
        x = self.norm(x)
        
        x = x.reshape(B, L, M, C)
        x = x.permute(0, 2, 3, 1)  # (B, M, C, L)
        return x


class CompoundDecoder(nn.Module):
    """Decodes aggregated features to compound spectra."""
    def __init__(self, encoder_channels=[64, 128, 256, 256, 512], out_channels=5):
        super().__init__()
        self.dec5 = UpBlock(encoder_channels[4], encoder_channels[4], encoder_channels[3])
        self.dec4 = UpBlock(encoder_channels[3], encoder_channels[3], encoder_channels[2])
        self.dec3 = UpBlock(encoder_channels[2], encoder_channels[2], encoder_channels[1])
        self.dec2 = UpBlock(encoder_channels[1], encoder_channels[1], encoder_channels[0])
        self.dec1 = UpBlock(encoder_channels[0], encoder_channels[0], encoder_channels[0])
        self.final = nn.Conv1d(encoder_channels[0], out_channels, kernel_size=1)
        self.activation = nn.Softplus()
    
    def forward(self, x, skips):
        # x: (B, C, L), skips: list of 5 skip connections
        x = self.dec5(x, skips[4])
        x = self.dec4(x, skips[3])
        x = self.dec3(x, skips[2])
        x = self.dec2(x, skips[1])
        x = self.dec1(x, skips[0])
        x = self.final(x)
        return self.activation(x)


class MixNetV3(nn.Module):
    """
    True Cross-Mixture Blind Source Separation.
    
    1. Encode each mixture INDEPENDENTLY (shared encoder)
    2. Cross-mixture attention: find co-varying patterns
    3. Aggregate and decode to compound spectra
    """
    def __init__(self, in_channels=20, out_channels=5, 
                 encoder_channels=[64, 128, 256, 256, 512]):
        super().__init__()
        self.num_mixtures = in_channels
        self.num_compounds = out_channels
        
        # Shared encoder — same weights for all mixtures
        self.encoder = SharedEncoder(encoder_channels)
        
        # Cross-mixture attention — finds co-varying patterns
        self.cross_attn = CrossMixtureAttention(
            d_model=encoder_channels[4], nhead=8, num_layers=2,
            num_mixtures=in_channels
        )
        
        # Aggregation: M=20 mixture features → meaningful representation
        # Pool across mixtures to get compound-relevant features
        self.aggregate = nn.Sequential(
            nn.Conv1d(encoder_channels[4] * in_channels, encoder_channels[4], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(encoder_channels[4], encoder_channels[4], kernel_size=1),
            nn.ReLU()
        )
        
        # Skip aggregation — average skips across mixtures
        # Decoder
        self.decoder = CompoundDecoder(encoder_channels, out_channels)
    
    def forward(self, x):
        # x: (B, M=20, L=16384)
        B, M, L = x.shape
        
        # 1. Encode each mixture independently
        bottlenecks = []
        all_skips = [[] for _ in range(5)]
        
        for i in range(M):
            mix_i = x[:, i:i+1, :]  # (B, 1, L)
            bn, skips = self.encoder(mix_i)
            bottlenecks.append(bn)  # (B, 512, 16)
            for j in range(5):
                all_skips[j].append(skips[j])
        
        # Stack bottlenecks: (B, M, 512, 16)
        encoded = torch.stack(bottlenecks, dim=1)
        
        # 2. Cross-mixture attention
        attended = self.cross_attn(encoded)  # (B, M, 512, 16)
        
        # 3. Aggregate across mixtures
        # Reshape: (B, M*512, 16)
        agg_input = attended.reshape(B, M * attended.shape[2], attended.shape[3])
        agg = self.aggregate(agg_input)  # (B, 512, 16)
        
        # Average skip connections across mixtures
        avg_skips = []
        for j in range(5):
            avg_skip = torch.stack(all_skips[j], dim=1).mean(dim=1)  # (B, C, L)
            avg_skips.append(avg_skip)
        
        # 4. Decode to compound spectra
        out = self.decoder(agg, avg_skips)  # (B, 5, 16384)
        return out


if __name__ == "__main__":
    model = MixNetV3(in_channels=20, out_channels=5)
    params = sum(p.numel() for p in model.parameters())
    print(f"MixNet V3 parameters: {params:,}")
    
    x = torch.randn(1, 20, 16384)
    with torch.no_grad():
        y = model(x)
    print(f"Input:  {x.shape}")
    print(f"Output: {y.shape}")
