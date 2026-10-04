import torch
import torch.nn as nn
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.models.blocks import ConvBlock, DownBlock, UpBlock

class MixNet(nn.Module):
    """
    MixNet: 1D U-Net for General-Purpose NMR Spectral Decomposition.
    
    Domain-agnostic blind source separation model.
    Takes M mixture NMR spectra as input and outputs N individual compound
    spectra — without requiring prior knowledge of compound identities.
    
    Applicable to any NMR mixture: biological, botanical, pharmaceutical, etc.
    """
    def __init__(self, in_channels, out_channels, encoder_channels=[64, 128, 256, 256, 512],
                 kernel_size=3, pool_kernel=4, pool_stride=4, use_batch_norm=True, activation='relu'):
        super().__init__()
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        kwargs = {
            'kernel_size': kernel_size,
            'use_bn': use_batch_norm,
            'activation': activation
        }
        
        # Encoder (5 stages)
        self.enc1 = DownBlock(in_channels, encoder_channels[0], pool_kernel, pool_stride, **kwargs)
        self.enc2 = DownBlock(encoder_channels[0], encoder_channels[1], pool_kernel, pool_stride, **kwargs)
        self.enc3 = DownBlock(encoder_channels[1], encoder_channels[2], pool_kernel, pool_stride, **kwargs)
        self.enc4 = DownBlock(encoder_channels[2], encoder_channels[3], pool_kernel, pool_stride, **kwargs)
        self.enc5 = DownBlock(encoder_channels[3], encoder_channels[4], pool_kernel, pool_stride, **kwargs)
        
        # Bottleneck
        bottleneck_ch = encoder_channels[4]
        self.bottleneck = ConvBlock(bottleneck_ch, bottleneck_ch, **kwargs)
        
        # Decoder (5 stages)
        self.dec5 = UpBlock(bottleneck_ch, encoder_channels[4], encoder_channels[3], scale_factor=pool_stride, **kwargs)
        self.dec4 = UpBlock(encoder_channels[3], encoder_channels[3], encoder_channels[2], scale_factor=pool_stride, **kwargs)
        self.dec3 = UpBlock(encoder_channels[2], encoder_channels[2], encoder_channels[1], scale_factor=pool_stride, **kwargs)
        self.dec2 = UpBlock(encoder_channels[1], encoder_channels[1], encoder_channels[0], scale_factor=pool_stride, **kwargs)
        self.dec1 = UpBlock(encoder_channels[0], encoder_channels[0], encoder_channels[0], scale_factor=pool_stride, **kwargs)
        
        # Final output layer
        self.final_conv = nn.Conv1d(encoder_channels[0], out_channels, kernel_size=1)
        self.output_activation = nn.Softplus()
        
    def forward(self, x):
        # Encoder
        skip1, x = self.enc1(x)
        skip2, x = self.enc2(x)
        skip3, x = self.enc3(x)
        skip4, x = self.enc4(x)
        skip5, x = self.enc5(x)
        
        # Bottleneck
        x = self.bottleneck(x)
        
        # Decoder
        x = self.dec5(x, skip5)
        x = self.dec4(x, skip4)
        x = self.dec3(x, skip3)
        x = self.dec2(x, skip2)
        x = self.dec1(x, skip1)
        
        # Final output
        x = self.final_conv(x)
        x = self.output_activation(x)
        
        return x

if __name__ == "__main__":
    # Test block
    M = 10  # Number of mixtures
    N = 5   # Number of compounds
    L = 4096 # Spectral length
    batch_size = 2
    
    model = MixNet(in_channels=M, out_channels=N, encoder_channels=[64, 128, 256, 256, 512])
    
    # Create random tensor
    x = torch.randn(batch_size, M, L)
    
    # Forward pass
    out = model(x)
    
    # Print results
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {out.shape}")
    
    # Parameter count
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
    
    assert out.shape == (batch_size, N, L), f"Expected shape {(batch_size, N, L)}, but got {out.shape}"
    print("Shape test passed!")
