import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    """
    Standard convolutional block for the 1D U-Net.
    Consists of two Conv1d layers, each optionally followed by BatchNorm and an activation function.
    """
    def __init__(self, in_ch, out_ch, kernel_size=3, use_bn=True, activation='relu'):
        super().__init__()
        padding = kernel_size // 2
        
        layers = []
        
        # First Conv1d
        layers.append(nn.Conv1d(in_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
        
        # Second Conv1d
        layers.append(nn.Conv1d(out_ch, out_ch, kernel_size=kernel_size, padding=padding, bias=not use_bn))
        if use_bn:
            layers.append(nn.BatchNorm1d(out_ch))
        if activation == 'relu':
            layers.append(nn.ReLU(inplace=True))
            
        self.block = nn.Sequential(*layers)
        
    def forward(self, x):
        return self.block(x)


class DownBlock(nn.Module):
    """
    Downsampling block for the encoder part of the U-Net.
    Applies a ConvBlock followed by MaxPool1d.
    Returns both the ConvBlock output (for the skip connection) and the pooled output.
    """
    def __init__(self, in_ch, out_ch, pool_kernel=4, pool_stride=4, **kwargs):
        super().__init__()
        self.conv = ConvBlock(in_ch, out_ch, **kwargs)
        self.pool = nn.MaxPool1d(kernel_size=pool_kernel, stride=pool_stride)
        
    def forward(self, x):
        conv_out = self.conv(x)
        pool_out = self.pool(conv_out)
        return conv_out, pool_out


class UpBlock(nn.Module):
    """
    Upsampling block for the decoder part of the U-Net.
    Applies ConvTranspose1d, concatenates with the skip connection from the encoder,
    and then applies a ConvBlock.
    """
    def __init__(self, in_ch, skip_ch, out_ch, scale_factor=4, **kwargs):
        super().__init__()
        
        # Upsampling layer
        self.up = nn.ConvTranspose1d(in_ch, in_ch, kernel_size=scale_factor, stride=scale_factor)
        
        # ConvBlock after concatenation (in_ch + skip_ch)
        self.conv = ConvBlock(in_ch + skip_ch, out_ch, **kwargs)
        
    def forward(self, x, skip_x):
        # Upsample the input
        up_x = self.up(x)
        
        # Concatenate with skip connection along the channel dimension (dim=1)
        # Note: If dimensions don't match exactly due to rounding in pooling/upsampling,
        # padding or interpolation might be needed. Assuming exact matching here.
        cat_x = torch.cat([up_x, skip_x], dim=1)
        
        # Process combined features
        return self.conv(cat_x)
