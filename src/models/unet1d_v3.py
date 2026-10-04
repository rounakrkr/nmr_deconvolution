"""
MixNet V3 — shared per-mixture encoder, attention across mixtures, order-invariant aggregation.

The output is invariant to the order of the input mixtures (in eval mode):
the encoder is shared, the transformer has no positional encoding, and the
mixture axis is reduced with mean and max pooling.
"""
import torch
import torch.nn as nn

from src.models.blocks import ConvBlock, DownBlock, UpBlock


class SharedEncoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 256, 512)):
        super().__init__()
        ch = list(encoder_channels)
        self.enc1 = DownBlock(1, ch[0])
        self.enc2 = DownBlock(ch[0], ch[1])
        self.enc3 = DownBlock(ch[1], ch[2])
        self.enc4 = DownBlock(ch[2], ch[3])
        self.enc5 = DownBlock(ch[3], ch[4])
        self.bottleneck = ConvBlock(ch[4], ch[4])

    def forward(self, x):
        s1, x = self.enc1(x)
        s2, x = self.enc2(x)
        s3, x = self.enc3(x)
        s4, x = self.enc4(x)
        s5, x = self.enc5(x)
        return self.bottleneck(x), [s1, s2, s3, s4, s5]


class CrossMixtureAttention(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_layers=2):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=1024, dropout=0.1, batch_first=True
        )
        self.mix_attn = nn.TransformerEncoder(layer, num_layers=num_layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, encoded):
        b, m, c, l = encoded.shape
        x = encoded.permute(0, 3, 1, 2).reshape(b * l, m, c)
        x = self.norm(self.mix_attn(x))
        return x.reshape(b, l, m, c).permute(0, 2, 3, 1)


class CompoundDecoder(nn.Module):
    def __init__(self, encoder_channels=(64, 128, 256, 256, 512), out_channels=5):
        super().__init__()
        ch = list(encoder_channels)
        self.dec5 = UpBlock(ch[4], ch[4], ch[3])
        self.dec4 = UpBlock(ch[3], ch[3], ch[2])
        self.dec3 = UpBlock(ch[2], ch[2], ch[1])
        self.dec2 = UpBlock(ch[1], ch[1], ch[0])
        self.dec1 = UpBlock(ch[0], ch[0], ch[0])
        self.final = nn.Conv1d(ch[0], out_channels, kernel_size=1)
        self.activation = nn.Softplus()

    def forward(self, x, skips):
        x = self.dec5(x, skips[4])
        x = self.dec4(x, skips[3])
        x = self.dec3(x, skips[2])
        x = self.dec2(x, skips[1])
        x = self.dec1(x, skips[0])
        return self.activation(self.final(x))


class MixNetV3(nn.Module):
    def __init__(self, in_channels=20, out_channels=5, encoder_channels=(64, 128, 256, 256, 512)):
        super().__init__()
        ch = list(encoder_channels)
        self.num_mixtures = in_channels
        self.num_compounds = out_channels
        self.encoder = SharedEncoder(ch)
        self.cross_attn = CrossMixtureAttention(d_model=ch[4], nhead=8, num_layers=2)
        self.aggregate = nn.Sequential(
            nn.Conv1d(2 * ch[4], ch[4], kernel_size=1),
            nn.ReLU(),
            nn.Conv1d(ch[4], ch[4], kernel_size=1),
            nn.ReLU(),
        )
        self.decoder = CompoundDecoder(ch, out_channels)

    def forward(self, x):
        b, m, l = x.shape
        bottleneck, skips = self.encoder(x.reshape(b * m, 1, l))
        c, lb = bottleneck.shape[1], bottleneck.shape[2]
        attended = self.cross_attn(bottleneck.reshape(b, m, c, lb))
        agg = self.aggregate(torch.cat([attended.mean(dim=1), attended.amax(dim=1)], dim=1))
        avg_skips = [s.reshape(b, m, *s.shape[1:]).mean(dim=1) for s in skips]
        return self.decoder(agg, avg_skips)
