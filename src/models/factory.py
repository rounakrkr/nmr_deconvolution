import torch.nn as nn

from src.models.unet1d import MixNet
from src.models.unet1d_v2 import MixNetV2
from src.models.unet1d_v3 import MixNetV3

MODEL_NAMES = ("v1", "v2", "v3")


def build_model(name: str, num_mixtures: int = 20, num_compounds: int = 5) -> nn.Module:
    if name == "v1":
        return MixNet(in_channels=num_mixtures, out_channels=num_compounds)
    if name == "v2":
        return MixNetV2(in_channels=num_mixtures, out_channels=num_compounds)
    if name == "v3":
        return MixNetV3(in_channels=num_mixtures, out_channels=num_compounds)
    raise ValueError(f"unknown model '{name}', expected one of {MODEL_NAMES}")
