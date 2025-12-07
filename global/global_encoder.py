import torch
import torch.nn as nn
import sys

sys.path.append("/content/CheXzero")
from model import build_model


class FrozenGlobalImageEncoder(nn.Module):
    """
    Loads CheXzero model EXACTLY as trained, using build_model().
    Automatically matches architecture defined by checkpoint.
    """

    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()
        self.device = device

        print("✓ Loading CheXzero checkpoint:", checkpoint_path)
        state = torch.load(checkpoint_path, map_location=device)

        # Use CheXzero’s architecture auto-builder
        self.model = build_model(state).to(device)
        self.model.eval()

        # Freeze everything
        for p in self.model.parameters():
            p.requires_grad = False

        print("✓ Loaded CheXzero encoder using build_model()")
        print("✓ All encoder params frozen")

    @torch.no_grad()
    def forward(self, images):
        # images: (B, 3, H, W)
        return self.model.encode_image(images.to(self.device))

