import torch
import torch.nn as nn

from global_encoder import FrozenGlobalImageEncoder
from impression_decoder import ImpressionDecoder


class GlobalReportModel(nn.Module):

    def __init__(self, clip_checkpoint, device="cuda"):
        super().__init__()

        self.device = device

        # Frozen CheXzero encoder
        self.encoder = FrozenGlobalImageEncoder(clip_checkpoint, device=device)

        # BioBART decoder
        self.decoder = ImpressionDecoder(device=device)

        print("✓ GlobalReportModel assembled.")

    def forward(self, images, input_ids, attention_mask):
        """
        images: (B, 3, 224, 224)
        """

        # 1) Encode image → fp16
        image_proj = self.encoder(images)

        # 2) Convert to fp32 BEFORE sending into BioBART
        image_proj = image_proj.float()

        # 3) Decoder computes loss
        loss = self.decoder(image_proj, input_ids, attention_mask)

        return loss

    # In global_model.py
    @torch.no_grad()
    def generate(self, images, max_length=64, num_beams=4):
        """
        Generate impression text from images.
        
        Args:
            images: (B, 3, H, W) tensor of chest X-rays
            max_length: Maximum number of tokens to generate
            num_beams: Number of beams for beam search
            
        Returns:
            list: Generated impression texts
        """
        # Encode image through frozen CheXzero
        image_proj = self.encoder(images).float()
        
        # Generate through decoder
        return self.decoder.generate(
            image_proj,
            max_length=max_length,
            num_beams=num_beams
        )

