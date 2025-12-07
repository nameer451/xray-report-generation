import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer
import sys

# CheXzero repo root
sys.path.append("/content/CheXzero")
from model import build_model


class LocalImagePatchEncoder(nn.Module):
    """
    Extract patch-level embeddings using frozen CheXzero ViT.
    Produces:
        patch_embeddings: (B, P, 768)
        global_embedding: (B, 768)
    """
    def __init__(self, checkpoint_path, device="cuda"):
        super().__init__()
        self.device = device

        print("✓ Loading CheXzero for patch encoding")
        state = torch.load(checkpoint_path, map_location=device)
        self.model = build_model(state).to(device)
        self.model.eval()

        self.visual = self.model.visual

        for p in self.model.parameters():
            p.requires_grad = False

        self.patch_size = self.visual.conv1.kernel_size[0]
        num_positions = self.visual.positional_embedding.shape[0]
        self.num_patches = num_positions - 1
        patches_per_dim = int(self.num_patches ** 0.5)
        self.expected_image_size = patches_per_dim * self.patch_size

        print(f"✓ Expected image size: {self.expected_image_size}×{self.expected_image_size}")
        print(f"✓ Produces {self.num_patches} patches per image")

    @torch.no_grad()
    def forward(self, images):
        B, C, H, W = images.shape

        if H != self.expected_image_size or W != self.expected_image_size:
            images = F.interpolate(
                images,
                size=(self.expected_image_size, self.expected_image_size),
                mode="bilinear",
                align_corners=False,
            )

        x = self.visual.conv1(images.type(self.visual.conv1.weight.dtype))
        x = x.reshape(B, x.shape[1], -1).permute(0, 2, 1)
        # CheXzero stores class_embedding as (768,), not (1,768)
        cls = self.visual.class_embedding.to(x.dtype).unsqueeze(0).unsqueeze(0)  # (1,1,768)
        cls = cls.expand(B, 1, -1)  # (B,1,768)

        x = torch.cat([cls, x], dim=1)


        x = x + self.visual.positional_embedding.to(x.dtype)
        x = self.visual.ln_pre(x)

        x = x.permute(1, 0, 2)
        x = self.visual.transformer(x)
        x = x.permute(1, 0, 2)

        global_emb = self.visual.ln_post(x[:, 0])
        patches = x[:, 1:]

        if self.visual.proj is not None:
            global_emb = global_emb @ self.visual.proj

        return patches, global_emb


class LocalSentenceEncoder(nn.Module):
    """BioClinicalBERT sentence encoder"""
    def __init__(self,
                 model_name="emilyalsentzer/Bio_ClinicalBERT",
                 device="cuda",
                 fine_tune=True):
        super().__init__()
        self.device = device
        self.fine_tune = fine_tune

        print(f"✓ Loading sentence encoder: {model_name}")
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.encoder = AutoModel.from_pretrained(model_name).to(device)
        self.hidden = self.encoder.config.hidden_size

        if not fine_tune:
            self.encoder.eval()
            for p in self.encoder.parameters():
                p.requires_grad = False
            print("Sentence encoder frozen")

    def forward(self, sentences):
        encoded = self.tokenizer(
            sentences,
            padding=True,
            truncation=True,
            max_length=256,
            return_tensors="pt",
        ).to(self.device)

        outputs = self.encoder(**encoded)
        return outputs.last_hidden_state[:, 0]


