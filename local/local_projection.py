import torch
import torch.nn as nn
import torch.nn.functional as F


class LocalProjectionHead(nn.Module):
    """
    Projects image patches + sentence embeddings into a shared aligned space.
    """
    def __init__(self, patch_dim=768, sentence_dim=768, projection_dim=256, device="cuda"):
        super().__init__()
        self.device = device
        self.proj_dim = projection_dim

        # VISUAL (PATCH) PROJECTOR — FROZEN IN STAGE-2
        self.patch_proj = nn.Sequential(
            nn.Linear(patch_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        ).to(device)

        # TEXT PROJECTOR — FROZEN IN STAGE-2
        self.sentence_proj = nn.Sequential(
            nn.Linear(sentence_dim, projection_dim),
            nn.ReLU(),
            nn.Linear(projection_dim, projection_dim),
        ).to(device)

    def forward(self, patch_emb, sentence_emb):
        # Fix FP16 → FP32 mismatch from CheXzero
        patch_emb = patch_emb.float()
        sentence_emb = sentence_emb.float()

        p = self.patch_proj(patch_emb)
        s = self.sentence_proj(sentence_emb)

        p = F.normalize(p, dim=-1)
        s = F.normalize(s, dim=-1)

        return p, s


class LocalAlignmentModule(nn.Module):
    """
    Region-level alignment loss for Stage-1.
    """
    def __init__(self, temperature=0.07, device="cuda"):
        super().__init__()
        self.device = device
        self.temp = temperature

    def forward(self, patch_proj, sent_proj):
        B, P, D = patch_proj.shape
        _, S, _ = sent_proj.shape

        patch_dim = int(P ** 0.5)
        grid = patch_proj.view(B, patch_dim, patch_dim, D)  # (B, 7, 7, D)

        pooled = F.avg_pool2d(
            grid.permute(0, 3, 1, 2),
            kernel_size=2,
            stride=2
        )  # (B, D, 3, 3)

        regions = pooled.permute(0, 2, 3, 1).reshape(B, 9, D)

        L = min(S, 9)
        regions = regions[:, :L]
        sent_proj = sent_proj[:, :L]

        sim = torch.bmm(regions, sent_proj.transpose(1, 2)) / self.temp  # (B, L, L)

        labels = torch.arange(L, device=self.device).repeat(B)

        loss_f = F.cross_entropy(sim.reshape(-1, L), labels)
        loss_b = F.cross_entropy(sim.transpose(1, 2).reshape(-1, L), labels)

        return (loss_f + loss_b) / 2




