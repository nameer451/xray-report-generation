import torch
import torch.nn as nn
import torch.nn.functional as F

from local_encoder import LocalImagePatchEncoder, LocalSentenceEncoder
from local_projection import LocalProjectionHead, LocalAlignmentModule
from local_decoder import LocalReportDecoder


class LocalReportModel(nn.Module):
    def __init__(self,
                 clip_checkpoint,
                 projection_dim=256,
                 device="cuda",
                 fine_tune_sentence_encoder=True,
                 reconstruction_loss_weight=0.5):
        super().__init__()
        self.device = device
        self.recon_w = reconstruction_loss_weight

        # encoders
        self.patch_encoder = LocalImagePatchEncoder(clip_checkpoint, device)
        self.sentence_encoder = LocalSentenceEncoder(device=device,
                                                     fine_tune=fine_tune_sentence_encoder)

        # projection + alignment
        self.projection = LocalProjectionHead(768, 768, projection_dim, device)
        self.align = LocalAlignmentModule(device=device)

        # decoder
        self.decoder = LocalReportDecoder(projection_dim, device=device)

        # reconstructor: patch pooled → global embedding
        # reconstructor: patch pooled → global embedding
        self.reconstruct = nn.Sequential(
            nn.Linear(projection_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 768),
        ).to(device)


    # helper for Stage-1 & Stage-2
    def encode_sentences_batch(self, sentences_list):
        all_s = []
        counts = []
        for lst in sentences_list:
            all_s.extend(lst)
            counts.append(len(lst))

        flat = self.sentence_encoder(all_s)
        B = len(sentences_list)
        max_s = max(counts)
        out = torch.zeros(B, max_s, flat.shape[-1], device=self.device)

        idx = 0
        for b in range(B):
            c = counts[b]
            out[b, :c] = flat[idx:idx + c]
            idx += c

        return out

    # ===== STAGE 1 =====
    def forward_alignment(self, images, sentences_list):
        patches, _ = self.patch_encoder(images)
        sent = self.encode_sentences_batch(sentences_list)

        p_proj, s_proj = self.projection(patches, sent)
        loss = self.align(p_proj, s_proj)
        return loss

    # ===== STAGE 2 =====
    def forward_generation(self, images, target_ids, target_mask, sentences_list):
        with torch.no_grad():
            patches, global_emb = self.patch_encoder(images)

        sent = self.encode_sentences_batch(sentences_list)
        patch_proj, _ = self.projection(patches, sent)

        gen_loss = self.decoder(patch_proj, target_ids, target_mask)

        pooled = patch_proj.mean(dim=1)
        recon = self.reconstruct(pooled)

        # CheXzero global is 512-dim, recon outputs 768-dim
        # Just compare first 512 dimensions
        recon_loss = 1 - F.cosine_similarity(recon[:, :512], global_emb, dim=-1).mean()

        total = gen_loss + self.recon_w * recon_loss
        return total, gen_loss.item(), recon_loss.item()

    @torch.no_grad()
    def generate(self, images, sentences_list):
        patches, _ = self.patch_encoder(images)
        sent = self.encode_sentences_batch(sentences_list)

        patch_proj, _ = self.projection(patches, sent)
        return self.decoder.generate(patch_proj)
