import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, BartTokenizer
from transformers.modeling_outputs import BaseModelOutput


class LocalReportDecoder(nn.Module):
    """
    Decodes BART text from adapted local patch embeddings.
    """
    def __init__(self,
                 projection_dim=256,
                 decoder_model="GanjinZero/biobart-base",
                 device="cuda",
                 max_length=512,
                 attention_loss_weight=0.1):
        super().__init__()
        self.device = device
        self.max_len = max_length
        self.attn_w = attention_loss_weight

        print(f"✓ Loading decoder: {decoder_model}")
        self.tokenizer = BartTokenizer.from_pretrained(decoder_model)
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_model).to(device)
        self.decoder.config.use_cache = False

        bart_dim = self.decoder.config.d_model

        # NEW: patch → BART encoder adapter
        self.adapter = nn.Sequential(
            nn.Linear(projection_dim, bart_dim),
            nn.ReLU(),
            nn.LayerNorm(bart_dim),
            nn.Linear(bart_dim, bart_dim),
        ).to(device)

    def compute_attention_loss(self, cross_attn):
        if cross_attn is None:
            return torch.tensor(0.0, device=self.device)

        losses = []
        for att in cross_attn:
            a = att.mean(dim=(0, 1)) + 1e-8
            entropy = -(a * torch.log(a)).sum(dim=-1).mean()
            losses.append(entropy)

        return torch.stack(losses).mean()

    def forward(self, adapted_patches, target_ids=None, target_mask=None):
        B, P, D = adapted_patches.shape

        enc_hidden = self.adapter(adapted_patches)

        enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
        enc_mask = torch.ones(B, P, dtype=torch.long, device=self.device)

        if target_ids is not None:
            out = self.decoder(
                encoder_outputs=enc_out,
                attention_mask=enc_mask,
                labels=target_ids,
                decoder_attention_mask=target_mask,
                output_attentions=False  # ← Changed to False to avoid SDPA issue
            )

            gen_loss = out.loss
            # Skip attention loss entirely
            return gen_loss

        return enc_out, enc_mask

    @torch.no_grad()
    def generate(self, adapted_patches, num_beams=3, use_constraints=True):
        B, P, D = adapted_patches.shape
        enc_hidden = self.adapter(adapted_patches)

        enc_out = BaseModelOutput(last_hidden_state=enc_hidden)
        enc_mask = torch.ones(B, P, dtype=torch.long, device=self.device)

        self.decoder.config.use_cache = True

        kwargs = dict(
            encoder_outputs=enc_out,
            attention_mask=enc_mask,
            max_length=self.max_len,
            num_beams=num_beams,
            no_repeat_ngram_size=4,
            early_stopping=True,
        )

        if use_constraints:
            kwargs.update(dict(
                repetition_penalty=1.2,
                top_p=0.9,
                temperature=1.0,
            ))

        ids = self.decoder.generate(**kwargs)
        self.decoder.config.use_cache = False

        return [self.tokenizer.decode(x, skip_special_tokens=True) for x in ids]
