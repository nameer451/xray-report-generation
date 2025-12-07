import torch
import torch.nn as nn
from transformers import BartForConditionalGeneration, AutoTokenizer
from transformers.modeling_outputs import BaseModelOutput

class ImpressionDecoder(nn.Module):
    def __init__(self, device="cuda"):
        super().__init__()
        self.device = device
        
        # Load BioBART model
        self.model = BartForConditionalGeneration.from_pretrained(
            "GanjinZero/biobart-base"
        ).to(device)
        
        # ✅ OPTIONAL: Explicitly disable cache for training
        self.model.config.use_cache = False  # ← ADD THIS LINE
        
        # Visual projection: 512 → 768
        self.visual_proj = nn.Linear(512, 768).to(device)
        
        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")
        
        print("✓ ImpressionDecoder initialized")
        print(f"  - Visual projection: 512 → 768")
        print(f"  - use_cache: {self.model.config.use_cache}")  # Should print False
    
    def forward(self, image_proj, input_ids, attention_mask):
        """
        image_proj: (B, 512) - output from frozen CheXzero projection
        input_ids: (B, seq_len) - tokenized impression text
        attention_mask: (B, seq_len)
        """
        # Project visual features to BART dimension
        encoder_hidden_states = self.visual_proj(image_proj)  # (B, 768)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)  # (B, 1, 768)
        
        # Wrap encoder output in proper format
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # OPTION 1: Simple approach - let BART handle token shifting internally
        # Just pass labels, BART will shift them automatically
        outputs = self.model(
            encoder_outputs=encoder_outputs,
            decoder_attention_mask=attention_mask,
            labels=input_ids,  # BART shifts these internally
        )
        
        return outputs.loss
    
    @torch.no_grad()
    def generate(self, image_proj, max_length=64, num_beams=4):
        """
        Generate impression text from image features
        """
        # Project visual features
        encoder_hidden_states = self.visual_proj(image_proj)
        encoder_hidden_states = encoder_hidden_states.unsqueeze(1)
        
        # Wrap in proper format
        encoder_outputs = BaseModelOutput(
            last_hidden_state=encoder_hidden_states
        )
        
        # Generate
        generated_ids = self.model.generate(
            encoder_outputs=encoder_outputs,
            max_length=max_length,
            num_beams=num_beams,
            early_stopping=True
        )
        
        # Decode to text
        return [self.tokenizer.decode(g, skip_special_tokens=True)
                for g in generated_ids]

