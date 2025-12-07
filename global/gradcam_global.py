"""
Grad-CAM for Global Impression Model
Visualizes which regions of the X-ray the model focuses on when generating impressions.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import h5py
import cv2
from pathlib import Path

from global_model import GlobalReportModel


class GlobalGradCAM:
    """
    Grad-CAM for the global impression model.
    Captures attention on the CheXzero ViT's last transformer layer.
    """
    
    def __init__(self, model, device="cuda"):
        """
        Args:
            model: GlobalReportModel instance
            device: cuda or cpu
        """
        self.model = model
        self.device = device
        self.model.eval()
        
        # Hook into ViT's last layer
        self.gradients = None
        self.activations = None
        
        # Get the visual encoder
        visual = self.model.encoder.model.visual
        
        # Hook the last transformer block
        self.target_layer = visual.transformer.resblocks[-1]
        
        # Register hooks
        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_full_backward_hook(self._save_gradient)
    
    def _save_activation(self, module, input, output):
        """Save forward pass activations"""
        self.activations = output.detach()
    
    def _save_gradient(self, module, grad_input, grad_output):
        """Save backward pass gradients"""
        self.gradients = grad_output[0].detach()
    
    def generate_cam(self, image, target_token_idx=None):
        """
        Generate Grad-CAM heatmap.
        
        Args:
            image: (1, 3, 224, 224) tensor
            target_token_idx: Which output token to use for gradients.
                            If None, uses the entire sequence.
        
        Returns:
            cam: (H, W) numpy array - the heatmap
            generated_text: The generated impression text
        """
        self.model.zero_grad()
        
        # Forward pass
        image_proj = self.model.encoder(image).float()
        
        # Generate text and get logits
        # We need to hook into the decoder to get output logits
        encoder_hidden = self.model.decoder.visual_proj(image_proj).unsqueeze(1)
        
        from transformers.modeling_outputs import BaseModelOutput
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_hidden)
        
        # Generate with return_dict_in_generate to get scores
        outputs = self.model.decoder.model.generate(
            encoder_outputs=encoder_outputs,
            max_length=64,
            num_beams=1,  # Use greedy for Grad-CAM
            return_dict_in_generate=True,
            output_scores=True
        )
        
        generated_ids = outputs.sequences
        scores = outputs.scores  # List of (B, vocab_size) for each step
        
        # Decode text
        generated_text = self.model.decoder.tokenizer.decode(
            generated_ids[0], skip_special_tokens=True
        )
        
        # For simplicity, use the sum of all token logits as target
        # This gives us "what regions matter for the entire impression"
        target_logits = torch.stack([s[0].max() for s in scores]).sum()
        
        # Backward pass
        target_logits.backward()
        
        # Get gradients and activations
        gradients = self.gradients  # (seq_len, B, D)
        activations = self.activations  # (seq_len, B, D)
        
        # Convert from (seq_len, B, D) to (B, seq_len, D)
        gradients = gradients.permute(1, 0, 2)
        activations = activations.permute(1, 0, 2)
        
        # Global average pooling of gradients
        weights = gradients.mean(dim=1, keepdim=True)  # (B, 1, D)
        
        # Weighted combination
        cam = (weights * activations).sum(dim=2)  # (B, seq_len)
        cam = cam[0]  # (seq_len,)
        
        # Remove CLS token, reshape to grid
        cam = cam[1:]  # Remove first token (CLS)
        
        # Reshape to spatial grid
        # For ViT-B/16 with 224x224 input: 14x14 patches
        grid_size = int(np.sqrt(len(cam)))
        cam = cam.reshape(grid_size, grid_size)
        
        # Apply ReLU and normalize
        cam = F.relu(cam)
        cam = cam.cpu().numpy()
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        
        return cam, generated_text
    
    def visualize(self, image, cam, generated_text, save_path=None):
        """
        Visualize Grad-CAM overlay on original image.
        
        Args:
            image: (1, 3, 224, 224) tensor
            cam: (H, W) numpy heatmap
            generated_text: Generated impression
            save_path: Path to save visualization
        """
        # Convert image to numpy
        img_np = image[0].cpu().numpy().transpose(1, 2, 0)
        
        # Denormalize
        mean = 101.48761
        std = 83.43944
        img_np = img_np * std + mean
        img_np = np.clip(img_np, 0, 255).astype(np.uint8)
        
        # Use only one channel (grayscale)
        img_np = img_np[:, :, 0]
        
        # Resize CAM to match image size
        cam_resized = cv2.resize(cam, (224, 224))
        
        # Apply colormap
        heatmap = cv2.applyColorMap(
            (cam_resized * 255).astype(np.uint8), 
            cv2.COLORMAP_JET
        )
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Convert grayscale to RGB
        img_rgb = cv2.cvtColor(img_np, cv2.COLOR_GRAY2RGB)
        
        # Overlay
        overlay = (heatmap * 0.5 + img_rgb * 0.5).astype(np.uint8)
        
        # Plot
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        axes[0].imshow(img_np, cmap='gray')
        axes[0].set_title('Original X-ray')
        axes[0].axis('off')
        
        axes[1].imshow(cam_resized, cmap='jet')
        axes[1].set_title('Grad-CAM Heatmap')
        axes[1].axis('off')
        
        axes[2].imshow(overlay)
        axes[2].set_title('Overlay')
        axes[2].axis('off')
        
        plt.suptitle(f'Generated: "{generated_text}"', fontsize=12, y=0.95)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"✓ Saved to {save_path}")
        
        plt.show()


def load_image_for_gradcam(h5_path, idx):
    """Load and preprocess image for Grad-CAM"""
    h5 = h5py.File(h5_path, 'r')
    img = h5['cxr'][idx]  # (320, 320)
    
    # Convert to tensor
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
    
    # Resize to 224x224
    img = F.interpolate(
        img.unsqueeze(0), 
        size=(224, 224), 
        mode='bilinear', 
        align_corners=False
    ).squeeze(0)
    
    # Repeat to 3 channels
    img = img.repeat(3, 1, 1)
    
    # Normalize
    img = (img - 101.48761) / 83.43944
    
    return img.unsqueeze(0)  # (1, 3, 224, 224)


def run_global_gradcam(
    h5_path,
    index,
    clip_checkpoint,
    decoder_checkpoint,
    save_dir="gradcam_results/global",
    device="cuda"
):
    """
    Run Grad-CAM on a single image with global model.
    
    Args:
        h5_path: Path to H5 file
        index: Image index
        clip_checkpoint: CheXzero weights
        decoder_checkpoint: Trained global decoder
        save_dir: Where to save results
        device: cuda or cpu
    """
    print("\n" + "="*60)
    print(" GRAD-CAM: GLOBAL IMPRESSION MODEL")
    print("="*60 + "\n")
    
    # Create save directory
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    
    # Load model
    print("✓ Loading model...")
    model = GlobalReportModel(clip_checkpoint, device=device).to(device)
    model.decoder.load_state_dict(
        torch.load(decoder_checkpoint, map_location=device)
    )
    model.eval()
    
    # Load image
    print(f"✓ Loading image {index}...")
    image = load_image_for_gradcam(h5_path, index).to(device)
    
    # Initialize Grad-CAM
    print("✓ Computing Grad-CAM...")
    gradcam = GlobalGradCAM(model, device)
    
    # Generate CAM
    cam, generated_text = gradcam.generate_cam(image)
    
    print(f"\n📝 Generated Impression:")
    print(f"   {generated_text}\n")
    
    # Visualize
    save_path = Path(save_dir) / f"global_gradcam_{index}.png"
    gradcam.visualize(image, cam, generated_text, save_path)
    
    print("✓ Grad-CAM complete!\n")
    
    return cam, generated_text


# Example usage
if __name__ == "__main__":
    H5_PATH = "/content/drive/MyDrive/mimic_unzipped/h5/mimic_train_p17.h5"
    CLIP_CHECKPOINT = "/content/drive/MyDrive/best_64_0.0001_original_35000_0.864.pt"
    DECODER_CHECKPOINT = "/content/drive/MyDrive/checkpoints_global/global_decoder_best.pt"
    
    # Run on multiple images
    for idx in [0, 10, 20, 30, 40]:
        cam, text = run_global_gradcam(
            H5_PATH,
            idx,
            CLIP_CHECKPOINT,
            DECODER_CHECKPOINT,
            device="cuda"
        )