def run_zero_shot_classification(h5_path, clip_checkpoint, pathologies, device):
    """Run zero-shot pathology classification"""
    print("\n" + "="*70)
    print(" STEP 2: ZERO-SHOT PATHOLOGY CLASSIFICATION")
    print("="*70 + "\n")
    
    # Use the pre-imported standard CLIP
    clip = standard_clip
    
    # Load CLIP model
    print("✓ Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # Load custom checkpoint
    checkpoint = torch.load(clip_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load images
    with h5py.File(h5_path, 'r') as f:
        images = f['cxr'][:]
        num_images = images.shape[0]
    
    print(f"✓ Loaded {num_images} images\n")
    
    # Prepare text prompts (contrastive pairs)
    cxr_pair_template = (
        "Findings consistent with {}",
        "No evidence of {}"
    )
    
    positive_prompts = [cxr_pair_template[0].format(p) for p in pathologies]
    negative_prompts = [cxr_pair_template[1].format(p) for p in pathologies]
    
    all_prompts = positive_prompts + negative_prompts
    text_tokens = clip.tokenize(all_prompts).to(device)
    
    print("✓ Encoding text prompts...")
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print("✓ Text prompts encoded\n")
    
    # Process images
    predictions = []
    
    print("Processing images...")
    for i in tqdm(range(num_images)):
        img = images[i]  # uint8 (0-255)
        
        # Convert to PIL Image (grayscale → RGB)
        pil_img = Image.fromarray(img).convert('RGB')
        
        # Use CLIP's preprocessing
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with all text prompts
            logits = (image_features @ text_features.T) * model.logit_scale.exp()
        
        # Calculate probability for each pathology
        pathology_probs = []
        for j in range(len(pathologies)):"""
Unified X-ray Report Generation Pipeline
Combines: Zero-shot pathology classification + Global impressions + Local findings + Grad-CAM
"""

import os
import torch
import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2

# CRITICAL FIX: Import standard CLIP before any other code runs
# This prevents conflicts with the local clip.py file
import sys
_original_sys_path = sys.path.copy()
sys.path = [p for p in sys.path if 'CheXzero' not in p and p != '' and p != '.']
import clip as standard_clip
sys.path = _original_sys_path

# Note: Don't add sys.path modifications here
# Import models directly when needed to avoid conflicts

# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    # Paths
    DATA_FOLDER = "./data/xray_images"  # Folder containing input images
    H5_OUTPUT = "./data/images.h5"
    RESULTS_DIR = "./results"
    
    # Model checkpoints
    CLIP_CHECKPOINT = "/checkpoints/best_64_0.0001_original_35000_0.864.pt"
    GLOBAL_DECODER_CHECKPOINT = "/checkpoints/global_decoder_epoch5.pt"
    LOCAL_MODEL_CHECKPOINT = "/checkpoints//stage2_final.pt"
    
    # Zero-shot pathologies
    PATHOLOGIES = ["Pneumonia", "Cardiomegaly", "Edema", "Pleural Effusion"]
    
    # Processing settings
    BATCH_SIZE = 8
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Grad-CAM settings
    NUM_GRADCAM_SAMPLES = 5  # Generate Grad-CAM for first N images


# ============================================================================
# STEP 1: PREPROCESS IMAGES TO H5
# ============================================================================

def preprocess_image(img, desired_size=320):
    """
    Preprocess image: resize maintaining aspect ratio, then pad to square.
    Matches the exact logic from your training preprocessing.
    """
    old_size = img.size  # (width, height)
    ratio = float(desired_size) / max(old_size)
    new_size = (int(old_size[0] * ratio), int(old_size[1] * ratio))
    
    # Resize using LANCZOS
    img = img.resize(new_size, Image.Resampling.LANCZOS)
    
    # Pad to square
    new_img = Image.new("L", (desired_size, desired_size))
    new_img.paste(
        img,
        ((desired_size - new_size[0]) // 2,
         (desired_size - new_size[1]) // 2)
    )
    return new_img


def preprocess_images_to_h5(image_folder, output_h5, target_size=320):
    """
    Convert all images in a folder to H5 format.
    Uses the exact same preprocessing logic as your training pipeline.
    Supports: .jpg, .jpeg, .png, .dcm
    """
    print("\n" + "="*70)
    print(" STEP 1: PREPROCESSING IMAGES")
    print("="*70 + "\n")
    
    image_folder = Path(image_folder)
    
    # Find all images
    image_extensions = ['.jpg', '.jpeg', '.png', '.dcm']
    image_files = []
    for ext in image_extensions:
        image_files.extend(list(image_folder.glob(f'*{ext}')))
        image_files.extend(list(image_folder.glob(f'*{ext.upper()}')))
    
    image_files = sorted(image_files)
    
    if len(image_files) == 0:
        raise ValueError(f"No images found in {image_folder}")
    
    print(f"✓ Found {len(image_files)} images")
    print(f"✓ Target size: {target_size}x{target_size}")
    print(f"✓ Output: {output_h5}\n")
    
    # Process images and save to H5
    dset_size = len(image_files)
    failed_images = []
    filenames = []
    
    print("Processing images...")
    with h5py.File(output_h5, 'w') as h5f:
        img_dset = h5f.create_dataset(
            'cxr',
            shape=(dset_size, target_size, target_size),
            dtype='uint8'
        )
        
        for idx, img_path in enumerate(tqdm(image_files, desc="Processing")):
            try:
                # Handle DICOM files
                if img_path.suffix.lower() == '.dcm':
                    import pydicom
                    dcm = pydicom.dcmread(str(img_path))
                    img_array = dcm.pixel_array
                    # Normalize to 0-255 range
                    img_array = ((img_array - img_array.min()) / 
                                (img_array.max() - img_array.min()) * 255).astype(np.uint8)
                    img_pil = Image.fromarray(img_array).convert('L')
                else:
                    # Open with PIL (handles JPEG/PNG robustly)
                    img_pil = Image.open(img_path).convert('L')
                
                # Preprocess (resize maintaining aspect ratio, then pad)
                img = preprocess_image(img_pil, desired_size=target_size)
                
                # Save to H5 as uint8
                img_dset[idx] = np.array(img)
                filenames.append(img_path.name)
                
            except Exception as e:
                failed_images.append((img_path.name, str(e)))
                print(f"⚠ Warning: Failed to process {img_path.name}: {str(e)}")
                # Fill with zeros for failed images
                img_dset[idx] = np.zeros((target_size, target_size), dtype=np.uint8)
                filenames.append(img_path.name)
    
    print(f"\n✓ Processed {len(image_files)} images")
    print(f"✓ Failed: {len(failed_images)} / {len(image_files)}")
    
    if failed_images:
        print("\nFailed images:")
        for fname, error in failed_images[:5]:  # Show first 5
            print(f"  - {fname}: {error}")
        if len(failed_images) > 5:
            print(f"  ... and {len(failed_images) - 5} more")
    
    print(f"\n✓ Saved to {output_h5}")
    
    # Save filenames mapping
    filenames_csv = output_h5.replace('.h5', '_filenames.csv')
    pd.DataFrame({'filename': filenames}).to_csv(filenames_csv, index=False)
    print(f"✓ Saved filename mapping to {filenames_csv}\n")
    
    return filenames


# ============================================================================
# STEP 2: ZERO-SHOT PATHOLOGY CLASSIFICATION
# ============================================================================

def run_zero_shot_classification(h5_path, clip_checkpoint, pathologies, device):
    """Run zero-shot pathology classification"""
    print("\n" + "="*70)
    print(" STEP 2: ZERO-SHOT PATHOLOGY CLASSIFICATION")
    print("="*70 + "\n")
    
    # CRITICAL: Import CLIP before any other modules to avoid conflicts
    # Remove current directory from sys.path temporarily
    import sys
    original_path = sys.path.copy()
    
    # Remove paths that might contain the custom clip.py
    sys.path = [p for p in sys.path if '/content/CheXzero' not in p and '.' not in p]
    
    # Now import the standard CLIP package
    import clip
    
    # Restore original path
    sys.path = original_path
    
    # Load CLIP model
    print("✓ Loading CLIP model...")
    model, preprocess = clip.load("ViT-B/32", device=device, jit=False)
    
    # Load custom checkpoint
    checkpoint = torch.load(clip_checkpoint, map_location=device)
    model.load_state_dict(checkpoint)
    model.eval()
    
    # Load images
    with h5py.File(h5_path, 'r') as f:
        images = f['cxr'][:]
        num_images = images.shape[0]
    
    print(f"✓ Loaded {num_images} images\n")
    
    # Prepare text prompts (contrastive pairs)
    cxr_pair_template = (
        "Findings consistent with {}",
        "No evidence of {}"
    )
    
    positive_prompts = [cxr_pair_template[0].format(p) for p in pathologies]
    negative_prompts = [cxr_pair_template[1].format(p) for p in pathologies]
    
    all_prompts = positive_prompts + negative_prompts
    text_tokens = clip.tokenize(all_prompts).to(device)
    
    print("✓ Encoding text prompts...")
    with torch.no_grad():
        text_features = model.encode_text(text_tokens)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)
    
    print("✓ Text prompts encoded\n")
    
    # Process images
    predictions = []
    
    print("Processing images...")
    for i in tqdm(range(num_images)):
        img = images[i]  # uint8 (0-255)
        
        # Convert to PIL Image (grayscale → RGB)
        pil_img = Image.fromarray(img).convert('RGB')
        
        # Use CLIP's preprocessing
        img_tensor = preprocess(pil_img).unsqueeze(0).to(device)
        
        # Get image features
        with torch.no_grad():
            image_features = model.encode_image(img_tensor)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            
            # Calculate similarity with all text prompts
            logits = (image_features @ text_features.T) * model.logit_scale.exp()
        
        # Calculate probability for each pathology
        pathology_probs = []
        for j in range(len(pathologies)):
            pos_idx = j
            neg_idx = j + len(pathologies)
            
            # Softmax between positive and negative prompts
            pair_logits = torch.stack([logits[0, pos_idx], logits[0, neg_idx]])
            pair_probs = pair_logits.softmax(dim=0)
            
            # Probability of positive class
            pathology_probs.append(pair_probs[0].item())
        
        predictions.append(pathology_probs)
    
    predictions_array = np.array(predictions)
    
    print(f"\n✓ Classification complete!")
    print(f"✓ Predictions shape: {predictions_array.shape}\n")
    
    return predictions_array


# ============================================================================
# STEP 3: GENERATE GLOBAL IMPRESSIONS
# ============================================================================

def generate_global_impressions(h5_path, clip_checkpoint, decoder_checkpoint, device, batch_size=8):
    """Generate global impressions"""
    print("\n" + "="*70)
    print(" STEP 3: GENERATING GLOBAL IMPRESSIONS")
    print("="*70 + "\n")
    
    # Import from global folder
    import sys
    sys.path.insert(0, './global')
    from global_model import GlobalReportModel
    
    # Load model
    print("✓ Loading global model...")
    model = GlobalReportModel(clip_checkpoint=clip_checkpoint, device=device).to(device)
    decoder_state = torch.load(decoder_checkpoint, map_location=device)
    model.decoder.load_state_dict(decoder_state)
    model.eval()
    
    # Load images
    h5 = h5py.File(h5_path, 'r')
    num_images = len(h5['cxr'])
    
    print(f"✓ Loaded {num_images} images")
    print(f"✓ Using batch size: {batch_size}\n")
    
    # Generate impressions
    impressions = []
    num_batches = (num_images + batch_size - 1) // batch_size
    
    print("Generating impressions...")
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_indices = list(range(start_idx, end_idx))
        
        # Load batch - images are uint8 (0-255)
        batch_imgs = []
        for idx in batch_indices:
            img = h5['cxr'][idx]  # uint8
            img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)
            img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode='bilinear', align_corners=False)
            img = img.squeeze(0).repeat(3, 1, 1)
            img = (img - 101.48761) / 83.43944
            batch_imgs.append(img)
        
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # Generate
        with torch.no_grad():
            generated_batch = model.generate(batch_tensor, max_length=64, num_beams=4)
        
        impressions.extend(generated_batch)
    
    h5.close()
    
    # Clean up sys.path
    sys.path.pop(0)
    
    print(f"\n✓ Generated {len(impressions)} impressions\n")
    
    return impressions


# ============================================================================
# STEP 4: GENERATE LOCAL FINDINGS
# ============================================================================

def generate_local_findings(h5_path, clip_checkpoint, model_checkpoint, device, batch_size=8):
    """Generate local findings"""
    print("\n" + "="*70)
    print(" STEP 4: GENERATING LOCAL FINDINGS")
    print("="*70 + "\n")
    
    # Import from local folder
    import sys
    sys.path.insert(0, './local')
    from local_model import LocalReportModel
    
    # Load model
    print("✓ Loading local model...")
    model = LocalReportModel(
        clip_checkpoint=clip_checkpoint,
        projection_dim=256,
        device=device,
        fine_tune_sentence_encoder=False
    ).to(device)
    
    state_dict = torch.load(model_checkpoint, map_location=device)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    
    # Load images
    h5 = h5py.File(h5_path, 'r')
    num_images = len(h5['cxr'])
    
    print(f"✓ Loaded {num_images} images")
    print(f"✓ Using batch size: {batch_size}\n")
    
    # Generate findings
    findings = []
    num_batches = (num_images + batch_size - 1) // batch_size
    
    print("Generating findings...")
    for batch_idx in tqdm(range(num_batches)):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_size_actual = end_idx - start_idx
        
        # Load batch - images are uint8 (0-255)
        batch_imgs = []
        for idx in range(start_idx, end_idx):
            img = h5['cxr'][idx].astype(np.float32)  # uint8 -> float32
            img = np.expand_dims(img, 0)
            img = np.repeat(img, 3, axis=0)
            img = torch.tensor(img).float()
            img = (img - 101.48761) / 83.43944
            batch_imgs.append(img)
        
        batch_tensor = torch.stack(batch_imgs).to(device)
        
        # Dummy sentences for generation
        dummy_sentences = [[""] for _ in range(batch_size_actual)]
        
        # Generate
        with torch.no_grad():
            generated_batch = model.generate(batch_tensor, dummy_sentences)
        
        findings.extend(generated_batch)
    
    h5.close()
    
    # Clean up sys.path
    sys.path.pop(0)
    
    print(f"\n✓ Generated {len(findings)} findings\n")
    
    return findings


# ============================================================================
# STEP 6: DISPLAY RESULTS
# ============================================================================

def display_results(filenames, predictions, impressions, findings, pathologies, results_dir):
    """Display and save all results"""
    print("\n" + "="*70)
    print(" STEP 6: RESULTS")
    print("="*70 + "\n")
    
    # Create results directory
    Path(results_dir).mkdir(parents=True, exist_ok=True)
    
    # Save to CSV
    results_df = pd.DataFrame({
        'filename': filenames,
        **{pathology: predictions[:, i] for i, pathology in enumerate(pathologies)},
        'global_impression': impressions,
        'local_findings': findings
    })
    
    csv_path = Path(results_dir) / 'results.csv'
    results_df.to_csv(csv_path, index=False)
    print(f"✓ Saved results to {csv_path}\n")
    
    # Display sample results
    print("="*70)
    print(" SAMPLE RESULTS (First 3 Images)")
    print("="*70 + "\n")
    
    for i in range(min(3, len(filenames))):
        print(f"{'='*70}")
        print(f" IMAGE {i+1}: {filenames[i]}")
        print(f"{'='*70}\n")
        
        print("📊 PATHOLOGY CLASSIFICATION:")
        for j, pathology in enumerate(pathologies):
            prob = predictions[i, j]
            bar = '█' * int(prob * 20)
            print(f"  {pathology:20s}: {prob:.2%} {bar}")
        
        print(f"\n💬 GLOBAL IMPRESSION:")
        print(f"  {impressions[i]}")
        
        print(f"\n🔍 LOCAL FINDINGS:")
        print(f"  {findings[i]}")
        
        print()
    
    # Summary statistics
    print("="*70)
    print(" SUMMARY STATISTICS")
    print("="*70 + "\n")
    
    print(f"Total images processed: {len(filenames)}\n")
    
    print("Pathology prevalence (threshold = 0.5):")
    for i, pathology in enumerate(pathologies):
        count = (predictions[:, i] > 0.5).sum()
        pct = count / len(filenames) * 100
        print(f"  {pathology:20s}: {count:3d} ({pct:5.1f}%)")
    
    print(f"\n✅ All results saved to: {results_dir}")
    print(f"✅ Full results CSV: {csv_path}\n")


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def run_pipeline(config):
    """Run the complete inference pipeline"""
    print("\n" + "="*70)
    print(" 🏥 UNIFIED X-RAY REPORT GENERATION PIPELINE ")
    print("="*70)
    print(f"\nDevice: {config.DEVICE}")
    print(f"Input folder: {config.DATA_FOLDER}")
    print(f"Results folder: {config.RESULTS_DIR}\n")
    
    # Step 1: Preprocess images
    filenames = preprocess_images_to_h5(
        config.DATA_FOLDER,
        config.H5_OUTPUT
    )
    
    # Step 2: Zero-shot classification
    predictions = run_zero_shot_classification(
        config.H5_OUTPUT,
        config.CLIP_CHECKPOINT,
        config.PATHOLOGIES,
        config.DEVICE
    )
    
    # Step 3: Global impressions
    impressions = generate_global_impressions(
        config.H5_OUTPUT,
        config.CLIP_CHECKPOINT,
        config.GLOBAL_DECODER_CHECKPOINT,
        config.DEVICE,
        config.BATCH_SIZE
    )
    
    # Step 4: Local findings
    findings = generate_local_findings(
        config.H5_OUTPUT,
        config.CLIP_CHECKPOINT,
        config.LOCAL_MODEL_CHECKPOINT,
        config.DEVICE,
        config.BATCH_SIZE
    )
    
    
    # Step 6: Display results
    display_results(
        filenames,
        predictions,
        impressions,
        findings,
        config.PATHOLOGIES,
        config.RESULTS_DIR
    )
    
    print("\n" + "="*70)
    print(" ✅ PIPELINE COMPLETE!")
    print("="*70 + "\n")


# ============================================================================
# RUN
# ============================================================================

if __name__ == "__main__":
    config = Config()
    
    try:
        run_pipeline(config)
    except KeyboardInterrupt:
        print("\n\n⚠️  Pipeline interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error: {str(e)}")
        import traceback

        traceback.print_exc()

