import os
import torch
import h5py
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

from local_model import LocalReportModel
from utils.sentence_splitter import SentenceSplitter


# =========================================================
# CONFIG
# =========================================================

H5_PATH = "/content/drive/MyDrive/mimic_unzipped/h5/mimic_train_p17.h5"
CSV_PATH = "/content/drive/MyDrive/mimic_cleaned_csvs/mimic_train_p17_findings.csv"
CLIP_CHECKPOINT = "/content/drive/MyDrive/best_64_0.0001_original_35000_0.864.pt"
MODEL_CHECKPOINT = "/content/drive/MyDrive/checkpoints_local/stage2_final.pt"  # or stage2b_best.pt

OUTPUT_CSV = "/content/drive/MyDrive/generated_findings_p17_full.csv"

MAX_SAMPLES = None  # Generate for entire P17 dataset
BATCH_SIZE = 8
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# DATASET FOR INFERENCE
# =========================================================

class InferenceDataset(Dataset):
    """Dataset for generating findings without requiring ground truth sentences"""
    
    def __init__(self, h5_path, max_samples=None):
        self.h5_file = h5py.File(h5_path, 'r')
        self.images = self.h5_file['cxr']
        
        self.length = min(len(self.images), max_samples) if max_samples else len(self.images)
        
        # CheXzero normalization
        self.mean = 101.48761
        self.std = 83.43944
        
        print(f"✓ InferenceDataset loaded: {self.length} samples")
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        # Load image
        img = self.images[idx]  # (320, 320)
        
        # Convert to 3-channel and normalize
        img = np.expand_dims(img, 0)  # (1, 320, 320)
        img = np.repeat(img, 3, axis=0)  # (3, 320, 320)
        img = torch.tensor(img).float()
        img = (img - self.mean) / self.std
        
        return {
            'image': img,
            'idx': idx
        }
    
    def __del__(self):
        if hasattr(self, 'h5_file'):
            self.h5_file.close()


def collate_inference_batch(batch):
    """Collate function for inference"""
    images = torch.stack([item['image'] for item in batch])
    indices = [item['idx'] for item in batch]
    
    return {
        'images': images,
        'indices': indices
    }


# =========================================================
# GENERATION FUNCTION
# =========================================================

def generate_findings():
    print("\n" + "="*60)
    print(" 🏥 GENERATING FINDINGS FOR P17 TEST SET ")
    print("="*60 + "\n")
    
    # Load ground truth CSV
    print(f"📂 Loading ground truth from: {CSV_PATH}")
    gt_df = pd.read_csv(CSV_PATH)
    print(f"✓ Ground truth loaded: {len(gt_df)} samples")
    
    # Create inference dataset
    print(f"\n📂 Loading images from: {H5_PATH}")
    dataset = InferenceDataset(H5_PATH, max_samples=MAX_SAMPLES)
    
    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_inference_batch
    )
    
    # Load model
    print(f"\n🤖 Loading model from: {MODEL_CHECKPOINT}")
    model = LocalReportModel(
        clip_checkpoint=CLIP_CHECKPOINT,
        projection_dim=256,
        device=DEVICE,
        fine_tune_sentence_encoder=False
    ).to(DEVICE)
    
    # Load trained weights
    state_dict = torch.load(MODEL_CHECKPOINT, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print("✓ Model loaded successfully\n")
    
    # Generate findings
    results = []
    
    print(f"🔮 Generating findings for first {MAX_SAMPLES} samples...\n")
    
    with torch.no_grad():
        for batch in tqdm(loader, desc="Generating", ncols=100):
            images = batch['images'].to(DEVICE)
            indices = batch['indices']
            
            # For generation, we need dummy sentences
            # Use empty list or single dummy sentence per image
            dummy_sentences = [[""] for _ in range(len(images))]
            
            # Generate reports
            generated_findings = model.generate(images, dummy_sentences)
            
            # Store results
            for idx, generated in zip(indices, generated_findings):
                # Get ground truth
                if idx < len(gt_df):
                    ground_truth = gt_df.iloc[idx]['findings']
                    if isinstance(ground_truth, float) and np.isnan(ground_truth):
                        ground_truth = "No significant findings."
                else:
                    ground_truth = "N/A"
                
                results.append({
                    'index': idx,
                    'generated_findings': generated,
                    'ground_truth_findings': ground_truth
                })
    
    # Create output DataFrame
    results_df = pd.DataFrame(results)
    
    # Save to CSV
    print(f"\n💾 Saving results to: {OUTPUT_CSV}")
    results_df.to_csv(OUTPUT_CSV, index=False)
    print(f"✓ Saved {len(results_df)} generated findings")
    
    # Show sample results
    print("\n" + "="*60)
    print(" 📊 SAMPLE RESULTS (First 3)")
    print("="*60 + "\n")
    
    for i in range(min(3, len(results_df))):
        row = results_df.iloc[i]
        print(f"Sample {i+1} (Index {row['index']}):")
        print(f"\n🔮 GENERATED:")
        print(f"   {row['generated_findings']}")
        print(f"\n✓ GROUND TRUTH:")
        print(f"   {row['ground_truth_findings'][:200]}{'...' if len(row['ground_truth_findings']) > 200 else ''}")
        print("\n" + "-"*60 + "\n")
    
    print("\n✅ Generation complete!\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    try:
        generate_findings()
    except KeyboardInterrupt:
        print("\n\n⚠️  Generation interrupted by user")
    except Exception as e:
        print(f"\n\n❌ Error during generation: {str(e)}")
        raise