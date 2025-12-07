import torch
import pandas as pd
import numpy as np
import h5py
from tqdm import tqdm

from global_model import GlobalReportModel


# ---------------------------------------------------------
# 1. Load image from H5
# ---------------------------------------------------------
import torch
import torch.nn.functional as F
import numpy as np

def load_image_from_h5(h5_file, idx):
    """
    Loads image (320×320 grayscale), converts to 3-channel,
    RESIZES to 224×224, and normalizes for CheXzero.
    """
    img = h5_file["cxr"][idx]  # (320, 320)

    # → (1, H, W)
    img = torch.tensor(img, dtype=torch.float32).unsqueeze(0)

    # Resize to 224×224 BEFORE repeating channels
    img = F.interpolate(img.unsqueeze(0), size=(224, 224), mode="bilinear", align_corners=False)
    img = img.squeeze(0)

    # Repeat channels → (3, 224, 224)
    img = img.repeat(3, 1, 1)

    # Normalize
    img = (img - 101.48761) / 83.43944

    return img  # (3, 224, 224) - no batch dimension


def load_batch_from_h5(h5_file, indices):
    """
    Load a batch of images from H5 file.
    Returns: (B, 3, 224, 224) tensor
    """
    batch = []
    for idx in indices:
        img = load_image_from_h5(h5_file, idx)
        batch.append(img)
    return torch.stack(batch)


# ---------------------------------------------------------
# 2. Generate impressions for an entire H5 file (BATCHED)
# ---------------------------------------------------------
def generate_for_partition(
    h5_path,
    csv_gt_path,
    clip_checkpoint,
    decoder_checkpoint,
    output_csv,
    batch_size=16,
    device="cuda"
):
    print("\n======================================")
    print(f" GENERATING IMPRESSIONS FOR {h5_path}")
    print("======================================\n")

    # ----------------------------
    # Load model
    # ----------------------------
    print("📦 Loading model...")
    model = GlobalReportModel(clip_checkpoint=clip_checkpoint, device=device).to(device)
    decoder_state = torch.load(decoder_checkpoint, map_location=device)
    model.decoder.load_state_dict(decoder_state)
    model.eval()
    print("✓ Model ready\n")

    # ----------------------------
    # Load ground truth CSV
    # ----------------------------
    print("📄 Loading ground truth impressions...")
    df_gt = pd.read_csv(csv_gt_path)
    print(f"✓ Loaded {len(df_gt)} ground truth impressions\n")

    # ----------------------------
    # Open H5 file
    # ----------------------------
    print("📸 Opening H5 file:", h5_path)
    h5 = h5py.File(h5_path, "r")
    num_images = len(h5["cxr"])
    print(f"✓ Found {num_images} images")
    print(f"✓ Using batch size: {batch_size}\n")

    # ----------------------------
    # Storage lists
    # ----------------------------
    filenames = []
    gt_impressions = []
    gen_impressions = []

    # ----------------------------
    # Loop through ALL images in BATCHES
    # ----------------------------
    print("🔄 Generating impressions in batches...")
    
    num_batches = (num_images + batch_size - 1) // batch_size
    
    for batch_idx in tqdm(range(num_batches), total=num_batches):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_images)
        batch_indices = list(range(start_idx, end_idx))
        
        # Load batch of images
        imgs = load_batch_from_h5(h5, batch_indices).to(device)
        
        # Generate text for batch
        with torch.no_grad():
            generated_batch = model.generate(imgs, max_length=64, num_beams=4)
        
        # Store results
        for idx, generated in zip(batch_indices, generated_batch):
            filenames.append(df_gt.iloc[idx]["filename"])
            gt_impressions.append(df_gt.iloc[idx]["impression"])
            gen_impressions.append(generated)

    # Close H5 file
    h5.close()

    # ----------------------------
    # Build output dataframe
    # ----------------------------
    print("\n🧪 Building output CSV...")
    df_out = pd.DataFrame({
        "filename": filenames,
        "ground_truth": gt_impressions,
        "generated": gen_impressions
    })

    df_out.to_csv(output_csv, index=False)
    print(f"✓ Saved → {output_csv}\n")

    return df_out


# ---------------------------------------------------------
# 3. Run if executed directly
# ---------------------------------------------------------
if __name__ == "__main__":
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

    CLIP_CHECKPOINT = "/content/drive/MyDrive/best_64_0.0001_original_35000_0.864.pt"
    DECODER_CHECKPOINT = "/content/drive/MyDrive/checkpoints_global/global_decoder_epoch5.pt"
    
    BATCH_SIZE = 16  # Adjust based on GPU memory (16 should work on most GPUs)

    # ----------------------------
    # RUN FOR P17
    # ----------------------------
    H5_P17 = "/content/drive/MyDrive/mimic_unzipped/h5/mimic_train_p17.h5"
    CSV_P17 = "/content/drive/MyDrive/mimic_unzipped/h5/mimic_train_p17_impressions.csv"
    OUT_P17 = "/content/drive/MyDrive/p17_generated_full.csv"

    generate_for_partition(
        H5_P17,
        CSV_P17,
        CLIP_CHECKPOINT,
        DECODER_CHECKPOINT,
        output_csv=OUT_P17,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )

    # ----------------------------
    # RUN FOR P18
    # ----------------------------
    H5_P18 = "/content/drive/MyDrive/mimic_unzipped/h5/mimic_train_p18.h5"
    CSV_P18 = "/content/drive/MyDrive/mimic_unzipped/h5/mimic_train_p18_impressions.csv"
    OUT_P18 = "/content/drive/MyDrive/p18_generated_full.csv"

    generate_for_partition(
        H5_P18,
        CSV_P18,
        CLIP_CHECKPOINT,
        DECODER_CHECKPOINT,
        output_csv=OUT_P18,
        batch_size=BATCH_SIZE,
        device=DEVICE
    )