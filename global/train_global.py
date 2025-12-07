import torch
import os
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from global_model import GlobalReportModel
from data.dataset_global import GlobalDataset
from transformers import AutoTokenizer


# ============================================================
# Load p10–p15 with 98/2 train/val split
# ============================================================

def load_p_datasets(
    base_path,
    p_start=10,
    p_end=15,
    train_ratio=0.98,
    batch_size=8,
    num_workers=2
):
    """
    Loads dataset for p10–p15.
    Performs deterministic 98% train / 2% validation split.
    Returns train_loaders and val_loaders.
    """

    train_loaders = []
    val_loaders = []

    for p in range(p_start, p_end + 1):
        h5_path = os.path.join(base_path, f"mimic_train_p{p}.h5")
        csv_path = os.path.join(base_path, f"mimic_train_p{p}_impressions.csv")

        if not os.path.exists(h5_path) or not os.path.exists(csv_path):
            print(f"⚠️ Missing dataset p{p}, skipping...")
            continue

        print(f"\n📂 Loading p{p}...")

        dataset = GlobalDataset(h5_path, csv_path)
        total_len = len(dataset)
        val_len = max(1, int(total_len * (1 - train_ratio)))
        train_len = total_len - val_len

        # Deterministic split
        generator = torch.Generator().manual_seed(42)
        train_set, val_set = random_split(dataset, [train_len, val_len], generator)

        print(f"  → Train: {train_len}   Val: {val_len}")

        # Create loaders
        train_loader = DataLoader(
            train_set,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True
        )

        val_loader = DataLoader(
            val_set,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True
        )

        train_loaders.append((train_loader, h5_path))
        val_loaders.append((val_loader, h5_path))

    return train_loaders, val_loaders


# ============================================================
# Global Training Loop
# ============================================================

def train():
    # ============================================
    # Configuration
    # ============================================
    BASE_PATH = "/content/drive/MyDrive/mimic_unzipped/h5"
    CLIP_CHECKPOINT = "/content/drive/MyDrive/best_64_0.0001_original_35000_0.864.pt"
    CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints_global"

    BATCH_SIZE = 8
    NUM_WORKERS = 2
    LEARNING_RATE = 1e-5
    NUM_EPOCHS = 10

    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print("\n" + "="*70)
    print(" 🟦 GLOBAL IMPRESSION TRAINING (with p10–p15 & validation)")
    print("="*70 + "\n")

    # ============================================
    # Load dataset splits
    # ============================================
    print("📦 Splitting datasets (p10–p15 into 98% train / 2% val)...")
    train_loaders, val_loaders = load_p_datasets(
        BASE_PATH,
        p_start=10,
        p_end=15,
        train_ratio=0.98,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS
    )

    if not train_loaders:
        print("❌ No datasets found. Exiting.")
        return

    print(f"\n✓ Loaded {len(train_loaders)} p-folders.\n")

    # ============================================
    # Initialize Model
    # ============================================
    print("🔧 Initializing model...")
    model = GlobalReportModel(CLIP_CHECKPOINT, device=DEVICE).to(DEVICE)
    model.train()

    # Only train decoder
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    print(f"Trainable params: {sum(p.numel() for p in trainable_params):,}")

    optimizer = torch.optim.Adam(trainable_params, lr=LEARNING_RATE)
    tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")

    # ============================================
    # Training Loop
    # ============================================
    best_val_loss = float("inf")

    for epoch in range(1, NUM_EPOCHS + 1):
        print(f"\n\n=== 🌟 Epoch {epoch}/{NUM_EPOCHS} ===\n")
        model.train()

        total_train_loss = 0
        train_steps = 0

        # =======================================
        # TRAINING on all p10–p15
        # =======================================
        for (loader, h5_path) in train_loaders:
            progress = tqdm(loader, desc=f"Training on {os.path.basename(h5_path)}")

            for batch in progress:
                imgs = batch["img"].to(DEVICE)
                input_ids = batch["input_ids"].to(DEVICE)
                attention_mask = batch["attention_mask"].to(DEVICE)

                loss = model(imgs, input_ids, attention_mask)

                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()

                total_train_loss += loss.item()
                train_steps += 1

                progress.set_postfix({"loss": f"{loss.item():.4f}"})

        # Compute average train loss
        avg_train_loss = total_train_loss / train_steps
        print(f"\n📉 **Average Train Loss (Epoch {epoch}): {avg_train_loss:.4f}**")

        # =======================================
        # VALIDATION
        # =======================================
        print("\n🧪 Running validation on p10–p15...")
        model.eval()

        val_loss_total = 0
        val_steps = 0

        with torch.no_grad():
            for (loader, h5_path) in val_loaders:
                for batch in loader:
                    imgs = batch["img"].to(DEVICE)
                    input_ids = batch["input_ids"].to(DEVICE)
                    attention_mask = batch["attention_mask"].to(DEVICE)

                    loss = model(imgs, input_ids, attention_mask)

                    val_loss_total += loss.item()
                    val_steps += 1

        avg_val_loss = val_loss_total / val_steps
        print(f"🧪 **Average Validation Loss (Epoch {epoch}): {avg_val_loss:.4f}**")

        # =======================================
        # CHECKPOINTING (unchanged)
        # =======================================
        ckpt_path = os.path.join(CHECKPOINT_DIR, f"global_decoder_epoch{epoch}.pt")
        torch.save(model.decoder.state_dict(), ckpt_path)
        print(f"💾 Saved checkpoint: {ckpt_path}")

        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            best_path = os.path.join(CHECKPOINT_DIR, "global_decoder_best.pt")
            torch.save(model.decoder.state_dict(), best_path)
            print(f"🏅 Best model updated! Validation Loss: {avg_val_loss:.4f}")

        # =======================================
        # SAMPLE GENERATION (unchanged)
        # =======================================
        first_loader = train_loaders[0][0]
        sample_batch = next(iter(first_loader))
        model.eval()

        with torch.no_grad():
            sample_img = sample_batch["img"][:1].to(DEVICE)
            sample_text = model.generate(sample_img, max_length=64, num_beams=4)

        print("\n📝 Sample impression:")
        print(sample_text[0])
        print("\n" + "="*70 + "\n")


    print("\n🎉 TRAINING COMPLETE!")
    print(f"Best validation loss: {best_val_loss:.4f}")


# ============================================================
# MAIN ENTRY POINT
# ============================================================

if __name__ == "__main__":
    train()

