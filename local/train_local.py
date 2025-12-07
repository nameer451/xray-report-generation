import os
import torch
from torch.utils.data import DataLoader, ConcatDataset, random_split
from tqdm import tqdm
from transformers import BartTokenizer

from local_model import LocalReportModel
from data.dataset_local import LocalDataset, collate_local_batch


# =========================================================
# CONFIG
# =========================================================

PARTITIONS = [10, 11, 12, 13, 14, 15]

H5_DIR = "/content/drive/MyDrive/mimic_unzipped/h5"
CSV_DIR = "/content/drive/MyDrive/mimic_cleaned_csvs"
CLIP_CHECKPOINT = "/content/drive/MyDrive/best_64_0.0001_original_35000_0.864.pt"

# CHECKPOINTS SAVED DIRECTLY TO GOOGLE DRIVE
CHECKPOINT_DIR = "/content/drive/MyDrive/checkpoints_local"

VAL_RATIO = 0.10
BATCH_SIZE = 4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# =========================================================
# DATASET LOADING
# =========================================================

def build_partition_datasets(max_sentences=10):
    datasets = []
    for p in PARTITIONS:
        h5_path = os.path.join(H5_DIR, f"mimic_train_p{p}.h5")
        csv_path = os.path.join(CSV_DIR, f"mimic_train_p{p}_findings.csv")

        if not (os.path.exists(h5_path) and os.path.exists(csv_path)):
            print(f"⚠️  Skipping missing partition {p}")
            continue

        ds = LocalDataset(h5_path, csv_path, max_sentences)
        datasets.append(ds)
        print(f"✓ Loaded partition {p}: {len(ds)} samples")

    return datasets


def build_train_val_loaders(max_sentences=10):
    datasets = build_partition_datasets(max_sentences)
    if not datasets:
        raise RuntimeError("No datasets found for training.")

    full_dataset = ConcatDataset(datasets)
    val_size = int(len(full_dataset) * VAL_RATIO)
    train_size = len(full_dataset) - val_size

    train_ds, val_ds = random_split(full_dataset, [train_size, val_size])

    print(f"\n📦 Full dataset: {len(full_dataset):,}")
    print(f"→ Train: {train_size:,}")
    print(f"→ Val:   {val_size:,}\n")

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_local_batch,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        collate_fn=collate_local_batch,
    )

    return train_loader, val_loader


# =========================================================
# STAGE 1 — ALIGNMENT TRAINING
# =========================================================

def train_stage1_alignment(epochs=5, lr=1e-4):
    print("\n" + "="*60)
    print(" STAGE 1 — ALIGNMENT TRAINING ")
    print("="*60 + "\n")

    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    train_loader, val_loader = build_train_val_loaders()

    model = LocalReportModel(
        clip_checkpoint=CLIP_CHECKPOINT,
        projection_dim=256,
        device=DEVICE,
        fine_tune_sentence_encoder=True,  # we override freezing rules below
    ).to(DEVICE)

    # =====================================================
    # FREEZING RULES FOR STAGE 1
    # =====================================================

    print("🔒 Freezing patch encoder, decoder, reconstructor...")
    for p in model.decoder.parameters(): p.requires_grad = False
    for p in model.reconstruct.parameters(): p.requires_grad = False
    # patch_encoder already frozen in its __init__

    # --- Freeze all BERT layers ---
    print("🔒 Freezing ALL BERT layers...")
    for p in model.sentence_encoder.parameters():
        p.requires_grad = False

    # --- UNFREEZE ONLY LAST 2 LAYERS of BERT ---
    print("🔓 Unfreezing LAST 2 layers of BERT (layers 10 and 11)...")
    for idx in [10, 11]:
        for p in model.sentence_encoder.encoder.encoder.layer[idx].parameters():
            p.requires_grad = True

    # --- UNFREEZE BERT POOLER ---
    print("🔓 Unfreezing BERT pooler...")
    for p in model.sentence_encoder.encoder.pooler.parameters():
        p.requires_grad = True

    # --- UNFREEZE PROJECTION HEAD COMPLETELY ---
    print("🔓 Unfreezing projection head...")
    for p in model.projection.parameters():
        p.requires_grad = True

    # Collect all trainable parameters
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    print("\n🔍 Trainable parameters in Stage 1:")
    total_params = 0
    for name, param in model.named_parameters():
        if param.requires_grad:
            total_params += param.numel()
            print(f"  ✓ {name}: {param.numel():,}")
    print(f"\n📊 Total trainable params: {total_params:,}\n")

    optimizer = torch.optim.AdamW(trainable_params, lr=lr, weight_decay=0.01)

    best_val_loss = float("inf")

    # =====================================================
    # EPOCH LOOP
    # =====================================================

    for epoch in range(epochs):
        print(f"\n────────── EPOCH {epoch+1}/{epochs} ──────────")

        # ===========================
        # TRAIN
        # ===========================
        model.train()
        train_loss_sum = 0
        train_batches = 0

        pbar = tqdm(train_loader, desc=f"Train {epoch+1}", ncols=120)
        for batch in pbar:
            images = batch["images"].to(DEVICE)
            sentences = batch["sentences"]

            loss = model.forward_alignment(images, sentences)

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_loss_sum += loss.item()
            train_batches += 1

            avg_loss = train_loss_sum / train_batches
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"})

        avg_train_loss = train_loss_sum / train_batches
        print(f"\n✓ Avg TRAIN alignment loss: {avg_train_loss:.4f}")

        # ===========================
        # VALIDATION
        # ===========================
        model.eval()
        val_loss_sum = 0
        val_batches = 0

        pbar = tqdm(val_loader, desc=f"Val {epoch+1}", ncols=120)
        with torch.no_grad():
            for batch in pbar:
                images = batch["images"].to(DEVICE)
                sentences = batch["sentences"]

                loss = model.forward_alignment(images, sentences)
                val_loss_sum += loss.item()
                val_batches += 1

                avg_loss = val_loss_sum / val_batches
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "avg": f"{avg_loss:.4f}"})

        avg_val_loss = val_loss_sum / val_batches
        print(f"✓ Avg VAL alignment loss: {avg_val_loss:.4f}")

        # SAVE BEST
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
            torch.save(model.state_dict(), path)
            print(f"🌟 New best model saved → {path}")

    # Save last checkpoint
    final_path = os.path.join(CHECKPOINT_DIR, "stage1_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Stage 1 complete. Final checkpoint saved: {final_path}\n")


# =========================================================
# STAGE 2 — GENERATION TRAINING
# =========================================================

def train_stage2_generation():
    print("\n" + "="*60)
    print(" STAGE 2 — REPORT GENERATION ")
    print("="*60 + "\n")

    train_loader, val_loader = build_train_val_loaders()

    model = LocalReportModel(
        clip_checkpoint=CLIP_CHECKPOINT,
        projection_dim=256,
        device=DEVICE,
        fine_tune_sentence_encoder=False,
    ).to(DEVICE)

    # LOAD STAGE 1 WEIGHTS
    best_path = os.path.join(CHECKPOINT_DIR, "stage1_best.pt")
    final_path = os.path.join(CHECKPOINT_DIR, "stage1_final.pt")

    if os.path.exists(best_path):
        load_path = best_path
    elif os.path.exists(final_path):
        load_path = final_path
    else:
        raise FileNotFoundError("❌ No Stage 1 checkpoints found in Drive!")

    print(f"📂 Loading Stage 1 weights: {load_path}")
    state_dict = torch.load(load_path, map_location=DEVICE)
    model.load_state_dict(state_dict, strict=False)
    print("⚠️  Loaded with strict=False (reconstructor randomly initialized)")

    # FREEZE ALL ALIGNMENT MODULES FOR STAGE 2
    print("🔒 Freezing projection, sentence encoder, and align modules...")
    for p in model.projection.parameters(): p.requires_grad = False
    for p in model.sentence_encoder.parameters(): p.requires_grad = False
    for p in model.align.parameters(): p.requires_grad = False
    # patch encoder already frozen

    # TRAIN ONLY DECODER + RECONSTRUCTOR
    trainable_params = list(model.decoder.parameters()) + list(model.reconstruct.parameters())
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-5, weight_decay=0.01)

    tokenizer = BartTokenizer.from_pretrained("GanjinZero/biobart-base")
    best_val_loss = float("inf")

    # =====================================================
    # PHASE 2A — ABNORMAL CURRICULUM
    # =====================================================

    def is_abnormal(text):
        keys = [
            "effusion", "pneumonia", "consolidation", "cardiomegaly", "edema",
            "atelectasis", "pneumothorax", "nodule", "mass", "fracture",
            "opacity", "infiltrate", "abnormal"
        ]
        lowercase = text.lower()
        return any(k in lowercase for k in keys)

    abnormal_indices = [
        i for i in range(len(train_loader.dataset))
        if is_abnormal(train_loader.dataset[i]["findings"])
    ]

    abnormal_subset = torch.utils.data.Subset(train_loader.dataset, abnormal_indices)
    abnormal_loader = DataLoader(
        abnormal_subset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        collate_fn=collate_local_batch
    )

    print(f"✓ Abnormal samples: {len(abnormal_indices):,}")

    # ------------------ PHASE 2A TRAINING ------------------

    for epoch in range(4):
        print(f"\n────────── PHASE 2A EPOCH {epoch+1}/4 ──────────")

        # TRAIN
        model.train()
        train_t = train_g = train_r = 0
        batches = 0

        pbar = tqdm(abnormal_loader, desc=f"P2A Train {epoch+1}", ncols=120)
        for batch in pbar:
            images = batch["images"].to(DEVICE)
            sentences_list = batch["sentences"]
            findings = batch["findings"]

            enc = tokenizer(findings, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(DEVICE)

            total, gen, rec = model.forward_generation(
                images, enc["input_ids"], enc["attention_mask"], sentences_list
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_t += total.item()
            train_g += gen
            train_r += rec
            batches += 1

            pbar.set_postfix({
                "tot": f"{total.item():.3f}",
                "avg": f"{train_t/batches:.3f}"
            })

        avg_total = train_t / batches
        avg_gen = train_g / batches
        avg_rec = train_r / batches
        print(f"✓ TRAIN → total={avg_total:.4f}, gen={avg_gen:.4f}, rec={avg_rec:.4f}")

        # VALIDATION
        model.eval()
        val_t = val_g = val_r = 0
        val_batches = 0

        pbar = tqdm(val_loader, desc=f"P2A Val {epoch+1}", ncols=120)
        with torch.no_grad():
            for batch in pbar:
                images = batch["images"].to(DEVICE)
                sentences_list = batch["sentences"]
                findings = batch["findings"]

                enc = tokenizer(findings, padding=True, truncation=True, max_length=512,
                                return_tensors="pt").to(DEVICE)

                total, gen, rec = model.forward_generation(
                    images, enc["input_ids"], enc["attention_mask"], sentences_list
                )

                val_t += total.item()
                val_g += gen
                val_r += rec
                val_batches += 1

                pbar.set_postfix({
                    "tot": f"{total.item():.3f}",
                    "avg": f"{val_t/val_batches:.3f}"
                })

        avg_val_total = val_t / val_batches
        print(f"✓ VAL   → total={avg_val_total:.4f}")

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            path = os.path.join(CHECKPOINT_DIR, "stage2a_best.pt")
            torch.save(model.state_dict(), path)
            print(f"🌟 New best Stage 2A model saved → {path}")

    # =====================================================
    # PHASE 2B — FULL TRAINING
    # =====================================================

    for g in optimizer.param_groups:
        g["lr"] *= 0.5
    print(f"\n📉 LR reduced to {optimizer.param_groups[0]['lr']:.2e}")

    for epoch in range(3):
        print(f"\n────────── PHASE 2B EPOCH {epoch+1}/3 ──────────")

        # TRAIN
        model.train()
        train_t = train_g = train_r = 0
        batches = 0

        pbar = tqdm(train_loader, desc=f"P2B Train {epoch+1}", ncols=120)
        for batch in pbar:
            images = batch["images"].to(DEVICE)
            sentences_list = batch["sentences"]
            findings = batch["findings"]

            enc = tokenizer(findings, padding=True, truncation=True, max_length=512,
                            return_tensors="pt").to(DEVICE)

            total, gen, rec = model.forward_generation(
                images, enc["input_ids"], enc["attention_mask"], sentences_list
            )

            optimizer.zero_grad()
            total.backward()
            torch.nn.utils.clip_grad_norm_(trainable_params, 1.0)
            optimizer.step()

            train_t += total.item()
            train_g += gen
            train_r += rec
            batches += 1

            pbar.set_postfix({
                "tot": f"{total.item():.3f}",
                "avg": f"{train_t/batches:.3f}"
            })

        avg_total = train_t / batches
        avg_gen = train_g / batches
        avg_rec = train_r / batches
        print(f"✓ TRAIN → total={avg_total:.4f}, gen={avg_gen:.4f}, rec={avg_rec:.4f}")

        # VALIDATION
        model.eval()
        val_t = val_g = val_r = 0
        val_batches = 0

        pbar = tqdm(val_loader, desc=f"P2B Val {epoch+1}", ncols=120)
        with torch.no_grad():
            for batch in pbar:
                images = batch["images"].to(DEVICE)
                sentences_list = batch["sentences"]
                findings = batch["findings"]

                enc = tokenizer(findings, padding=True, truncation=True, max_length=512,
                                return_tensors="pt").to(DEVICE)

                total, gen, rec = model.forward_generation(
                    images, enc["input_ids"], enc["attention_mask"], sentences_list
                )

                val_t += total.item()
                val_g += gen
                val_r += rec
                val_batches += 1

                pbar.set_postfix({
                    "tot": f"{total.item():.3f}",
                    "avg": f"{val_t/val_batches:.3f}"
                })

        avg_val_total = val_t / val_batches
        print(f"✓ VAL   → total={avg_val_total:.4f}")

        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            path = os.path.join(CHECKPOINT_DIR, "stage2b_best.pt")
            torch.save(model.state_dict(), path)
            print(f"🌟 New best Stage 2B model saved → {path}")

    # FINAL SAVE
    final_path = os.path.join(CHECKPOINT_DIR, "stage2_final.pt")
    torch.save(model.state_dict(), final_path)
    print(f"\n✓ Stage 2 complete. Final model saved to {final_path}\n")


# =========================================================
# MAIN
# =========================================================

if __name__ == "__main__":
    train_stage2_generation()




