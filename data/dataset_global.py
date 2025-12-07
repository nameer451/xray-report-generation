import torch
from torch.utils.data import Dataset
import h5py
import pandas as pd
from transformers import AutoTokenizer
from torchvision.transforms import Resize, InterpolationMode


class GlobalDataset(Dataset):

    def __init__(self, h5_path, csv_path, device="cuda", max_len=64):
        print(f"✓ Loading H5 images: {h5_path}")
        print(f"✓ Loading impressions: {csv_path}")

        self.imgs = h5py.File(h5_path, "r")["cxr"]
        df = pd.read_csv(csv_path)

        self.texts = df["impression"].fillna("").tolist()
        self.tokenizer = AutoTokenizer.from_pretrained("GanjinZero/biobart-base")

        # REQUIRED FIX FOR CHEXZERO VI
        self.resize = Resize((224, 224), interpolation=InterpolationMode.BICUBIC)

        self.max_len = max_len
        self.device = device

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):

        # Image (single-channel replicated to 3)
        img = self.imgs[idx]  # (320, 320)
        img = torch.tensor(img).unsqueeze(0).repeat(3, 1, 1).float()

        # FIX: resize to 224x224 which the checkpoint expects
        img = self.resize(img)

        # Tokenize text
        enc = self.tokenizer(
            self.texts[idx],
            padding="max_length",
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        )

        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)

        return {
            "img": img,
            "input_ids": input_ids,
            "attention_mask": attention_mask
        }
