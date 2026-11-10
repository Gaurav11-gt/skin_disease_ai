import pandas as pd
import torch
from pathlib import Path
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms as T
from config import IMAGENET_MEAN, IMAGENET_STD


def get_transforms(img_size, train):
    norm = T.Normalize(IMAGENET_MEAN, IMAGENET_STD)
    if train:
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
            T.RandomHorizontalFlip(), T.RandomVerticalFlip(),
            T.RandomRotation(30),
            T.ColorJitter(0.2, 0.2, 0.2, 0.02),
            T.ToTensor(), norm])
    return T.Compose([T.Resize((img_size, img_size)), T.ToTensor(), norm])


class SkinDataset(Dataset):
    def __init__(self, csv_path, img_size, train):
        self.df = pd.read_csv(csv_path)
        self.tf = get_transforms(img_size, train)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        r = self.df.iloc[i]
        img = Image.open(r["path"]).convert("RGB")
        return self.tf(img), int(r["label"])


def make_loaders(data_dir, img_size=224, batch_size=32, workers=2):
    d = Path(data_dir)
    sets = {s: SkinDataset(d / f"{s}.csv", img_size, train=(s == "train")) for s in ["train", "val", "test"]}
    pin = torch.cuda.is_available()
    tr = DataLoader(sets["train"], batch_size, shuffle=True, num_workers=workers, pin_memory=pin, drop_last=True)
    va = DataLoader(sets["val"], batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    te = DataLoader(sets["test"], batch_size, shuffle=False, num_workers=workers, pin_memory=pin)
    return tr, va, te, sets["train"].df
