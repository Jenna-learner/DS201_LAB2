import json
import os

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]


def get_transforms(img_size, augment=True):
    test_tf = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    if not augment:
        return test_tf, test_tf

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(img_size, scale=(0.7, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])
    return train_tf, test_tf


def split_index(n, val_ratio=0.1, seed=42, path="splits/split_seed42.json"):
    # chia 1 lần rồi lưu ra file, các bài sau đọc lại file này cho chắc cùng 1 phép chia
    if os.path.exists(path):
        with open(path) as f:
            split = json.load(f)
        if split["n"] == n:
            return split["train"], split["val"]

    g = torch.Generator().manual_seed(seed)
    idx = torch.randperm(n, generator=g).tolist()
    n_val = int(n * val_ratio)
    split = {"n": n, "seed": seed, "train": sorted(idx[n_val:]), "val": sorted(idx[:n_val])}

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump(split, f)
    return split["train"], split["val"]


def get_train_val_loaders(data_root, img_size, batch_size=32, augment=True, num_workers=2, seed=42):
    train_tf, test_tf = get_transforms(img_size, augment)
    train_dir = os.path.join(data_root, "train")
    val_dir = os.path.join(data_root, "val")

    if os.path.isdir(val_dir):
        train_ds = datasets.ImageFolder(train_dir, train_tf)
        val_ds = datasets.ImageFolder(val_dir, test_tf)
        classes = train_ds.classes
    else:
        # không có val/ thì tách 10% từ train, tạo 2 ImageFolder để val không bị augment
        train_full = datasets.ImageFolder(train_dir, train_tf)
        val_full = datasets.ImageFolder(train_dir, test_tf)
        train_idx, val_idx = split_index(len(train_full), seed=seed)
        train_ds = Subset(train_full, train_idx)
        val_ds = Subset(val_full, val_idx)
        classes = train_full.classes

    pin = torch.cuda.is_available()
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=num_workers, pin_memory=pin)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin)
    return train_loader, val_loader, classes


def get_test_loader(data_root, img_size, batch_size=64, num_workers=2):
    _, test_tf = get_transforms(img_size, augment=False)
    test_ds = datasets.ImageFolder(os.path.join(data_root, "test"), test_tf)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=num_workers)
    return test_loader, test_ds.classes
