"""
Data loader module for the Distracted Driver Detection project.
Provides utilities for traditional ML (flat loading) and PyTorch DataLoaders.
"""

import os
import glob
import numpy as np
from PIL import Image
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import torch
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
from torchvision import transforms


def load_images_flat(data_dir: str, img_size: int = 64):
    """
    Loads all images from train/ subdirectories (c0-c9), resizes to img_size x img_size,
    flattens to vectors, returns X and y arrays, and performs 80/20 stratified split.

    Args:
        data_dir (str): Root directory containing class subdirectories (c0-c9).
        img_size (int): Size to resize images to before flattening.

    Returns:
        tuple: (X_train, X_val, y_train, y_val)
    """
    from src.config import VAL_SPLIT, RANDOM_SEED

    X = []
    y = []

    classes = [f'c{i}' for i in range(10)]
    
    for label, class_name in enumerate(classes):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue
            
        img_paths = glob.glob(os.path.join(class_path, '*.jpg'))
        
        for img_path in tqdm(img_paths, desc=f"Loading {class_name}"):
            try:
                with Image.open(img_path) as img:
                    img_resized = img.convert('RGB').resize((img_size, img_size))
                    img_array = np.array(img_resized, dtype=np.float32) / 255.0
                    X.append(img_array.flatten())
                    y.append(label)
            except Exception as e:
                print(f"Error loading {img_path}: {e}")

    X = np.array(X)
    y = np.array(y)

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=y
    )

    return X_train, X_val, y_train, y_val


class DriverDataset(Dataset):
    """
    PyTorch Dataset for Distracted Driver dataset.
    Loads images lazily and applies transforms.
    """
    def __init__(self, data_dir: str, transform=None):
        self.data_dir = data_dir
        self.transform = transform
        self.img_paths = []
        self.labels = []

        classes = [f'c{i}' for i in range(10)]
        for label, class_name in enumerate(classes):
            class_path = os.path.join(data_dir, class_name)
            if not os.path.isdir(class_path):
                continue
                
            paths = glob.glob(os.path.join(class_path, '*.jpg'))
            self.img_paths.extend(paths)
            self.labels.extend([label] * len(paths))

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, idx):
        img_path = self.img_paths[idx]
        label = self.labels[idx]

        img = Image.open(img_path).convert('RGB')
        
        if self.transform:
            img = self.transform(img)
            
        return img, label


def get_data_loaders(data_dir: str, img_size: int = 224, batch_size: int = 32, augment: bool = True):
    """
    Returns train and val DataLoaders with stratified split.

    Args:
        data_dir (str): Root directory containing class subdirectories.
        img_size (int): Image resize dimension.
        batch_size (int): Batch size.
        augment (bool): Whether to apply data augmentation to training data.

    Returns:
        tuple: (train_loader, val_loader)
    """
    from src.config import VAL_SPLIT, RANDOM_SEED

    # ImageNet stats
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            normalize
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            normalize
        ])

    val_transform = transforms.Compose([
        transforms.Resize(img_size + 32),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize
    ])

    # To apply different transforms to train/val, we create two dataset instances
    # and use SubsetRandomSampler with stratified split indices.
    
    # Use full dataset just to get labels for stratified split
    temp_dataset = DriverDataset(data_dir, transform=None)
    labels = temp_dataset.labels
    indices = list(range(len(labels)))
    
    if len(indices) == 0:
        raise ValueError(f"No images found in {data_dir}")

    train_indices, val_indices = train_test_split(
        indices, test_size=VAL_SPLIT, random_state=RANDOM_SEED, stratify=labels
    )
    
    train_sampler = SubsetRandomSampler(train_indices)
    val_sampler = SubsetRandomSampler(val_indices)

    train_dataset = DriverDataset(data_dir, transform=train_transform)
    val_dataset = DriverDataset(data_dir, transform=val_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=train_sampler)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=val_sampler)

    return train_loader, val_loader
