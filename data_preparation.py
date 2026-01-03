import os
import json
import random
from pathlib import Path
from typing import Tuple, List, Dict
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

class ImageCaptionDataset(Dataset):
    """Base dataset for image-caption pairs"""
    def __init__(self, data: List[Dict], transform=None, split="train"):
        self.data = data
        self.transform = transform
        self.split = split
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        image = item['image']
        caption = item['caption']
        image = image.convert("RGB")
        if self.transform:
            image = self.transform(image)        
        return {
            'image': image,
            'caption': caption,
            'image_id': item.get('image_id', idx),
            'is_ood': item.get('is_ood', False)
        }

def create_ood_distortions(image: Image.Image) -> Image.Image:
    """
    Create OOD samples through distortions:
    - Gaussian blur
    - Color shift (adjust saturation and hue)
    - Brightness changes
    """
    # Random blur
    print(type(image))
    print(image.mode)
    blur_radius = random.uniform(2.0, 5.0)
    image = image.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Color shift
    enhancer = ImageEnhance.Color(image)
    image = enhancer.enhance(random.uniform(0.3, 0.7))  # Desaturate
    
    # Brightness shift
    enhancer = ImageEnhance.Brightness(image)
    image = enhancer.enhance(random.uniform(0.6, 0.9))  # Darken
    print(image.size)
    print(image.mode)    
    return image

def load_coco_subset(num_samples: int, data_dir: str, cache_dir: str) -> List[Dict]:
    """Load COCO dataset subset"""
    print(f"Loading COCO dataset (target: {num_samples} samples)...")
    
    # Load COCO captions from HuggingFace
    dataset = load_dataset("jxie/coco_captions", split="train", cache_dir=cache_dir)

   # Sample subset
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    
    data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing COCO")):
        data.append({
            'image': item['image'],
            'caption': item['caption'],
            'image_id': idx,
            'is_ood': False
        })
    
    return data[:num_samples]

def load_flickr_subset(num_samples: int, data_dir: str, cache_dir: str) -> List[Dict]:
    """Load Flickr30k dataset subset"""
    print(f"Loading Flickr30k dataset (target: {num_samples} samples)...")
    
    dataset = load_dataset("nlphuji/flickr30k", split="test", cache_dir=cache_dir)
    
    if len(dataset) > num_samples:
        indices = random.sample(range(len(dataset)), num_samples)
        dataset = dataset.select(indices)
    
    data = []
    for idx, item in enumerate(tqdm(dataset, desc="Processing Flickr")):
        captions = item.get('caption', [])
        if not captions:
            continue
        
        data.append({
            'image': item['image'],
            'caption': captions[0],
            'image_id': idx,
            'is_ood': False
        })
    
    return data[:num_samples]

def create_ood_split(data: List[Dict], num_ood: int, seed: int) -> List[Dict]:
    """
    Create OOD split by distorting a subset of images
    """
    print(f"\nCreating {num_ood} OOD samples with distortions...")
    random.seed(seed)
    
    # Select random samples to distort
    ood_indices = random.sample(range(len(data)), min(num_ood, len(data)))
    
    # Assign unique integer IDs for OOD samples starting after existing indices
    base_id = len(data)
    ood_data = []
    for i, idx in enumerate(tqdm(ood_indices, desc="Generating OOD")):
        original = data[idx]
        distorted_image = create_ood_distortions(original['image'])
        
        ood_data.append({
            'image': distorted_image,
            'caption': original['caption'],
            'image_id': base_id + i,
            'is_ood': True,
            'original_id': original['image_id']
        })
    
    return ood_data

def prepare_datasets(config) -> Tuple[List, List, List]:
    """
    Main data preparation function
    Returns: (train_data, val_data, test_data)
    Test data includes OOD samples
    """
    print("="*60)
    print("DATA PREPARATION")
    print("="*60)
    
    # Create directories
    Path(config.data.data_dir).mkdir(parents=True, exist_ok=True)
    Path(config.data.cache_dir).mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    if config.data.dataset_name == "coco":
        data = load_coco_subset(
            config.data.num_samples,
            config.data.data_dir,
            config.data.cache_dir
        )
    else:
        data = load_flickr_subset(
            config.data.num_samples,
            config.data.data_dir,
            config.data.cache_dir
        )
    print(f"\nLoaded {len(data)} samples")
    # Create OOD split
    ood_data = create_ood_split(data, config.data.num_ood, config.experiment.seeds[0])
    
    # Split data: 800 train / 100 val / 100 test + 100 OOD
    random.seed(config.experiment.seeds[0])
    random.shuffle(data)
    
    train_data = data[:config.data.train_size]
    val_data = data[config.data.train_size:config.data.train_size + config.data.val_size]
    test_data = data[config.data.train_size + config.data.val_size:
                     config.data.train_size + config.data.val_size + config.data.test_size]
    
    # Add OOD to test set
    test_data.extend(ood_data)
    
    print(f"\nDataset splits:")
    print(f"  Train: {len(train_data)} samples")
    print(f"  Val:   {len(val_data)} samples")
    print(f"  Test:  {len(test_data)} samples ({config.data.test_size} in-dist + {len(ood_data)} OOD)")
    
    # Save split info
    split_info = {
        'train_size': len(train_data),
        'val_size': len(val_data),
        'test_size': len(test_data),
        'ood_size': len(ood_data),
        'seed': config.experiment.seeds[0]
    }
    
    with open(os.path.join(config.data.data_dir, 'split_info.json'), 'w') as f:
        json.dump(split_info, f, indent=2)
    
    return train_data, val_data, test_data

def get_transforms(image_size: int, is_train: bool = False):
    """Get image transforms for preprocessing"""
    if is_train:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])
    else:
        return transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                               std=[0.229, 0.224, 0.225])
        ])

def create_dataloaders(train_data, val_data, test_data, config):
    """Create PyTorch dataloaders"""
    image_size = config.hardware.get_image_size()
    num_workers = config.hardware.get_num_workers()
    
    train_dataset = ImageCaptionDataset(
        train_data, 
        transform=get_transforms(image_size, is_train=True),
        split="train"
    )
    val_dataset = ImageCaptionDataset(
        val_data,
        transform=get_transforms(image_size, is_train=False),
        split="val"
    )
    test_dataset = ImageCaptionDataset(
        test_data,
        transform=get_transforms(image_size, is_train=False),
        split="test"
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.hardware.get_clip_batch_size(),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config.hardware.get_clip_batch_size(),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.hardware.get_clip_batch_size(),
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=True if num_workers > 0 else False
    )
    
    return train_loader, val_loader, test_loader
