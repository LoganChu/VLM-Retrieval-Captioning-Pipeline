"""
Linear probe on top of frozen CLIP embeddings
Maps image embeddings to text embedding space using contrastive learning
Time budget: 1-1.25 hours (with 3 seeds)
"""

import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from typing import Dict
import os
import json

class LinearProbe(nn.Module):
    """Single linear layer probe for image-to-text retrieval"""
    
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim)
        self.temperature = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))
    
    def forward(self, image_features):
        # Project image features
        projected = self.linear(image_features)
        # Normalize
        projected = F.normalize(projected, p=2, dim=-1)
        return projected
    
    def get_temperature(self):
        return self.temperature.exp()

class EmbeddingDataset(Dataset):
    """Dataset for pre-computed embeddings"""
    
    def __init__(self, image_embeddings, text_embeddings):
        self.image_embeddings = image_embeddings
        self.text_embeddings = text_embeddings
    
    def __len__(self):
        return len(self.image_embeddings)
    
    def __getitem__(self, idx):
        return {
            'image_emb': self.image_embeddings[idx],
            'text_emb': self.text_embeddings[idx]
        }

def info_nce_loss(image_features, text_features, temperature):
    """
    InfoNCE contrastive loss
    Treats all other samples in batch as negatives
    """
    batch_size = image_features.size(0)
    
    # Normalize features
    image_features = F.normalize(image_features, p=2, dim=-1)
    text_features = F.normalize(text_features, p=2, dim=-1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, text_features.t()) * temperature
    
    # Labels are diagonal (i-th image matches i-th text)
    labels = torch.arange(batch_size, device=logits.device)
    
    # Compute cross-entropy loss
    loss_i2t = F.cross_entropy(logits, labels)
    loss_t2i = F.cross_entropy(logits.t(), labels)
    
    loss = (loss_i2t + loss_t2i) / 2
    
    # Compute accuracy
    with torch.no_grad():
        pred_i2t = logits.argmax(dim=1)
        acc_i2t = (pred_i2t == labels).float().mean()
    
    return loss, acc_i2t

class LinearProbeTrainer:
    """Trainer for linear probe"""
    
    def __init__(self, config, train_embs, val_embs, test_embs):
        self.config = config
        self.device = config.device
        
        # Create datasets
        train_dataset = EmbeddingDataset(
            train_embs['image'], train_embs['text']
        )
        val_dataset = EmbeddingDataset(
            val_embs['image'], val_embs['text']
        )
        test_dataset = EmbeddingDataset(
            test_embs['image'], test_embs['text']
        )
        
        # Create dataloaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=config.hardware.get_linear_probe_bs(),
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=config.hardware.get_linear_probe_bs(),
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=config.hardware.get_linear_probe_bs(),
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Store full embeddings for evaluation
        self.val_image_embs = val_embs['image'].to(self.device)
        self.val_text_embs = val_embs['text'].to(self.device)
        self.test_image_embs = test_embs['image'].to(self.device)
        self.test_text_embs = test_embs['text'].to(self.device)
        self.test_is_ood = test_embs['is_ood']
        
        # Initialize model
        embed_dim = config.model.clip_embed_dim
        self.model = LinearProbe(embed_dim, embed_dim).to(self.device)
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config.model.probe_lr,
            weight_decay=config.model.weight_decay
        )
        
        # Training history
        self.history = {
            'train_loss': [], 'train_acc': [],
            'val_loss': [], 'val_acc': [],
            'val_recall@1': [], 'val_recall@5': []
        }
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        total_acc = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            image_emb = batch['image_emb'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            
            # Forward pass
            projected = self.model(image_emb)
            temperature = self.model.get_temperature()
            
            # Compute loss
            loss, acc = info_nce_loss(projected, text_emb, temperature)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), 
                self.config.model.gradient_clip
            )
            self.optimizer.step()
            
            # Update metrics
            total_loss += loss.item()
            total_acc += acc.item()
            num_batches += 1
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{acc.item():.4f}',
                'temp': f'{temperature.item():.2f}'
            })
        
        return {
            'loss': total_loss / num_batches,
            'acc': total_acc / num_batches
        }
    
    @torch.no_grad()
    def evaluate(self, split="val") -> Dict:
        """Evaluate on validation or test set"""
        self.model.eval()
        
        if split == "val":
            image_embs = self.val_image_embs
            text_embs = self.val_text_embs
            is_ood = None
        else:
            image_embs = self.test_image_embs
            text_embs = self.test_text_embs
            is_ood = self.test_is_ood
        
        # Project image embeddings
        projected = self.model(image_embs)
        
        # Compute similarity
        similarity = torch.matmul(projected, text_embs.t())
        
        # Compute retrieval metrics
        num_samples = similarity.size(0)
        ranks = []
        
        for i in range(num_samples):
            scores = similarity[i]
            rank = (scores > scores[i]).sum().item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        metrics = {
            'recall@1': (ranks == 1).mean() * 100,
            'recall@5': (ranks <= 5).mean() * 100,
            'recall@10': (ranks <= 10).mean() * 100,
            'mean_rank': ranks.mean()
        }
        
        # Separate in-dist and OOD metrics
        if is_ood is not None:
            is_ood_array = np.array(is_ood)
            in_dist_mask = ~is_ood_array
            
            if in_dist_mask.sum() > 0:
                in_dist_ranks = ranks[in_dist_mask]
                metrics['in_dist_recall@1'] = (in_dist_ranks == 1).mean() * 100
                metrics['in_dist_recall@5'] = (in_dist_ranks <= 5).mean() * 100
            
            if is_ood_array.sum() > 0:
                ood_ranks = ranks[is_ood_array]
                metrics['ood_recall@1'] = (ood_ranks == 1).mean() * 100
                metrics['ood_recall@5'] = (ood_ranks <= 5).mean() * 100
        
        # Compute loss
        loss, acc = 0, 0
        if split == "val":
            loader = self.val_loader
        else:
            loader = self.test_loader
        
        for batch in loader:
            image_emb = batch['image_emb'].to(self.device)
            text_emb = batch['text_emb'].to(self.device)
            projected = self.model(image_emb)
            temperature = self.model.get_temperature()
            batch_loss, batch_acc = info_nce_loss(projected, text_emb, temperature)
            loss += batch_loss.item()
            acc += batch_acc.item()
        
        metrics['loss'] = loss / len(loader)
        metrics['acc'] = acc / len(loader)
        
        return metrics
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Training Linear Probe")
        print(f"{'='*60}")
        
        best_val_recall = 0
        start_time = time.time()
        
        for epoch in range(self.config.model.probe_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.model.probe_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate("val")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['train_acc'].append(train_metrics['acc'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_acc'].append(val_metrics['acc'])
            self.history['val_recall@1'].append(val_metrics['recall@1'])
            self.history['val_recall@5'].append(val_metrics['recall@5'])
            
            # Print metrics
            print(f"Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['acc']:.4f}")
            print(f"Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['acc']:.4f}")
            print(f"Val   - R@1: {val_metrics['recall@1']:.2f}%, R@5: {val_metrics['recall@5']:.2f}%")
            
            # Save best model
            if val_metrics['recall@1'] > best_val_recall:
                best_val_recall = val_metrics['recall@1']
                checkpoint_path = os.path.join(
                    self.config.experiment.checkpoint_dir,
                    'best_linear_probe.pt'
                )
                torch.save({
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': val_metrics
                }, checkpoint_path)
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
        
        # Evaluate on test set
        test_metrics = self.evaluate("test")
        
        print(f"\nTest Results:")
        print(f"  Recall@1:  {test_metrics['recall@1']:.2f}%")
        print(f"  Recall@5:  {test_metrics['recall@5']:.2f}%")
        print(f"  Recall@10: {test_metrics['recall@10']:.2f}%")
        
        if 'in_dist_recall@1' in test_metrics:
            print(f"  In-dist R@1: {test_metrics['in_dist_recall@1']:.2f}%")
            print(f"  OOD R@1:     {test_metrics['ood_recall@1']:.2f}%")
        
        return {
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'history': self.history,
            'training_time': elapsed
        }

def run_linear_probe(config, seed):
    """Run linear probe with specified seed"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'#'*60}")
    print(f"Running Linear Probe with seed {seed}")
    print(f"{'#'*60}")
    
    # Load embeddings from CLIP baseline
    embeddings_path = os.path.join(
        config.experiment.output_dir,
        f'clip_embeddings_seed{seed}.pt'
    )
    
    if not os.path.exists(embeddings_path):
        raise FileNotFoundError(
            f"CLIP embeddings not found at {embeddings_path}. "
            "Run CLIP baseline first."
        )
    
    embs = torch.load(embeddings_path)
    
    # For training, we need to encode training set
    # Load training embeddings (need to generate these)
    train_embs_path = os.path.join(
        config.experiment.output_dir,
        f'train_embeddings_seed{seed}.pt'
    )
    
    if not os.path.exists(train_embs_path):
        print("Training embeddings not found. Need to generate from train_loader.")
        return None
    
    train_embs = torch.load(train_embs_path)
    
    # Prepare embeddings
    val_embs = {
        'image': embs['val_image_emb'],
        'text': embs['val_text_emb']
    }
    test_embs = {
        'image': embs['test_image_emb'],
        'text': embs['test_text_emb'],
        'is_ood': embs['test_is_ood']
    }
    
    # Train linear probe
    trainer = LinearProbeTrainer(config, train_embs, val_embs, test_embs)
    results = trainer.train()
    
    # Save results
    results['seed'] = seed
    results_path = os.path.join(
        config.experiment.output_dir,
        f'linear_probe_results_seed{seed}.json'
    )
    
    with open(results_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_results = {
            k: (v.tolist() if isinstance(v, np.ndarray) else 
                float(v) if isinstance(v, (np.floating, torch.Tensor)) else v)
            for k, v in results.items() if k != 'history'
        }
        json.dump(json_results, f, indent=2)
    
    return results
