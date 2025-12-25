"""
CLIP Zero-shot baseline for image-text retrieval
Time budget: 45-60 minutes (includes encoding + evaluation with 3 seeds)
"""

import time
import torch
import torch.nn.functional as F
import numpy as np
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from typing import Dict, Tuple
import json
import os

class CLIPRetrieval:
    """CLIP-based zero-shot retrieval system"""
    
    def __init__(self, config):
        self.config = config
        self.device = config.device
        
        print("Loading CLIP model...")
        self.model = CLIPModel.from_pretrained(
            config.model.clip_model,
            cache_dir=config.data.cache_dir
        ).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(
            config.model.clip_model,
            cache_dir=config.data.cache_dir
        )
        self.model.eval()
        
        # Storage for embeddings
        self.image_embeddings = None
        self.text_embeddings = None
        self.image_ids = None
    
    @torch.no_grad()
    def encode_images(self, dataloader, split_name="train") -> Tuple[torch.Tensor, list]:
        """
        Encode all images in a dataloader
        Returns: (embeddings, image_ids)
        """
        print(f"\nEncoding images for {split_name} split...")
        
        all_embeddings = []
        all_ids = []
        
        start_time = time.time()
        num_images = 0
        
        for batch in tqdm(dataloader, desc=f"Encoding {split_name} images"):
            images = batch['image'].to(self.device)
            image_ids = batch['image_id']
            
            # Encode images
            image_features = self.model.get_image_features(pixel_values=images)
            image_features = F.normalize(image_features, p=2, dim=-1)
            
            all_embeddings.append(image_features.cpu())
            all_ids.extend(image_ids)
            num_images += images.size(0)
        
        elapsed = time.time() - start_time
        throughput = num_images / elapsed
        
        embeddings = torch.cat(all_embeddings, dim=0)
        
        print(f"  Encoded {num_images} images in {elapsed:.2f}s")
        print(f"  Throughput: {throughput:.2f} images/sec")
        print(f"  GPU Memory: {torch.cuda.max_memory_allocated()/1e9:.2f} GB")
        
        return embeddings, all_ids
    
    @torch.no_grad()
    def encode_texts(self, texts: list, batch_size: int = None) -> torch.Tensor:
        """Encode text captions"""
        if batch_size is None:
            batch_size = self.config.hardware.get_clip_batch_size()
        
        print(f"\nEncoding {len(texts)} text captions...")
        
        all_embeddings = []
        
        for i in tqdm(range(0, len(texts), batch_size), desc="Encoding texts"):
            batch_texts = texts[i:i+batch_size]
            
            # Tokenize and encode
            inputs = self.processor(
                text=batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)
            
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, p=2, dim=-1)
            
            all_embeddings.append(text_features.cpu())
        
        embeddings = torch.cat(all_embeddings, dim=0)
        return embeddings
    
    def compute_similarity(self, image_embeddings: torch.Tensor, 
                          text_embeddings: torch.Tensor) -> torch.Tensor:
        """Compute cosine similarity between image and text embeddings"""
        # Normalize if not already
        image_embeddings = F.normalize(image_embeddings, p=2, dim=-1)
        text_embeddings = F.normalize(text_embeddings, p=2, dim=-1)
        
        # Compute similarity matrix (num_images x num_texts)
        similarity = torch.matmul(image_embeddings, text_embeddings.t())
        return similarity
    
    def compute_retrieval_metrics(self, similarity_matrix: torch.Tensor, 
                                  is_ood_mask: torch.Tensor = None) -> Dict:
        """
        Compute Recall@K metrics for image-to-text retrieval
        Assumes diagonal elements are correct matches
        """
        num_samples = similarity_matrix.size(0)
        
        # Get ranking of correct caption for each image
        # Correct caption is at position i for image i
        correct_indices = torch.arange(num_samples)
        
        # For each image, rank all captions
        ranks = []
        for i in range(num_samples):
            # Get similarity scores for this image
            scores = similarity_matrix[i]
            # Get rank of correct caption (how many captions score higher)
            rank = (scores > scores[i]).sum().item() + 1
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        # Compute Recall@K
        metrics = {
            'recall@1': (ranks == 1).mean() * 100,
            'recall@5': (ranks <= 5).mean() * 100,
            'recall@10': (ranks <= 10).mean() * 100,
            'mean_rank': ranks.mean(),
            'median_rank': np.median(ranks)
        }
        
        # Separate metrics for in-distribution vs OOD
        if is_ood_mask is not None:
            in_dist_mask = ~is_ood_mask
            if in_dist_mask.sum() > 0:
                in_dist_ranks = ranks[in_dist_mask.cpu().numpy()]
                metrics['in_dist_recall@1'] = (in_dist_ranks == 1).mean() * 100
                metrics['in_dist_recall@5'] = (in_dist_ranks <= 5).mean() * 100
                metrics['in_dist_recall@10'] = (in_dist_ranks <= 10).mean() * 100
            
            if is_ood_mask.sum() > 0:
                ood_ranks = ranks[is_ood_mask.cpu().numpy()]
                metrics['ood_recall@1'] = (ood_ranks == 1).mean() * 100
                metrics['ood_recall@5'] = (ood_ranks <= 5).mean() * 100
                metrics['ood_recall@10'] = (ood_ranks <= 10).mean() * 100
        
        return metrics
    
    def evaluate(self, dataloader, split_name="test") -> Dict:
        """
        Full evaluation pipeline: encode and compute metrics
        """
        print(f"\n{'='*60}")
        print(f"CLIP Zero-Shot Evaluation on {split_name}")
        print(f"{'='*60}")
        
        # Encode images
        image_embeddings, image_ids = self.encode_images(dataloader, split_name)
        
        # Collect captions and OOD labels
        captions = []
        is_ood = []
        for batch in dataloader:
            captions.extend(batch['caption'])
            is_ood.extend(batch['is_ood'])
        
        # Encode captions
        text_embeddings = self.encode_texts(captions)
        
        # Compute similarity
        print("\nComputing similarity matrix...")
        similarity = self.compute_similarity(image_embeddings, text_embeddings)
        
        # Compute metrics
        is_ood_tensor = torch.tensor(is_ood)
        metrics = self.compute_retrieval_metrics(similarity, is_ood_tensor)
        
        # Print results
        print(f"\nResults on {split_name}:")
        print(f"  Recall@1:  {metrics['recall@1']:.2f}%")
        print(f"  Recall@5:  {metrics['recall@5']:.2f}%")
        print(f"  Recall@10: {metrics['recall@10']:.2f}%")
        print(f"  Mean Rank: {metrics['mean_rank']:.2f}")
        
        if 'in_dist_recall@1' in metrics:
            print(f"\n  In-distribution:")
            print(f"    Recall@1:  {metrics['in_dist_recall@1']:.2f}%")
            print(f"    Recall@5:  {metrics['in_dist_recall@5']:.2f}%")
            print(f"  OOD:")
            print(f"    Recall@1:  {metrics['ood_recall@1']:.2f}%")
            print(f"    Recall@5:  {metrics['ood_recall@5']:.2f}%")
        
        return {
            'metrics': metrics,
            'similarity': similarity,
            'image_embeddings': image_embeddings,
            'text_embeddings': text_embeddings,
            'captions': captions,
            'is_ood': is_ood
        }

def run_clip_baseline(config, train_loader, val_loader, test_loader, seed):
    """
    Run CLIP zero-shot baseline with specified seed
    """
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    print(f"\n{'#'*60}")
    print(f"Running CLIP Baseline with seed {seed}")
    print(f"{'#'*60}")
    
    # Initialize retrieval system
    clip_retrieval = CLIPRetrieval(config)
    
    # Evaluate on validation set
    val_results = clip_retrieval.evaluate(val_loader, "validation")
    
    # Evaluate on test set (includes OOD)
    test_results = clip_retrieval.evaluate(test_loader, "test")
    
    # Save embeddings for linear probe
    embeddings_path = os.path.join(config.experiment.output_dir, f'clip_embeddings_seed{seed}.pt')
    torch.save({
        'val_image_emb': val_results['image_embeddings'],
        'val_text_emb': val_results['text_embeddings'],
        'test_image_emb': test_results['image_embeddings'],
        'test_text_emb': test_results['text_embeddings'],
        'val_captions': val_results['captions'],
        'test_captions': test_results['captions'],
        'test_is_ood': test_results['is_ood']
    }, embeddings_path)
    
    print(f"\nSaved embeddings to {embeddings_path}")
    
    return {
        'val_metrics': val_results['metrics'],
        'test_metrics': test_results['metrics'],
        'seed': seed
    }
