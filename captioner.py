"""
Small captioning model: Frozen CLIP encoder + GPT-2 decoder with LoRA option
Time budget: 1.5-2 hours (with 3 seeds and LoRA comparison)
"""

import time
import torch
import torch.nn as nn
from transformers import GPT2LMHeadModel, GPT2Tokenizer, GPT2Config
from transformers import CLIPModel
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import numpy as np
from typing import Dict, List
import os
import json

class LoRALayer(nn.Module):
    """Low-Rank Adaptation layer"""
    
    def __init__(self, in_features: int, out_features: int, rank: int = 8, 
                 alpha: int = 16, dropout: float = 0.1):
        super().__init__()
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank
        
        # LoRA matrices
        self.lora_A = nn.Parameter(torch.randn(in_features, rank) * 0.01)
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x):
        # x: (batch, seq_len, in_features)
        lora_out = self.dropout(x) @ self.lora_A @ self.lora_B
        return lora_out * self.scaling

class ImageCaptioningModel(nn.Module):
    """Image captioning with frozen CLIP + GPT-2 decoder"""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Frozen CLIP encoder
        self.clip = CLIPModel.from_pretrained(
            config.model.clip_model,
            cache_dir=config.data.cache_dir
        )
        # Freeze CLIP
        for param in self.clip.parameters():
            param.requires_grad = False
        
        # GPT-2 decoder with hardware-specific configuration
        if config.hardware.device_type == "rtx5000_ada":
            # Larger model for RTX 5000 Ada (32GB VRAM)
            gpt_config = GPT2Config(
                vocab_size=50257,
                n_positions=512,
                n_embd=768,
                n_layer=6,
                n_head=12,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
        elif config.hardware.device_type == "rtx3070":
            # Smaller config for 8GB GPU
            gpt_config = GPT2Config(
                vocab_size=50257,
                n_positions=512,
                n_embd=384,
                n_layer=2,
                n_head=6,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
        else:
            # Default config for A5000
            gpt_config = GPT2Config(
                vocab_size=50257,
                n_positions=512,
                n_embd=512,
                n_layer=4,
                n_head=8,
                resid_pdrop=0.1,
                embd_pdrop=0.1,
                attn_pdrop=0.1,
            )
        
        self.decoder = GPT2LMHeadModel(gpt_config)
        
        # Projection from CLIP to GPT-2
        self.projection = nn.Linear(
            config.model.clip_embed_dim,
            gpt_config.n_embd
        )
        
        # LoRA layers on cross-attention (if enabled)
        self.use_lora = config.model.use_lora
        if self.use_lora:
            self.lora_layers = nn.ModuleList([
                LoRALayer(
                    gpt_config.n_embd,
                    gpt_config.n_embd,
                    rank=config.model.lora_r,
                    alpha=config.model.lora_alpha,
                    dropout=config.model.lora_dropout
                )
                for _ in range(gpt_config.n_layer)
            ])
        
        # Tokenizer
        self.tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        self.tokenizer.pad_token = self.tokenizer.eos_token
    
    def encode_image(self, pixel_values):
        """Encode images with frozen CLIP"""
        with torch.no_grad():
            image_features = self.clip.get_image_features(pixel_values=pixel_values)
            image_features = F.normalize(image_features, p=2, dim=-1)
        return image_features
    
    def forward(self, pixel_values, input_ids, attention_mask=None, labels=None):
        """
        Forward pass
        pixel_values: (batch, 3, H, W)
        input_ids: (batch, seq_len)
        """
        batch_size = pixel_values.size(0)
        
        # Encode images
        image_features = self.encode_image(pixel_values)  # (batch, clip_dim)
        
        # Project to decoder dimension
        image_embeds = self.projection(image_features)  # (batch, gpt_dim)
        image_embeds = image_embeds.unsqueeze(1)  # (batch, 1, gpt_dim)
        
        # Get text embeddings
        text_embeds = self.decoder.transformer.wte(input_ids)  # (batch, seq_len, gpt_dim)
        
        # Concatenate image and text embeddings
        inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
        
        # Adjust attention mask
        if attention_mask is not None:
            # Add attention for image token
            image_attention = torch.ones(
                batch_size, 1, 
                dtype=attention_mask.dtype, 
                device=attention_mask.device
            )
            attention_mask = torch.cat([image_attention, attention_mask], dim=1)
        
        # Forward through decoder
        outputs = self.decoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            labels=labels if labels is not None else None,
            use_cache=False
        )
        
        return outputs
    
    @torch.no_grad()
    def generate(self, pixel_values, max_length=32, temperature=1.0, top_k=50):
        """Generate captions for images"""
        batch_size = pixel_values.size(0)
        
        # Encode images
        image_features = self.encode_image(pixel_values)
        image_embeds = self.projection(image_features).unsqueeze(1)
        
        # Start with BOS token
        generated = torch.full(
            (batch_size, 1),
            self.tokenizer.bos_token_id,
            dtype=torch.long,
            device=pixel_values.device
        )
        
        for _ in range(max_length):
            # Get text embeddings
            text_embeds = self.decoder.transformer.wte(generated)
            
            # Concatenate with image embeddings
            inputs_embeds = torch.cat([image_embeds, text_embeds], dim=1)
            
            # Forward pass
            outputs = self.decoder(inputs_embeds=inputs_embeds)
            logits = outputs.logits[:, -1, :]  # Get last token logits
            
            # Apply temperature
            logits = logits / temperature
            
            # Top-k sampling
            if top_k > 0:
                indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
                logits[indices_to_remove] = float('-inf')
            
            # Sample next token
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
            
            # Stop if all sequences have generated EOS
            if (next_token == self.tokenizer.eos_token_id).all():
                break
        
        return generated
    
    def count_parameters(self):
        """Count trainable parameters"""
        total = sum(p.numel() for p in self.parameters())
        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        return {'total': total, 'trainable': trainable}

class CaptionDataset(Dataset):
    """Dataset for image captioning"""
    
    def __init__(self, data_list, tokenizer, max_length=32):
        self.data = data_list
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Tokenize caption
        caption = item['caption']
        encoding = self.tokenizer(
            caption,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        
        return {
            'image': item['image'],
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'caption': caption,
            'is_ood': item.get('is_ood', False)
        }

class CaptionerTrainer:
    """Trainer for image captioning model"""
    
    def __init__(self, config, train_data, val_data, test_data):
        self.config = config
        self.device = config.device
        
        # Initialize model
        self.model = ImageCaptioningModel(config).to(self.device)
        
        # Count parameters
        param_count = self.model.count_parameters()
        print(f"\nModel parameters:")
        print(f"  Total: {param_count['total']:,}")
        print(f"  Trainable: {param_count['trainable']:,}")
        
        # Create datasets
        tokenizer = self.model.tokenizer
        train_dataset = CaptionDataset(
            train_data, tokenizer, config.model.max_caption_length
        )
        val_dataset = CaptionDataset(
            val_data, tokenizer, config.model.max_caption_length
        )
        test_dataset = CaptionDataset(
            test_data, tokenizer, config.model.max_caption_length
        )
        
        # Create dataloaders
        batch_size = config.hardware.get_captioner_bs()
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True
        )
        self.val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True
        )
        
        # Optimizer and scheduler
        self.optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.model.parameters()),
            lr=config.model.captioner_lr,
            weight_decay=config.model.weight_decay
        )
        
        total_steps = len(self.train_loader) * config.model.captioner_epochs
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=total_steps
        )
        
        # AMP scaler
        self.scaler = torch.cuda.amp.GradScaler() if config.model.use_amp else None
        
        # Gradient checkpointing
        if config.hardware.use_gradient_checkpointing():
            self.model.decoder.gradient_checkpointing_enable()
        
        # Training history
        self.history = {
            'train_loss': [], 'val_loss': [], 'val_perplexity': []
        }
    
    def train_epoch(self) -> Dict:
        """Train for one epoch"""
        self.model.train()
        
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc="Training")
        for batch in pbar:
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Create labels (shifted input_ids)
            labels = input_ids.clone()
            labels[labels == self.model.tokenizer.pad_token_id] = -100
            
            # Forward pass with AMP
            if self.scaler is not None:
                with torch.cuda.amp.autocast():
                    outputs = self.model(
                        images, input_ids,
                        attention_mask=attention_mask,
                        labels=labels
                    )
                    loss = outputs.loss
                
                # Backward pass
                self.optimizer.zero_grad()
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model.gradient_clip
                )
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                outputs = self.model(
                    images, input_ids,
                    attention_mask=attention_mask,
                    labels=labels
                )
                loss = outputs.loss
                
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.model.gradient_clip
                )
                self.optimizer.step()
            
            self.scheduler.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        return {'loss': total_loss / num_batches}
    
    @torch.no_grad()
    def evaluate(self, split="val") -> Dict:
        """Evaluate model"""
        self.model.eval()
        
        loader = self.val_loader if split == "val" else self.test_loader
        
        total_loss = 0
        num_batches = 0
        generated_captions = []
        reference_captions = []
        
        for batch in tqdm(loader, desc=f"Evaluating {split}"):
            images = batch['image'].to(self.device)
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            
            # Compute loss
            labels = input_ids.clone()
            labels[labels == self.model.tokenizer.pad_token_id] = -100
            
            outputs = self.model(
                images, input_ids,
                attention_mask=attention_mask,
                labels=labels
            )
            total_loss += outputs.loss.item()
            num_batches += 1
            
            # Generate captions
            generated_ids = self.model.generate(
                images,
                max_length=self.config.model.max_caption_length
            )
            
            # Decode
            for gen_ids, ref_caption in zip(generated_ids, batch['caption']):
                gen_caption = self.model.tokenizer.decode(
                    gen_ids, skip_special_tokens=True
                )
                generated_captions.append(gen_caption)
                reference_captions.append(ref_caption)
        
        avg_loss = total_loss / num_batches
        perplexity = np.exp(avg_loss)
        
        return {
            'loss': avg_loss,
            'perplexity': perplexity,
            'generated_captions': generated_captions,
            'reference_captions': reference_captions
        }
    
    def train(self):
        """Full training loop"""
        print(f"\n{'='*60}")
        print("Training Caption Model")
        print(f"{'='*60}")
        
        best_val_loss = float('inf')
        start_time = time.time()
        
        for epoch in range(self.config.model.captioner_epochs):
            print(f"\nEpoch {epoch+1}/{self.config.model.captioner_epochs}")
            
            # Train
            train_metrics = self.train_epoch()
            
            # Evaluate
            val_metrics = self.evaluate("val")
            
            # Update history
            self.history['train_loss'].append(train_metrics['loss'])
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_perplexity'].append(val_metrics['perplexity'])
            
            # Print metrics
            print(f"Train Loss: {train_metrics['loss']:.4f}")
            print(f"Val Loss: {val_metrics['loss']:.4f}, Perplexity: {val_metrics['perplexity']:.2f}")
            
            # Save best model
            if val_metrics['loss'] < best_val_loss:
                best_val_loss = val_metrics['loss']
                checkpoint_path = os.path.join(
                    self.config.experiment.checkpoint_dir,
                    'best_captioner.pt'
                )
                torch.save({
                    'model_state': self.model.state_dict(),
                    'optimizer_state': self.optimizer.state_dict(),
                    'epoch': epoch,
                    'metrics': val_metrics
                }, checkpoint_path)
        
        elapsed = time.time() - start_time
        print(f"\nTraining completed in {elapsed:.2f}s ({elapsed/60:.2f} min)")
        
        # Final evaluation on test set
        test_metrics = self.evaluate("test")
        
        print(f"\nTest Results:")
        print(f"  Loss: {test_metrics['loss']:.4f}")
        print(f"  Perplexity: {test_metrics['perplexity']:.2f}")
        
        # Show example captions
        print(f"\nExample Generated Captions:")
        for i in range(min(10, len(test_metrics['generated_captions']))):
            print(f"\n{i+1}.")
            print(f"  Reference: {test_metrics['reference_captions'][i]}")
            print(f"  Generated: {test_metrics['generated_captions'][i]}")
        
        return {
            'test_metrics': test_metrics,
            'val_metrics': val_metrics,
            'history': self.history,
            'training_time': elapsed
        }

def run_captioner(config, train_data, val_data, test_data, seed, use_lora=False):
    """Run captioner training with specified seed and LoRA setting"""
    torch.manual_seed(seed)
    np.random.seed(seed)
    
    # Set LoRA config
    config.model.use_lora = use_lora
    
    print(f"\n{'#'*60}")
    print(f"Running Captioner (LoRA={use_lora}) with seed {seed}")
    print(f"{'#'*60}")
    
    trainer = CaptionerTrainer(config, train_data, val_data, test_data)
    results = trainer.train()
    results['seed'] = seed
    results['use_lora'] = use_lora
    
    return results
