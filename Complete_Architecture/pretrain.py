"""
Pre-train Encoder with Masked Chunk Autoencoding
=================================================

Strategy: Masked autoencoding of genomic chunk embeddings
- Randomly mask 30% of the 64 chunk tokens
- Encoder reconstructs the original embeddings
- MSE loss on masked positions only

This teaches the encoder to understand relationships between chunks
without needing disease labels.

Usage:
  python Complete_Architecture/pretrain.py
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add parent directory for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.dataset import GenomicDataset
from Complete_Architecture.model_pretrain import get_pretrain_model


class PreTrainer:
    """Trainer for masked chunk pre-training."""
    
    def __init__(self, model, train_loader, val_loader, 
                 lr=5e-4, weight_decay=0.01, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        
        # Tracking
        self.best_val_loss = float('inf')
        self.history = {
            'train_loss': [], 
            'val_loss': []
        }
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Cosine schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50
        )
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(self.train_loader, desc='Pre-training')
        for batch in pbar:
            cat_data, num_data, labels = batch
            categorical_input = {k: v.to(self.device) for k, v in cat_data.items()}
            numerical_input = num_data.to(self.device) if num_data is not None else None
            
            # Forward pass
            predictions, targets, mask = self.model(categorical_input, numerical_input)
            
            # Compute loss on masked positions only
            raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
            loss = raw_model.compute_loss(predictions, targets, mask)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self):
        """Validate on val set."""
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                cat_data, num_data, labels = batch
                categorical_input = {k: v.to(self.device) for k, v in cat_data.items()}
                numerical_input = num_data.to(self.device) if num_data is not None else None
                
                predictions, targets, mask = self.model(categorical_input, numerical_input)
                raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                loss = raw_model.compute_loss(predictions, targets, mask)
                
                total_loss += loss.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def train(self, num_epochs=50, patience=10, checkpoint_dir='checkpoints'):
        """
        Train the pre-training model.
        
        Args:
            num_epochs: Number of epochs
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  PRE-TRAINING STARTED")
        print(f"{'='*60}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            
            # Validate
            val_loss = self.validate()
            self.history['val_loss'].append(val_loss)
            
            # Step scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                patience_counter = 0
                
                checkpoint_path = checkpoint_dir / 'pretrain_best.pt'
                # Unwrap DataParallel for saving so checkpoint loads cleanly
                raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_loss': val_loss,
                    'history': self.history
                }, checkpoint_path)
                print(f"  ✓ Best model saved (val_loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            print()
        
        print(f"\n{'='*60}")
        print(f"  PRE-TRAINING COMPLETED")
        print(f"{'='*60}")
        print(f"  Best val loss: {self.best_val_loss:.4f}")
        print(f"{'='*60}\n")
        
        return self.history


def main():
    """Main pre-training function."""
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        n_gpus = torch.cuda.device_count()
        print(f"GPUs available: {n_gpus}")
        for i in range(n_gpus):
            print(f"  GPU {i}: {torch.cuda.get_device_name(i)} ({torch.cuda.get_device_properties(i).total_memory / 1e9:.1f} GB)")
    
    # Load metadata
    print("\nLoading metadata...")
    with open('processed_data/metadata.json', 'r') as f:
        metadata = json.load(f)
    
    categorical_features = metadata['categorical_features']
    numerical_features = metadata['numerical_features']
    
    # Load data
    print("Loading data...")
    train_df = pd.read_csv('processed_data/train.csv')
    val_df = pd.read_csv('processed_data/val.csv')
    
    train_dataset = GenomicDataset(
        train_df, categorical_features, numerical_features, 'disease'
    )
    val_dataset = GenomicDataset(
        val_df, categorical_features, numerical_features, 'disease'
    )
    
    # Scale batch size with number of GPUs
    batch_size = 64 * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    print(f"Batch size: {batch_size} ({64} per GPU x {max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)} GPUs)")

    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=0,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    categorical_vocab_sizes = metadata['vocab_sizes']
    
    # Create model
    print("\nCreating model...")
    model = get_pretrain_model(
        categorical_vocab_sizes=categorical_vocab_sizes,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        hidden_dim=256,
        num_chunks=64,
        num_encoder_layers=4,
        num_attention_heads=8,
        feedforward_dim=1024,
        dropout=0.1,
        mask_ratio=0.3
    )
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

    # Wrap with DataParallel if multiple GPUs available
    if torch.cuda.is_available() and torch.cuda.device_count() > 1:
        print(f"\nUsing DataParallel across {torch.cuda.device_count()} GPUs")
        model = nn.DataParallel(model)

    # Create trainer
    trainer = PreTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=5e-4,
        weight_decay=0.01,
        device=device
    )
    
    # Train
    history = trainer.train(
        num_epochs=50,
        patience=10,
        checkpoint_dir='checkpoints'
    )
    
    # Save history
    history_path = Path('results') / f'pretrain_history_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    history_path.parent.mkdir(exist_ok=True)
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    
    print(f"\nHistory saved to {history_path}")


if __name__ == '__main__':
    main()
