"""
Train the Compressed Encoder-Decoder Transformer
==================================================

Architecture: Perceiver-style compression + Encoder-Decoder

  2,361 features → Embed → Compress to 64 groups → Encoder (self-attention)
  → Decoder (cross-attention) → Classification → 131 diseases

This script:
  1. Loads the processed genomic data
  2. Normalizes numerical features (prevents NaN from extreme values)
  3. Creates the Compressed Encoder-Decoder model
  4. Trains end-to-end with warmup + cosine schedule
  5. Evaluates on validation and test sets
  6. Saves results to JSON

Usage:
  python Complete_Architecture/train.py
"""

import sys
import json
import time
import math
from pathlib import Path
from datetime import datetime

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, f1_score, classification_report
from tqdm import tqdm
import numpy as np
import pandas as pd

# Add parent directory for imports
ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

import config
from utils.dataset import GenomicDataset
from Complete_Architecture.model import CompressedEncoderDecoder, GenomicClassifier


class EncoderDecoderTrainer:
    """
    Trainer for the Compressed Encoder-Decoder Transformer.
    Trains the ENTIRE model end-to-end: 
      embedding + compression + encoder + decoder + classifier
    """
    
    def __init__(self, model, train_loader, val_loader, num_classes, 
                 lr=3e-4, weight_decay=0.01, warmup_epochs=3, 
                 total_epochs=30, device='cpu'):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.num_classes = num_classes
        
        # Tracking
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        self.history = {
            'train_loss': [], 'train_acc': [], 
            'val_loss': [], 'val_acc': [], 'val_f1': []
        }
        
        # Optimizer (trains ALL parameters end-to-end)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(0.9, 0.999),
            eps=1e-8
        )
        
        # Learning rate schedule: linear warmup + cosine decay
        num_batches = len(train_loader)
        self.warmup_steps = warmup_epochs * num_batches
        self.total_steps = total_epochs * num_batches
        self.current_step = 0
        
        def lr_lambda(step):
            if step < self.warmup_steps:
                # Linear warmup: 0 → 1
                return float(step) / float(max(1, self.warmup_steps))
            # Cosine decay: 1 → 0.05
            progress = float(step - self.warmup_steps) / float(
                max(1, self.total_steps - self.warmup_steps)
            )
            return max(0.05, 0.5 * (1.0 + math.cos(math.pi * progress)))
        
        self.scheduler = torch.optim.lr_scheduler.LambdaLR(
            self.optimizer, lr_lambda
        )
        
        # Loss function with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        
        # Count and display parameters
        total_params = sum(p.numel() for p in model.parameters())
        trainable_params = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        print(f"  Total parameters:     {total_params:,} ({total_params/1e6:.1f}M)")
        print(f"  Trainable parameters: {trainable_params:,} ({trainable_params/1e6:.1f}M)")
        print(f"  Learning rate: {lr}")
        print(f"  Warmup: {warmup_epochs} epochs ({self.warmup_steps} steps)")
        print(f"  Schedule: cosine decay to 5% of peak LR")
    
    def train_epoch(self):
        """Train one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        for batch in tqdm(self.train_loader, desc="  Training", leave=False):
            cat_data, num_data, labels = batch
            
            # Move to device
            cat_data = {k: v.to(self.device) for k, v in cat_data.items()}
            if num_data is not None:
                num_data = num_data.to(self.device)
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(cat_data, num_data)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Gradient clipping (prevents explosion)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            
            self.optimizer.step()
            self.scheduler.step()
            self.current_step += 1
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        avg_loss = total_loss / len(self.train_loader)
        return avg_loss, accuracy
    
    def validate(self):
        """Validate the model."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc="  Validating", leave=False):
                cat_data, num_data, labels = batch
                
                cat_data = {k: v.to(self.device) for k, v in cat_data.items()}
                if num_data is not None:
                    num_data = num_data.to(self.device)
                labels = labels.to(self.device)
                
                logits = self.model(cat_data, num_data)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        avg_loss = total_loss / len(self.val_loader)
        
        return avg_loss, accuracy, f1_macro, f1_weighted
    
    def train(self, num_epochs, save_path=None):
        """Train for multiple epochs."""
        print(f"\n{'='*60}")
        print(f"  TRAINING COMPRESSED ENCODER-DECODER")
        print(f"  Epochs: {num_epochs}")
        print(f"  Device: {self.device}")
        if torch.cuda.is_available():
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"{'='*60}\n")
        
        start_time = time.time()
        patience = 8  # early stopping patience
        no_improve = 0
        
        for epoch in range(num_epochs):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            val_loss, val_acc, val_f1_macro, val_f1_weighted = self.validate()
            
            # Track history
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1_macro)
            
            # Print results
            epoch_time = time.time() - epoch_start
            current_lr = self.optimizer.param_groups[0]['lr']
            print(f"  Epoch {epoch+1}/{num_epochs} ({epoch_time:.0f}s) | LR: {current_lr:.6f}")
            print(f"    Train  | Loss: {train_loss:.4f} | Acc: {train_acc*100:.2f}%")
            print(f"    Val    | Loss: {val_loss:.4f} | Acc: {val_acc*100:.2f}% | "
                  f"F1-macro: {val_f1_macro:.4f} | F1-weighted: {val_f1_weighted:.4f}")
            
            # Save best model
            if val_acc > self.best_val_acc:
                self.best_val_acc = val_acc
                self.best_val_f1 = val_f1_macro
                no_improve = 0
                if save_path:
                    torch.save(self.model.state_dict(), save_path)
                    print(f"    ✅ New best! Saved to {save_path}")
            else:
                no_improve += 1
                if no_improve >= patience:
                    print(f"\n  ⚠️ Early stopping: no improvement for {patience} epochs")
                    break
            
            print()
        
        total_time = time.time() - start_time
        print(f"{'='*60}")
        print(f"  Training complete in {total_time:.0f}s ({total_time/60:.1f} min)")
        print(f"  Best validation accuracy: {self.best_val_acc*100:.2f}%")
        print(f"  Best validation Macro F1: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")
        
        return total_time


def evaluate_on_test(model, test_loader, device):
    """Evaluate model on test set with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="  Testing", leave=False):
            cat_data, num_data, labels = batch
            
            cat_data = {k: v.to(device) for k, v in cat_data.items()}
            if num_data is not None:
                num_data = num_data.to(device)
            labels = labels.to(device)
            
            logits = model(cat_data, num_data)
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    test_acc = accuracy_score(all_labels, all_preds)
    test_f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    test_f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    
    return test_acc, test_f1_macro, test_f1_weighted, all_preds, all_labels


def main():
    config.set_seed()
    
    print("\n" + "="*60)
    print("  COMPRESSED ENCODER-DECODER TRANSFORMER")
    print("  Perceiver-style Feature Grouping + Encoder-Decoder")
    print("  Genomic Disease Classification")
    print("="*60 + "\n")
    
    # ================================================================
    # Device setup
    # ================================================================
    if torch.cuda.is_available():
        device = torch.device('cuda')
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        if torch.cuda.device_count() > 1:
            print(f"  Available GPUs: {torch.cuda.device_count()}")
    else:
        device = torch.device('cpu')
    print(f"  Device: {device}\n")
    
    # ================================================================
    # Load data
    # ================================================================
    print("  Loading data...")
    train_df = pd.read_csv(config.PROCESSED_DIR / 'train.csv')
    val_df = pd.read_csv(config.PROCESSED_DIR / 'val.csv')
    test_df = pd.read_csv(config.PROCESSED_DIR / 'test.csv')
    
    with open(config.PROCESSED_DIR / 'metadata.json', 'r') as f:
        metadata = json.load(f)
    
    print(f"  Train: {len(train_df)} samples")
    print(f"  Val:   {len(val_df)} samples")
    print(f"  Test:  {len(test_df)} samples")
    print(f"  Features: {len(metadata['categorical_features'])} categorical + "
          f"{len(metadata['numerical_features'])} numerical = "
          f"{len(metadata['categorical_features']) + len(metadata['numerical_features'])} total")
    
    num_classes = metadata['vocab_sizes'][metadata['target_column']]
    print(f"  Classes: {num_classes}")
    
    # ================================================================
    # Normalize numerical features
    # CRITICAL: Some species abundance values are >100,000 which
    # causes overflow in attention softmax → NaN loss.
    # Standardize to zero mean, unit variance, then clip outliers.
    # ================================================================
    num_feats = metadata['numerical_features']
    num_cols_in_data = [f for f in num_feats if f in train_df.columns]
    
    if num_cols_in_data:
        # Compute stats from TRAINING data only (no data leakage)
        train_mean = train_df[num_cols_in_data].mean()
        train_std = train_df[num_cols_in_data].std().replace(0, 1)
        
        # Apply to all splits
        for df in [train_df, val_df, test_df]:
            df[num_cols_in_data] = (df[num_cols_in_data] - train_mean) / train_std
            df[num_cols_in_data] = df[num_cols_in_data].clip(-10, 10)
        
        print(f"  ✅ Normalized {len(num_cols_in_data)} numerical features "
              f"(standardized + clipped to [-10, 10])")
    
    # ================================================================
    # Create datasets and dataloaders
    # ================================================================
    train_dataset = GenomicDataset(
        train_df, metadata['categorical_features'],
        metadata['numerical_features'], metadata['target_column']
    )
    val_dataset = GenomicDataset(
        val_df, metadata['categorical_features'],
        metadata['numerical_features'], metadata['target_column']
    )
    test_dataset = GenomicDataset(
        test_df, metadata['categorical_features'],
        metadata['numerical_features'], metadata['target_column']
    )
    
    # Batch size 64 — feasible now with only 64 tokens (was 2362 before)
    batch_size = 64
    
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=0, pin_memory=True, drop_last=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True
    )
    
    print(f"  Batch size: {batch_size}")
    print(f"  Training batches: {len(train_loader)}")
    
    # ================================================================
    # Create model
    # ================================================================
    print("\n  Creating Compressed Encoder-Decoder model...")
    
    # Hyperparameters
    HIDDEN_DIM = 256
    NUM_GROUP_TOKENS = 64       # compress 2361 features → 64 groups
    NUM_ENCODER_LAYERS = 4      # self-attention on 64 tokens
    NUM_DECODER_LAYERS = 2      # cross-attention to encoder
    NUM_HEADS = 8
    FFN_DIM = 1024
    DROPOUT = 0.15
    LR = 5e-4
    WEIGHT_DECAY = 0.01
    WARMUP_EPOCHS = 2
    NUM_EPOCHS = 30
    
    encoder_decoder = CompressedEncoderDecoder(
        categorical_vocab_sizes=metadata['vocab_sizes'],
        categorical_features=metadata['categorical_features'],
        numerical_features=metadata['numerical_features'],
        hidden_dim=HIDDEN_DIM,
        num_chunks=NUM_GROUP_TOKENS,
        num_encoder_layers=NUM_ENCODER_LAYERS,
        num_decoder_layers=NUM_DECODER_LAYERS,
        num_attention_heads=NUM_HEADS,
        feedforward_dim=FFN_DIM,
        dropout=DROPOUT
    )
    
    model = GenomicClassifier(
        encoder_decoder=encoder_decoder,
        num_classes=num_classes,
        hidden_dim=HIDDEN_DIM,
        dropout=DROPOUT
    )
    
    # Multi-GPU support
    if torch.cuda.device_count() > 1:
        print(f"  Using {torch.cuda.device_count()} GPUs with DataParallel")
        model = nn.DataParallel(model)
    
    # ================================================================
    # Quick forward pass test
    # ================================================================
    print("\n  Testing forward pass...")
    try:
        sample_batch = next(iter(train_loader))
        cat_data, num_data, labels = sample_batch
        cat_data_dev = {k: v.to(device) for k, v in cat_data.items()}
        num_data_dev = num_data.to(device) if num_data is not None else None
        
        model_test = model.to(device)
        with torch.no_grad():
            output = model_test(cat_data_dev, num_data_dev)
        print(f"  ✅ Forward pass OK! Output shape: {output.shape}")
        del model_test, output, cat_data_dev, num_data_dev
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
    except Exception as e:
        print(f"  ❌ Forward pass FAILED: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # ================================================================
    # Train
    # ================================================================
    save_path = config.MODEL_DIR / 'classifier_compressed_encoder_decoder.pt'
    
    trainer = EncoderDecoderTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        lr=LR,
        weight_decay=WEIGHT_DECAY,
        warmup_epochs=WARMUP_EPOCHS,
        total_epochs=NUM_EPOCHS,
        device=device
    )
    
    training_time = trainer.train(
        num_epochs=NUM_EPOCHS,
        save_path=save_path
    )
    
    # ================================================================
    # Evaluate on test set
    # ================================================================
    print("  Loading best model for test evaluation...")
    model.load_state_dict(torch.load(save_path, map_location=device))
    model = model.to(device)
    
    test_acc, test_f1_macro, test_f1_weighted, test_preds, test_labels = \
        evaluate_on_test(model, test_loader, device)
    
    print(f"\n{'='*60}")
    print(f"  TEST RESULTS")
    print(f"{'='*60}")
    print(f"  Test Accuracy:    {test_acc*100:.2f}%")
    print(f"  Test Macro F1:    {test_f1_macro:.4f}")
    print(f"  Test Weighted F1: {test_f1_weighted:.4f}")
    print(f"{'='*60}\n")
    
    # ================================================================
    # Save results
    # ================================================================
    results = {
        'model': 'Compressed Encoder-Decoder',
        'architecture': {
            'type': 'Perceiver-style compression + Encoder-Decoder',
            'num_group_tokens': NUM_GROUP_TOKENS,
            'encoder_layers': NUM_ENCODER_LAYERS,
            'decoder_layers': NUM_DECODER_LAYERS,
            'attention_heads': NUM_HEADS,
            'hidden_dim': HIDDEN_DIM,
            'feedforward_dim': FFN_DIM,
            'dropout': DROPOUT,
            'num_features': len(metadata['categorical_features']) + len(metadata['numerical_features']),
            'num_classes': num_classes,
            'total_params': sum(p.numel() for p in model.parameters()),
        },
        'training': {
            'epochs_run': len(trainer.history['train_loss']),
            'max_epochs': NUM_EPOCHS,
            'batch_size': batch_size,
            'learning_rate': LR,
            'weight_decay': WEIGHT_DECAY,
            'warmup_epochs': WARMUP_EPOCHS,
            'label_smoothing': 0.1,
            'training_time_seconds': training_time,
            'training_time_minutes': training_time / 60,
        },
        'results': {
            'best_val_acc': trainer.best_val_acc,
            'best_val_f1_macro': trainer.best_val_f1,
            'test_acc': test_acc,
            'test_f1_macro': test_f1_macro,
            'test_f1_weighted': test_f1_weighted,
        },
        'history': trainer.history,
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }
    
    results_path = config.RESULTS_DIR / 'compressed_encoder_decoder_results.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"  Results saved to: {results_path}")
    
    # Print per-class summary for top classes
    print(f"\n  Per-class accuracy (top 20 most common):")
    from collections import Counter
    label_counts = Counter(test_labels)
    top_classes = label_counts.most_common(20)
    for cls, count in top_classes:
        cls_preds = [test_preds[i] for i, l in enumerate(test_labels) if l == cls]
        cls_acc = sum(1 for p in cls_preds if p == cls) / len(cls_preds)
        print(f"    Class {cls:3d} ({count:4d} samples): {cls_acc*100:.1f}%")


if __name__ == "__main__":
    main()
