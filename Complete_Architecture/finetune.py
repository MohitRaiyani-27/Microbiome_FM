"""
Fine-tune Pre-trained Encoder for Disease Classification
=========================================================

Loads the pre-trained encoder from masked chunk autoencoding
and fine-tunes it for 40-class disease classification.

Uses attention pooling to compress 64 encoder tokens → 1 vector.

Usage:
  python Complete_Architecture/finetune.py
  
  # Or specify pre-trained checkpoint:
  python Complete_Architecture/finetune.py --pretrain checkpoints/pretrain_best.pt
"""

import sys
import json
import time
import argparse
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
from Complete_Architecture.model_finetune import get_finetune_model


class FinetuneTrainer:
    """Trainer for fine-tuning on disease classification."""
    
    def __init__(self, model, train_loader, val_loader, num_classes,
                 lr=1e-4, weight_decay=0.01, device='cpu'):
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
        
        # Optimizer (smaller LR since we're fine-tuning)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay
        )
        
        # Cosine schedule
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=30
        )
        
        # Loss with label smoothing
        self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.train_loader, desc='Training')
        for batch in pbar:
            cat_data, num_data, labels = batch
            categorical_input = {k: v.to(self.device) for k, v in cat_data.items()}
            numerical_input = num_data.to(self.device) if num_data is not None else None
            labels = labels.to(self.device)
            
            # Forward pass
            logits = self.model(categorical_input, numerical_input)
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(self.train_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        
        return avg_loss, accuracy
    
    def validate(self):
        """Validate on val set."""
        self.model.eval()
        total_loss = 0
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for batch in tqdm(self.val_loader, desc='Validating'):
                cat_data, num_data, labels = batch
                categorical_input = {k: v.to(self.device) for k, v in cat_data.items()}
                numerical_input = num_data.to(self.device) if num_data is not None else None
                labels = labels.to(self.device)
                
                logits = self.model(categorical_input, numerical_input)
                loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        avg_loss = total_loss / len(self.val_loader)
        accuracy = accuracy_score(all_labels, all_preds)
        f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
        f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
        
        return avg_loss, accuracy, f1_macro, f1_weighted
    
    def train(self, num_epochs=30, patience=8, checkpoint_dir='checkpoints'):
        """
        Fine-tune the model.
        
        Args:
            num_epochs: Number of epochs
            patience: Early stopping patience
            checkpoint_dir: Directory to save checkpoints
        """
        checkpoint_dir = Path(checkpoint_dir)
        checkpoint_dir.mkdir(exist_ok=True)
        
        print(f"\n{'='*60}")
        print(f"  FINE-TUNING STARTED")
        print(f"{'='*60}")
        print(f"  Total epochs: {num_epochs}")
        print(f"  Learning rate: {self.optimizer.param_groups[0]['lr']:.2e}")
        print(f"  Device: {self.device}")
        print(f"{'='*60}\n")
        
        patience_counter = 0
        
        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()
            
            # Train
            train_loss, train_acc = self.train_epoch()
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            
            # Validate
            val_loss, val_acc, val_f1_macro, val_f1_weighted = self.validate()
            self.history['val_loss'].append(val_loss)
            self.history['val_acc'].append(val_acc)
            self.history['val_f1'].append(val_f1_macro)
            
            # Step scheduler
            self.scheduler.step()
            
            epoch_time = time.time() - epoch_start
            
            print(f"Epoch {epoch}/{num_epochs} ({epoch_time:.1f}s)")
            print(f"  Train Loss: {train_loss:.4f} | Acc: {train_acc:.2%}")
            print(f"  Val Loss:   {val_loss:.4f} | Acc: {val_acc:.2%}")
            print(f"  F1-macro:   {val_f1_macro:.4f} | F1-weighted: {val_f1_weighted:.4f}")
            print(f"  LR: {self.optimizer.param_groups[0]['lr']:.2e}")
            
            # Save best model (based on F1-macro since it's class-balanced)
            if val_f1_macro > self.best_val_f1:
                self.best_val_f1 = val_f1_macro
                self.best_val_acc = val_acc
                patience_counter = 0
                
                checkpoint_path = checkpoint_dir / 'finetune_best.pt'
                # Unwrap DataParallel for saving so checkpoint loads cleanly
                raw_model = self.model.module if isinstance(self.model, nn.DataParallel) else self.model
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': raw_model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_acc': val_acc,
                    'val_f1_macro': val_f1_macro,
                    'val_f1_weighted': val_f1_weighted,
                    'history': self.history
                }, checkpoint_path)
                print(f"  ✓ Best model saved (F1-macro: {val_f1_macro:.4f})")
            else:
                patience_counter += 1
                print(f"  Patience: {patience_counter}/{patience}")
            
            # Early stopping
            if patience_counter >= patience:
                print(f"\nEarly stopping at epoch {epoch}")
                break
            
            print()
        
        print(f"\n{'='*60}")
        print(f"  FINE-TUNING COMPLETED")
        print(f"{'='*60}")
        print(f"  Best val accuracy: {self.best_val_acc:.2%}")
        print(f"  Best val F1-macro: {self.best_val_f1:.4f}")
        print(f"{'='*60}\n")
        
        return self.history


def evaluate_test_set(model, test_loader, device, num_classes):
    """Evaluate on test set with detailed metrics."""
    model.eval()
    all_preds = []
    all_labels = []
    
    print("\nEvaluating on test set...")
    with torch.no_grad():
        for batch in tqdm(test_loader):
            cat_data, num_data, labels = batch
            categorical_input = {k: v.to(device) for k, v in cat_data.items()}
            numerical_input = num_data.to(device) if num_data is not None else None
            labels = labels.to(device)
            
            logits = model(categorical_input, numerical_input)
            preds = torch.argmax(logits, dim=1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    # Compute metrics
    accuracy = accuracy_score(all_labels, all_preds)
    f1_macro = f1_score(all_labels, all_preds, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None, zero_division=0).tolist()
    report = classification_report(all_labels, all_preds, zero_division=0)
    
    print(f"\n{'='*60}")
    print(f"  TEST SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy:    {accuracy:.2%}")
    print(f"  F1-macro:    {f1_macro:.4f}")
    print(f"  F1-weighted: {f1_weighted:.4f}")
    print(f"{'='*60}\n")
    print(report)
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'classification_report': report
    }


def main():
    """Main fine-tuning function."""
    
    parser = argparse.ArgumentParser(description='Fine-tune pre-trained encoder')
    parser.add_argument('--pretrain', type=str, default='checkpoints/pretrain_best.pt',
                        help='Path to pre-training checkpoint')
    parser.add_argument('--no-pretrain', action='store_true',
                        help='Train from scratch (no pre-training)')
    args = parser.parse_args()
    
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
    test_df = pd.read_csv('processed_data/test.csv')
    
    # Use max label + 1 as num_classes (labels go 0-130, so 131 classes)
    # This matches the old baseline and keeps all labels intact
    all_labels = pd.concat([train_df['disease'], val_df['disease'], test_df['disease']])
    num_classes = int(all_labels.max()) + 1  # 131
    print(f"Number of disease classes: {num_classes}")
    print(f"Label range: {int(all_labels.min())} to {int(all_labels.max())}")
    
    train_dataset = GenomicDataset(
        train_df, categorical_features, numerical_features, 'disease'
    )
    val_dataset = GenomicDataset(
        val_df, categorical_features, numerical_features, 'disease'
    )
    test_dataset = GenomicDataset(
        test_df, categorical_features, numerical_features, 'disease'
    )
    
    # Scale batch size with number of GPUs
    batch_size = 64 * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)
    print(f"Batch size: {batch_size} ({64} per GPU x {max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)} GPUs)")

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, 
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    test_loader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=True if torch.cuda.is_available() else False
    )
    
    print(f"Train: {len(train_dataset)} | Val: {len(val_dataset)} | Test: {len(test_dataset)}")
    
    # Create model
    print("\nCreating model...")
    pretrain_checkpoint = None if args.no_pretrain else args.pretrain
    
    model = get_finetune_model(
        categorical_vocab_sizes=metadata['vocab_sizes'],
        categorical_features=metadata['categorical_features'],
        numerical_features=metadata['numerical_features'],
        num_classes=num_classes,
        pretrain_checkpoint=pretrain_checkpoint,
        hidden_dim=256,
        num_chunks=64,
        num_encoder_layers=4,
        num_attention_heads=8,
        feedforward_dim=1024,
        dropout=0.1
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
    trainer = FinetuneTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        num_classes=num_classes,
        lr=1e-4,  # Smaller LR for fine-tuning
        weight_decay=0.01,
        device=device
    )
    
    # Train
    history = trainer.train(
        num_epochs=30,
        patience=8,
        checkpoint_dir='checkpoints'
    )
    
    # Load best model and evaluate on test set
    print("\nLoading best model for test evaluation...")
    checkpoint = torch.load('checkpoints/finetune_best.pt')
    raw_model = model.module if isinstance(model, nn.DataParallel) else model
    raw_model.load_state_dict(checkpoint['model_state_dict'])
    
    test_results = evaluate_test_set(model, test_loader, device, num_classes)
    
    # Save all results
    results = {
        'train_history': history,
        'test_results': test_results,
        'config': {
            'pretrain_checkpoint': pretrain_checkpoint,
            'lr': 1e-4,
            'batch_size': 64,
            'num_classes': num_classes
        }
    }
    
    results_path = Path('results') / f'finetune_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    results_path.parent.mkdir(exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to {results_path}")


if __name__ == '__main__':
    main()
