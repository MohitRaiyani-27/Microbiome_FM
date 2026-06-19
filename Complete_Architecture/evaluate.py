"""
Evaluate a fine-tuned model on validation and test sets.

Usage:
  python Complete_Architecture/evaluate.py
  python Complete_Architecture/evaluate.py --checkpoint checkpoints/finetune_best.pt
"""

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, f1_score, classification_report, confusion_matrix
)
from tqdm import tqdm

ROOT_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT_DIR))

from utils.dataset import GenomicDataset
from Complete_Architecture.model_finetune import get_finetune_model


def evaluate(model, loader, device, split_name):
    model.eval()
    all_preds, all_labels = [], []

    print(f"\nEvaluating on {split_name} set...")
    with torch.no_grad():
        for batch in tqdm(loader, desc=split_name):
            cat_data, num_data, labels = batch
            categorical_input = {k: v.to(device) for k, v in cat_data.items()}
            numerical_input = num_data.to(device) if num_data is not None else None
            labels = labels.to(device)

            logits = model(categorical_input, numerical_input)
            preds = torch.argmax(logits, dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    accuracy    = accuracy_score(all_labels, all_preds)
    f1_macro    = f1_score(all_labels, all_preds, average='macro',    zero_division=0)
    f1_weighted = f1_score(all_labels, all_preds, average='weighted', zero_division=0)
    f1_per_class = f1_score(all_labels, all_preds, average=None,      zero_division=0).tolist()
    report      = classification_report(all_labels, all_preds, zero_division=0)

    print(f"\n{'='*60}")
    print(f"  {split_name.upper()} SET RESULTS")
    print(f"{'='*60}")
    print(f"  Accuracy    : {accuracy:.4f}  ({accuracy:.2%})")
    print(f"  F1-macro    : {f1_macro:.4f}")
    print(f"  F1-weighted : {f1_weighted:.4f}")
    print(f"{'='*60}")
    print("\nPer-class report:")
    print(report)

    return {
        'split': split_name,
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_per_class': f1_per_class,
        'classification_report': report,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', type=str,
                        default='checkpoints/finetune_best.pt',
                        help='Path to fine-tuned checkpoint')
    args = parser.parse_args()

    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    # Load metadata
    print("Loading metadata...")
    with open('processed_data/metadata.json') as f:
        metadata = json.load(f)

    categorical_features = metadata['categorical_features']
    numerical_features   = metadata['numerical_features']

    # Load data
    print("Loading data...")
    val_df  = pd.read_csv('processed_data/val.csv')
    test_df = pd.read_csv('processed_data/test.csv')
    train_df = pd.read_csv('processed_data/train.csv')

    all_labels  = pd.concat([train_df['disease'], val_df['disease'], test_df['disease']])
    num_classes = int(all_labels.max()) + 1
    print(f"Classes: {num_classes}")

    batch_size = 64 * max(1, torch.cuda.device_count() if torch.cuda.is_available() else 1)

    val_loader = DataLoader(
        GenomicDataset(val_df, categorical_features, numerical_features, 'disease'),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )
    test_loader = DataLoader(
        GenomicDataset(test_df, categorical_features, numerical_features, 'disease'),
        batch_size=batch_size, shuffle=False,
        num_workers=4, pin_memory=torch.cuda.is_available()
    )

    # Load model
    print(f"\nLoading checkpoint: {args.checkpoint}")
    model = get_finetune_model(
        categorical_vocab_sizes=metadata['vocab_sizes'],
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        num_classes=num_classes,
        pretrain_checkpoint=None,
        hidden_dim=256,
        num_chunks=64,
        num_encoder_layers=4,
        num_attention_heads=8,
        feedforward_dim=1024,
        dropout=0.1
    )
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print(f"Loaded from epoch {checkpoint.get('epoch', '?')}")

    # Evaluate
    val_results  = evaluate(model, val_loader,  device, 'Validation')
    test_results = evaluate(model, test_loader, device, 'Test')

    # Save results
    results = {
        'checkpoint': args.checkpoint,
        'num_classes': num_classes,
        'validation': val_results,
        'test': test_results,
    }
    out_path = Path('results') / f'eval_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    out_path.parent.mkdir(exist_ok=True)
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2)

    print(f"\nResults saved to {out_path}")
    print("\nSUMMARY")
    print(f"  Val  accuracy={val_results['accuracy']:.4f}  F1-macro={val_results['f1_macro']:.4f}")
    print(f"  Test accuracy={test_results['accuracy']:.4f}  F1-macro={test_results['f1_macro']:.4f}")


if __name__ == '__main__':
    main()
