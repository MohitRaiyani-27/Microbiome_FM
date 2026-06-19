"""
Masked Chunk Pre-training Model for Genomic Foundation Model
=============================================================

Strategy: Masked autoencoding of genomic chunk embeddings
- Randomly mask 30% of the 64 chunk tokens
- Encoder processes all 64 chunks (including masked)
- Prediction head reconstructs the original embeddings of masked chunks
- MSE loss compares predicted vs original embeddings

This teaches the encoder to understand relationships between chunks
without needing disease labels.
"""

import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding."""
    
    def __init__(self, d_model, max_len=500):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class ChunkedFeatureEmbedding(nn.Module):
    """
    Split 2,361 features into 64 chunks, embed each chunk as one rich token.
    """
    
    def __init__(self, categorical_features, numerical_features,
                 categorical_vocab_sizes, num_chunks=64, 
                 embed_dim=8, hidden_dim=256):
        super().__init__()
        
        self.categorical_features = categorical_features
        self.numerical_features = numerical_features
        self.num_chunks = num_chunks
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # All feature names in order
        self.all_features = list(categorical_features) + list(numerical_features)
        self.num_features = len(self.all_features)
        self.num_cat = len(categorical_features)
        self.num_num = len(numerical_features)
        
        # Categorical embeddings (shared across chunks)
        self.categorical_embeddings = nn.ModuleDict()
        for feat in categorical_features:
            vocab_size = categorical_vocab_sizes.get(feat, 100)
            safe_feat = self._sanitize_name(feat)
            self.categorical_embeddings[safe_feat] = nn.Embedding(
                vocab_size, embed_dim
            )
        
        # Numerical projection (shared)
        self.numerical_projection = nn.Linear(1, embed_dim)
        
        # Compute chunk assignments
        self.chunk_sizes = []
        self.chunk_ranges = []
        base_size = self.num_features // num_chunks
        remainder = self.num_features % num_chunks
        
        start = 0
        for i in range(num_chunks):
            size = base_size + (1 if i < remainder else 0)
            self.chunk_sizes.append(size)
            self.chunk_ranges.append((start, start + size))
            start += size
        
        # Each chunk: concat embed_dim vectors → project to hidden_dim
        self.chunk_projections = nn.ModuleList()
        for i in range(num_chunks):
            input_dim = self.chunk_sizes[i] * embed_dim
            self.chunk_projections.append(
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.LayerNorm(hidden_dim)
                )
            )
        
        print(f"  Chunked Embedding:")
        print(f"    {self.num_features} features → {num_chunks} chunks")
        print(f"    Chunk sizes: {min(self.chunk_sizes)}-{max(self.chunk_sizes)} features each")
    
    def _sanitize_name(self, name):
        """Clean feature names for PyTorch module keys."""
        name = str(name).replace(':', '_').replace('.', '_').replace(' ', '_')
        name = name.replace('(', '_').replace(')', '_').replace('-', '_')
        name = name.replace('/', '_').replace('[', '_').replace(']', '_')
        while '__' in name:
            name = name.replace('__', '_')
        return name.strip('_')
    
    def forward(self, categorical_input, numerical_input=None):
        """
        Embed all features, then group into chunks.
        
        Input:  categorical_input dict + numerical_input [batch, num_numerical]
        Output: [batch, 64, 256]  (64 chunk tokens, each rich with ~37 features)
        """
        batch_size = list(categorical_input.values())[0].size(0)
        
        # Step 1: Embed ALL features individually to embed_dim
        all_embeddings = []
        for feat in self.categorical_features:
            if feat in categorical_input:
                safe_feat = self._sanitize_name(feat)
                emb = self.categorical_embeddings[safe_feat](categorical_input[feat])
                all_embeddings.append(emb)  # [batch, 8]
        
        # Numerical: [batch, 1101] → [batch, 1101, 1] → [batch, 1101, 8]
        if numerical_input is not None and self.num_num > 0:
            num_embs = self.numerical_projection(numerical_input.unsqueeze(-1))
            for i in range(self.num_num):
                all_embeddings.append(num_embs[:, i, :])  # [batch, 8]
        
        # Step 2: Group into chunks and project each chunk to hidden_dim
        chunk_tokens = []
        for chunk_idx in range(self.num_chunks):
            start, end = self.chunk_ranges[chunk_idx]
            
            # Gather embeddings for this chunk: list of [batch, 8]
            chunk_embs = all_embeddings[start:end]
            
            # Concatenate: [batch, chunk_size * 8]
            chunk_concat = torch.cat(chunk_embs, dim=1)
            
            # Project to hidden_dim: [batch, 256]
            chunk_token = self.chunk_projections[chunk_idx](chunk_concat)
            chunk_tokens.append(chunk_token.unsqueeze(1))  # [batch, 1, 256]
        
        # Stack all chunks: [batch, 64, 256]
        tokens = torch.cat(chunk_tokens, dim=1)
        
        return tokens


class MaskedChunkPretraining(nn.Module):
    """
    Masked chunk autoencoding for pre-training.
    
    Pipeline:
      1. Embed 64 chunks normally → save as targets
      2. Mask 30% randomly
      3. Encoder processes masked chunks
      4. Prediction head reconstructs masked embeddings
      5. MSE loss on masked positions
    """
    
    def __init__(self, categorical_vocab_sizes, categorical_features, 
                 numerical_features, hidden_dim=256, num_chunks=64,
                 num_encoder_layers=4, num_attention_heads=8, 
                 feedforward_dim=1024, dropout=0.1, mask_ratio=0.3):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks
        self.mask_ratio = mask_ratio
        
        # Chunked feature embedding
        self.chunk_embedding = ChunkedFeatureEmbedding(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            num_chunks=num_chunks,
            embed_dim=config.DEFAULT_EMBEDDING_DIM,
            hidden_dim=hidden_dim
        )
        
        # Learnable [MASK] token
        self.mask_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=200)
        
        # Encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(
            encoder_layer,
            num_layers=num_encoder_layers
        )
        self.encoder_norm = nn.LayerNorm(hidden_dim)
        
        # Prediction head (reconstructs chunk embeddings)
        self.prediction_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        print(f"\n{'='*60}")
        print(f"  MASKED CHUNK PRE-TRAINING MODEL")
        print(f"{'='*60}")
        print(f"  Chunks: {num_chunks}")
        print(f"  Mask ratio: {mask_ratio:.0%} ({int(num_chunks * mask_ratio)} chunks)")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Encoder layers: {num_encoder_layers}")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  Feed-forward dim: {feedforward_dim}")
        print(f"{'='*60}\n")
    
    def random_masking(self, x):
        """
        Randomly mask 30% of chunks.
        
        Args:
            x: [batch, num_chunks, hidden_dim]
        
        Returns:
            x_masked: [batch, num_chunks, hidden_dim] with masked positions
            mask: [batch, num_chunks] (1 = keep, 0 = masked)
        """
        batch_size, num_chunks, hidden_dim = x.shape
        
        # Number of chunks to mask
        num_masked = int(num_chunks * self.mask_ratio)
        
        # Generate random mask for each sample in batch
        mask = torch.ones(batch_size, num_chunks, device=x.device)
        for i in range(batch_size):
            # Randomly select positions to mask
            masked_indices = torch.randperm(num_chunks)[:num_masked]
            mask[i, masked_indices] = 0
        
        # Replace masked positions with [MASK] token
        x_masked = x.clone()
        mask_token = self.mask_token.expand(batch_size, num_chunks, -1)
        x_masked = x_masked * mask.unsqueeze(-1) + mask_token * (1 - mask.unsqueeze(-1))
        
        return x_masked, mask
    
    def forward(self, categorical_input, numerical_input=None):
        """
        Forward pass for pre-training.
        
        Returns:
            predictions: [batch, num_chunks, hidden_dim] predicted embeddings
            targets: [batch, num_chunks, hidden_dim] original embeddings
            mask: [batch, num_chunks] (1 = keep, 0 = masked)
        """
        # Step 1: Embed chunks normally (these are our targets)
        chunk_embeddings = self.chunk_embedding(categorical_input, numerical_input)
        targets = chunk_embeddings.clone()  # [batch, 64, 256]
        
        # Step 2: Apply masking
        chunk_embeddings_masked, mask = self.random_masking(chunk_embeddings)
        
        # Step 3: Add positional encoding
        x = self.pos_encoding(chunk_embeddings_masked)
        
        # Step 4: Encoder
        encoded = self.encoder(x)
        encoded = self.encoder_norm(encoded)  # [batch, 64, 256]
        
        # Step 5: Prediction head (reconstruct embeddings)
        predictions = self.prediction_head(encoded)  # [batch, 64, 256]
        
        return predictions, targets, mask
    
    def compute_loss(self, predictions, targets, mask):
        """
        Compute MSE loss only on masked positions.
        
        Args:
            predictions: [batch, num_chunks, hidden_dim]
            targets: [batch, num_chunks, hidden_dim]
            mask: [batch, num_chunks] (1 = keep, 0 = masked)
        
        Returns:
            loss: scalar
        """
        # Only compute loss on masked positions (mask == 0)
        masked_positions = (mask == 0).unsqueeze(-1)  # [batch, num_chunks, 1]
        
        # MSE on masked positions only
        loss = ((predictions - targets) ** 2) * masked_positions
        loss = loss.sum() / masked_positions.sum()
        
        return loss


def get_pretrain_model(categorical_vocab_sizes, categorical_features, 
                       numerical_features, **kwargs):
    """Create pre-training model."""
    return MaskedChunkPretraining(
        categorical_vocab_sizes=categorical_vocab_sizes,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        **kwargs
    )
