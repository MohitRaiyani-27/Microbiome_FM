"""
Chunked Encoder-Decoder Transformer for Genomic Disease Classification
=======================================================================

KEY INSIGHT: Each feature projected to 256-dim individually carries
only 1 number's worth of information. Cross-attention over 2,361 such
tokens fails because they all look nearly identical at initialization.

SOLUTION: Group raw features into 64 CHUNKS of ~37 features each.
Each chunk is embedded as a single rich 256-dim token containing
information from ~37 features. Then self-attention on 64 meaningful
tokens actually works.

Architecture Flow:

    INPUT (2,361 features per patient)
        │
        ▼
    ┌──────────────────────────────────────┐
    │  CHUNKED EMBEDDING                    │
    │                                       │
    │  Split 2,361 features into 64 chunks  │
    │  Each chunk: ~37 features             │
    │                                       │
    │  Categorical: Embedding → 8-dim each  │
    │  Numerical: Linear(1,8) → 8-dim each  │
    │  Chunk: concat 37 × 8-dim = 296-dim   │
    │  Project: Linear(296, 256)             │
    │                                       │
    │  Each token has RICH info from         │
    │  37 features (not just 1!)            │
    │                                       │
    │  Output: [batch, 64, 256]             │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  ENCODER (4 layers)                   │
    │                                       │
    │  SELF-ATTENTION on 64 chunk tokens    │
    │  Each chunk learns relationships      │
    │  with other chunks                    │
    │                                       │
    │  e.g., "gut bacteria chunk" learns    │
    │  it correlates with "diet chunk"      │
    │                                       │
    │  Output: [batch, 64, 256]             │
    └──────────────┬───────────────────────┘
                   │
                   │  encoder output passed as MEMORY
                   ▼
    ┌──────────────────────────────────────┐
    │  DECODER (2 layers)                   │
    │                                       │
    │  Classification query [batch, 1, 256] │
    │                                       │
    │  Each layer:                          │
    │    1. Self-attention (query → query)   │
    │    2. CROSS-ATTENTION                 │
    │       (query attends to ALL 64        │
    │        encoded chunk tokens)          │
    │    3. Feed-forward                    │
    │                                       │
    │  Output: [batch, 1, 256]              │
    └──────────────┬───────────────────────┘
                   │
                   ▼
    ┌──────────────────────────────────────┐
    │  CLASSIFICATION HEAD                  │
    │  Linear(256, 256) → GELU → Dropout   │
    │  Linear(256, num_classes)             │
    │  Output: disease probabilities        │
    └──────────────────────────────────────┘

Why this works but per-feature tokens didn't:
  - Per-feature token: 1 scalar → 256-dim = mostly noise
  - Chunk token: 37 scalars → 296-dim → 256-dim = REAL signal
  - Self-attention on rich tokens can actually learn patterns
"""

import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

# Add parent directory for imports
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
    
    Instead of:  1 feature → 1 token (only 1 number, useless for attention)
    We do:       37 features → 1 token (37 numbers, rich signal!)
    
    Each chunk gets its own Linear projection to hidden_dim.
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
        # Split features into num_chunks groups as evenly as possible
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
        # chunk_i has chunk_sizes[i] features, each embedded to embed_dim
        # So input dim = chunk_sizes[i] * embed_dim
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
        print(f"    Each chunk: {min(self.chunk_sizes)}×{embed_dim}={min(self.chunk_sizes)*embed_dim}d "
              f"→ {hidden_dim}d token")
    
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
        # Categorical: [batch, embed_dim] each
        all_embeddings = []
        for feat in self.categorical_features:
            if feat in categorical_input:
                safe_feat = self._sanitize_name(feat)
                emb = self.categorical_embeddings[safe_feat](categorical_input[feat])
                all_embeddings.append(emb)  # [batch, 8]
        
        # Numerical: [batch, 1] → [batch, embed_dim] each (vectorized)
        if numerical_input is not None and self.num_num > 0:
            # [batch, 1101] → [batch, 1101, 1] → [batch, 1101, 8]
            num_embs = self.numerical_projection(numerical_input.unsqueeze(-1))
            # Split into list of [batch, 8] tensors
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


class CompressedEncoderDecoder(nn.Module):
    """
    Chunked Encoder-Decoder Transformer for Genomic Classification.
    
    Pipeline:
      1. Chunk 2,361 features into 64 groups  →  [batch, 64, 256]
      2. Encoder: self-attention on 64 chunks  →  [batch, 64, 256]
      3. Decoder: query cross-attends encoder  →  [batch, 1, 256]
    """
    
    def __init__(self, categorical_vocab_sizes, categorical_features, 
                 numerical_features, hidden_dim=256, num_chunks=64,
                 num_encoder_layers=4, num_decoder_layers=2,
                 num_attention_heads=8, feedforward_dim=1024, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks
        
        # ================================================================
        # STEP 1: CHUNKED FEATURE EMBEDDING
        # 2,361 features → 64 chunks → 64 rich tokens
        # ================================================================
        
        self.chunk_embedding = ChunkedFeatureEmbedding(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            num_chunks=num_chunks,
            embed_dim=config.DEFAULT_EMBEDDING_DIM,
            hidden_dim=hidden_dim
        )
        
        # Positional encoding for the 64 chunk tokens
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=200)
        
        # ================================================================
        # STEP 2: ENCODER
        # Self-attention on 64 chunk tokens
        # Each chunk learns relationships with other chunks
        # ================================================================
        
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
        
        # ================================================================
        # STEP 3: DECODER
        # Classification query cross-attends to encoder output
        # ================================================================
        
        self.classification_query = nn.Parameter(
            torch.randn(1, 1, hidden_dim) * 0.02
        )
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_dim,
            nhead=num_attention_heads,
            dim_feedforward=feedforward_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer,
            num_layers=num_decoder_layers
        )
        self.decoder_norm = nn.LayerNorm(hidden_dim)
        
        # ================================================================
        # Architecture summary
        # ================================================================
        num_features = len(categorical_features) + len(numerical_features)
        print(f"\n{'='*60}")
        print(f"  CHUNKED ENCODER-DECODER TRANSFORMER")
        print(f"{'='*60}")
        print(f"  Input features: {num_features}")
        print(f"  Chunks: {num_chunks} tokens (each ~{num_features // num_chunks} features)")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Encoder: {num_encoder_layers} layers (self-attention on {num_chunks} tokens)")
        print(f"  Decoder: {num_decoder_layers} layers (cross-attention to encoder)")
        print(f"  Attention heads: {num_attention_heads}")
        print(f"  Feed-forward dim: {feedforward_dim}")
        print(f"{'='*60}\n")
    
    def encode(self, chunk_tokens):
        """
        STEP 2: Encoder — self-attention on 64 chunk tokens.
        
        Input:  [batch, 64, 256]
        Output: [batch, 64, 256]
        """
        # Add positional encoding
        x = self.pos_encoding(chunk_tokens)
        
        # Self-attention across all 64 chunks
        encoded = self.encoder(x)
        encoded = self.encoder_norm(encoded)
        return encoded
    
    def decode(self, encoder_output):
        """
        STEP 3: Decoder — classification query cross-attends to encoder.
        
        Input:  encoder_output [batch, 64, 256]
        Output: [batch, 256]
        """
        batch_size = encoder_output.size(0)
        query = self.classification_query.expand(batch_size, -1, -1)
        
        decoded = self.decoder(tgt=query, memory=encoder_output)
        decoded = self.decoder_norm(decoded)
        
        return decoded[:, 0, :]  # [batch, 256]
    
    def forward(self, categorical_input, numerical_input=None):
        """
        Full forward pass:
          1. Chunk 2,361 features → 64 tokens   [batch, 64, 256]
          2. Encoder (self-attention on chunks)  [batch, 64, 256]
          3. Decoder (cross-attention)           [batch, 256]
        """
        # Step 1: Embed features into 64 chunk tokens
        chunk_tokens = self.chunk_embedding(categorical_input, numerical_input)
        
        # Step 2: Encode (self-attention among chunks)
        encoder_output = self.encode(chunk_tokens)
        
        # Step 3: Decode (cross-attention for classification)
        classification_vector = self.decode(encoder_output)
        
        return classification_vector  # [batch, 256]


class GenomicClassifier(nn.Module):
    """
    Complete model: Chunked Encoder-Decoder + Classification Head.
    
    Pipeline:
      Input → Chunk → Encode → Decode → Classify → diseases
    """
    
    def __init__(self, encoder_decoder, num_classes, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.encoder_decoder = encoder_decoder
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, categorical_input, numerical_input=None):
        """Input → Encoder-Decoder → 256-dim → Classification → logits"""
        embedding = self.encoder_decoder(categorical_input, numerical_input)
        logits = self.classifier(embedding)
        return logits
