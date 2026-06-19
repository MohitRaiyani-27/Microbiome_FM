"""
Fine-tuning Model with Attention Pooling
=========================================

After pre-training the encoder with masked chunk autoencoding,
this model:
1. Loads the pre-trained encoder weights
2. Uses attention pooling to compress 64 tokens → 1 vector
3. Adds classification head for 131 diseases

No decoder — classification happens directly from encoder output.
"""

import torch
import torch.nn as nn
import math
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
import config

# Import the chunked embedding and encoder from pre-training model
from Complete_Architecture.model_pretrain import ChunkedFeatureEmbedding, PositionalEncoding


class AttentionPooling(nn.Module):
    """
    Attention-based pooling to compress 64 tokens into 1 vector.
    
    Uses a learnable query that attends to all 64 encoder tokens,
    computing attention weights and producing a weighted sum.
    """
    
    def __init__(self, hidden_dim=256):
        super().__init__()
        self.hidden_dim = hidden_dim
        
        # Learnable pooling query
        self.query = nn.Parameter(torch.randn(1, hidden_dim) * 0.02)
        
        # Query, Key, Value projections
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = nn.Linear(hidden_dim, hidden_dim)
        
        self.scale = math.sqrt(hidden_dim)
    
    def forward(self, encoder_output):
        """
        Pool 64 tokens into 1 vector using attention.
        
        Args:
            encoder_output: [batch, 64, hidden_dim]
        
        Returns:
            pooled: [batch, hidden_dim]
        """
        batch_size = encoder_output.size(0)
        
        # Expand query for batch
        query = self.query.unsqueeze(0).expand(batch_size, -1, -1)  # [batch, 1, hidden_dim]
        
        # Project to Q, K, V
        Q = self.q_proj(query)  # [batch, 1, hidden_dim]
        K = self.k_proj(encoder_output)  # [batch, 64, hidden_dim]
        V = self.v_proj(encoder_output)  # [batch, 64, hidden_dim]
        
        # Attention scores: Q @ K^T / sqrt(d)
        attn_scores = torch.matmul(Q, K.transpose(1, 2)) / self.scale  # [batch, 1, 64]
        attn_weights = torch.softmax(attn_scores, dim=-1)  # [batch, 1, 64]
        
        # Weighted sum: attention @ V
        pooled = torch.matmul(attn_weights, V)  # [batch, 1, hidden_dim]
        pooled = pooled.squeeze(1)  # [batch, hidden_dim]
        
        return pooled, attn_weights.squeeze(1)  # Return weights for visualization


class EncoderClassifier(nn.Module):
    """
    Encoder-only model with attention pooling for disease classification.
    
    Pipeline:
      1. Chunk features → 64 tokens
      2. Encoder (pre-trained) → 64 enriched tokens
      3. Attention pooling → 1 vector
      4. Classification head → 131 diseases
    """
    
    def __init__(self, categorical_vocab_sizes, categorical_features, 
                 numerical_features, num_classes, hidden_dim=256, 
                 num_chunks=64, num_encoder_layers=4, num_attention_heads=8,
                 feedforward_dim=1024, dropout=0.1):
        super().__init__()
        
        self.hidden_dim = hidden_dim
        self.num_chunks = num_chunks
        self.num_classes = num_classes
        
        # Chunked feature embedding (same as pre-training)
        self.chunk_embedding = ChunkedFeatureEmbedding(
            categorical_features=categorical_features,
            numerical_features=numerical_features,
            categorical_vocab_sizes=categorical_vocab_sizes,
            num_chunks=num_chunks,
            embed_dim=config.DEFAULT_EMBEDDING_DIM,
            hidden_dim=hidden_dim
        )
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(hidden_dim, max_len=200)
        
        # Encoder (same as pre-training)
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
        
        # Attention pooling (NEW for fine-tuning)
        self.attention_pooling = AttentionPooling(hidden_dim)
        
        # Classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes)
        )
        
        print(f"\n{'='*60}")
        print(f"  ENCODER CLASSIFIER (FINE-TUNING)")
        print(f"{'='*60}")
        print(f"  Chunks: {num_chunks}")
        print(f"  Hidden dim: {hidden_dim}")
        print(f"  Encoder layers: {num_encoder_layers}")
        print(f"  Pooling: Attention-based")
        print(f"  Num classes: {num_classes}")
        print(f"{'='*60}\n")
    
    def load_pretrained_encoder(self, pretrain_checkpoint_path):
        """
        Load pre-trained encoder weights.
        
        Args:
            pretrain_checkpoint_path: Path to pre-training checkpoint
        """
        print(f"Loading pre-trained encoder from {pretrain_checkpoint_path}...")
        checkpoint = torch.load(pretrain_checkpoint_path, map_location='cpu')
        
        # Extract encoder-related weights from pre-training checkpoint
        pretrain_state_dict = checkpoint['model_state_dict']
        
        # Map pre-training weights to fine-tuning model
        model_state_dict = self.state_dict()
        
        # Load matching keys
        loaded_keys = []
        for key in pretrain_state_dict:
            if key in model_state_dict:
                if pretrain_state_dict[key].shape == model_state_dict[key].shape:
                    model_state_dict[key] = pretrain_state_dict[key]
                    loaded_keys.append(key)
        
        self.load_state_dict(model_state_dict)
        
        print(f"Loaded {len(loaded_keys)} pre-trained parameters")
        print(f"  - chunk_embedding: ✓")
        print(f"  - encoder: ✓")
        print(f"  - attention_pooling: new (randomly initialized)")
        print(f"  - classifier: new (randomly initialized)")
    
    def forward(self, categorical_input, numerical_input=None, return_attention=False):
        """
        Forward pass for disease classification.
        
        Args:
            categorical_input: Dict of categorical features
            numerical_input: Tensor of numerical features
            return_attention: If True, also return attention weights
        
        Returns:
            logits: [batch, num_classes]
            (optional) attn_weights: [batch, 64] if return_attention=True
        """
        # Step 1: Embed chunks
        chunk_embeddings = self.chunk_embedding(categorical_input, numerical_input)
        
        # Step 2: Add positional encoding
        x = self.pos_encoding(chunk_embeddings)
        
        # Step 3: Encoder
        encoded = self.encoder(x)
        encoded = self.encoder_norm(encoded)  # [batch, 64, 256]
        
        # Step 4: Attention pooling (64 tokens → 1 vector)
        pooled, attn_weights = self.attention_pooling(encoded)  # [batch, 256], [batch, 64]
        
        # Step 5: Classify
        logits = self.classifier(pooled)  # [batch, num_classes]
        
        if return_attention:
            return logits, attn_weights
        else:
            return logits


def get_finetune_model(categorical_vocab_sizes, categorical_features, 
                       numerical_features, num_classes, pretrain_checkpoint=None, **kwargs):
    """
    Create fine-tuning model and optionally load pre-trained weights.
    
    Args:
        categorical_vocab_sizes: Vocab sizes for categorical features
        categorical_features: List of categorical feature names
        numerical_features: List of numerical feature names
        num_classes: Number of disease classes
        pretrain_checkpoint: Path to pre-training checkpoint (optional)
        **kwargs: Additional model arguments
    
    Returns:
        model: EncoderClassifier
    """
    model = EncoderClassifier(
        categorical_vocab_sizes=categorical_vocab_sizes,
        categorical_features=categorical_features,
        numerical_features=numerical_features,
        num_classes=num_classes,
        **kwargs
    )
    
    # Load pre-trained weights if provided
    if pretrain_checkpoint is not None:
        model.load_pretrained_encoder(pretrain_checkpoint)
    
    return model
