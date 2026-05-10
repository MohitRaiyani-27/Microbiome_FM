# MicrobiomeFM: Transformer Encoder for Gut Microbiome Disease Classification

A transformer-based encoder foundation model that classifies 40 human disease
conditions from gut microbiome metagenomic data. The model is pre-trained using
masked chunk autoencoding on unlabeled samples and then fine-tuned for disease
classification. It was developed as part of research at the University of New Haven.

---

## Overview

Working with microbiome data is hard because the feature space is very wide
(2,361 features per patient) and the classes are heavily imbalanced, with some
diseases having hundreds of samples while others have only a handful. Standard
approaches like projecting each feature to its own token fail here because a single
scalar value does not carry enough signal for attention to work well.

The core idea in this project is to group the 2,361 features into 64 chunks of
roughly 37 features each. Each chunk is embedded into a single 256-dimensional
token that carries real information from all 37 features in that group. A
transformer encoder then runs self-attention over these 64 rich tokens, learning
relationships between feature groups rather than individual noisy scalars.

Pre-training is done without any disease labels, using masked autoencoding where
30% of the chunk tokens are masked and the encoder must reconstruct them. This
gives the encoder a strong initialization before fine-tuning on the classification task.

---

## Dataset

- **14,488** patient samples collected from NCBI and published microbiome studies
- **2,361** features per sample: bacterial species abundances, patient demographics, sequencing metrics, and clinical metadata
- **40 disease classes** — filtered from an original 131 by removing all conditions with fewer than 20 samples, ensuring every class has sufficient representation for reliable learning
- Classes include: Type 2 Diabetes, Colorectal Cancer, Inflammatory Bowel Disease, Parkinson's Disease, Schizophrenia, Melanoma, and 34 others
- Train / Validation / Test split: **10,136 / 2,176 / 2,176** (stratified)

The processed dataset is publicly available on Hugging Face:
https://huggingface.co/datasets/mohitraiyani27/Human_Gut_Microbiome_Data

Download the files and place them in the `processed_data/` directory before running any scripts.

---

## Architecture

**Chunked Feature Embedding**

Instead of projecting each feature to its own token, all 2,361 features are
split into 64 chunks of roughly 37 features each. Categorical features are passed
through an embedding layer to an 8-dimensional vector. Numerical features are
projected to 8 dimensions with a linear layer. All vectors in a chunk are
concatenated and then projected to 256 dimensions through a linear layer followed by GELU
and LayerNorm. The result is 64 tokens of shape `[batch, 64, 256]` where each token carries meaningful information.

**Encoder**

A standard transformer encoder with 4 layers runs self-attention over the 64
chunk tokens. Each chunk learns which other chunks it is related to. For example, a
chunk containing gut bacteria abundances learns to attend to a chunk containing
dietary or metabolic features.

**Pre-training with Masked Autoencoding**

The encoder is pre-trained by randomly masking 30% of the 64 tokens and asking
the model to reconstruct the original embeddings at those positions. MSE loss is
computed only on the masked positions. This is done without any disease labels,
so all 14,488 samples can be used for pre-training.

**Fine-tuning for Classification**

After pre-training, the encoder weights are loaded and a classification head is
added. An attention pooling layer compresses the 64 encoder output tokens into a
single 256-dimensional vector using a learnable query. This vector is then passed
through a classification head with two linear layers separated by GELU and dropout
to produce logits over 40 disease classes. Cross-entropy loss with label smoothing
(ε = 0.1) is used during fine-tuning.

---

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Defines the `GenomicClassifier` class. Contains the chunked feature embedding, the 4-layer transformer encoder, and the classification head. This is the main model definition file. |
| `model_pretrain.py` | Defines `MaskedChunkPretraining`. Adds a mask token and a prediction head on top of the encoder so the model can be trained with masked autoencoding before any labeled data is used. |
| `model_finetune.py` | Defines `EncoderClassifier`. Loads pre-trained encoder weights and attaches an attention pooling layer and classification head for fine-tuning on 40-class disease prediction. |
| `pretrain.py` | Training script for the masked pre-training stage. Handles data loading, the optimizer (AdamW), cosine learning rate schedule, early stopping, and checkpoint saving. |
| `finetune.py` | Training script for the fine-tuning stage. Loads a pre-trained checkpoint, trains the classification model with cross-entropy loss and label smoothing (ε = 0.1), and reports accuracy and F1 scores. Supports multi-GPU training via DataParallel. |
| `train.py` | End-to-end training script that trains the full model from scratch without a separate pre-training stage. Useful for quick baselines or ablations. |

---

## How to Run

**Step 1: Pre-train the encoder**

```bash
python Complete_Architecture/pretrain.py
```

This will train the masked autoencoder and save the best checkpoint to `checkpoints/pretrain_best.pt`.

**Step 2: Fine-tune for disease classification**

```bash
python Complete_Architecture/finetune.py --pretrain checkpoints/pretrain_best.pt
```

This loads the pre-trained encoder weights and fine-tunes the model for 40-class classification.

**Alternative: Train from scratch**

If you want to skip pre-training and train end-to-end directly:

```bash
python Complete_Architecture/train.py
```

---

## Requirements

```
torch
numpy
pandas
scikit-learn
tqdm
```

Install with:

```bash
pip install -r requirements.txt
```

---

## Results

| Metric | Value |
|--------|-------|
| Test Accuracy | **96.05%** |
| Validation Accuracy | **93.90%** |
| F1-Macro | **0.691** |
| F1-Weighted | **0.950** |
| Classes | 40 |
| Total Samples | 14,488 |

The two-stage approach of masked pre-training followed by fine-tuning consistently
outperforms training from scratch. Pre-training achieves a best validation loss of
0.0025 after 50 epochs. Fine-tuning runs for 30 epochs with AdamW (lr = 1e-4,
batch size = 64). Classes with fewer than 20 samples were excluded from the
dataset, which significantly improved F1-Macro from 0.34 (131-class run) to 0.691
(40-class run) while also improving accuracy from 93.76% to 96.05%.

---

## Authors

Mohit Raiyani and Khaled Sayed
University of New Haven, May 2026
