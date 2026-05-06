# MicrobiomeFM: Transformer Encoder for Gut Microbiome Disease Classification

A transformer-based encoder foundation model that classifies 131 human disease conditions from gut microbiome metagenomic data. The model is pre-trained using masked chunk autoencoding on unlabeled samples and then fine-tuned for disease classification. It was developed as part of research at the University of New Haven.

---

## Overview

Working with microbiome data is hard because the feature space is very wide (2,361 features per patient) and the classes are heavily imbalanced, with some diseases having hundreds of samples while others have only a handful. Standard approaches like projecting each feature to its own token fail here because a single scalar value does not carry enough signal for attention to work well.

The core idea in this project is to group the 2,361 features into 64 chunks of roughly 37 features each. Each chunk is embedded into a single 256-dimensional token that carries real information from all 37 features in that group. A transformer encoder then runs self-attention over these 64 rich tokens, learning relationships between feature groups rather than individual noisy scalars.

Pre-training is done without any disease labels, using masked autoencoding where 30% of the chunk tokens are masked and the encoder must reconstruct them. This gives the encoder a strong initialization before fine-tuning on the classification task.

---

## Dataset

- 14,855 patient samples collected from NCBI and published microbiome studies
- 2,361 features per sample: bacterial species abundances, patient demographics, sequencing metrics, and clinical metadata
- 131 disease classes including Type 2 Diabetes, Colorectal Cancer, Inflammatory Bowel Disease, and many others

The processed dataset is publicly available on Hugging Face:
https://huggingface.co/datasets/mohitraiyani27/Human_Gut_Microbiome_Data

Download the files and place them in the `processed_data/` directory before running any scripts.

---

## Architecture

**Chunked Feature Embedding**

Instead of projecting each feature to its own token, all 2,361 features are split into 64 chunks of roughly 37 features each. Categorical features are passed through an embedding layer to an 8-dimensional vector. Numerical features are projected to 8 dimensions with a linear layer. All vectors in a chunk are concatenated and then projected to 256 dimensions through a linear layer followed by GELU and LayerNorm. The result is 64 tokens of shape `[batch, 64, 256]` where each token carries meaningful information.

**Encoder**

A standard transformer encoder with 4 layers runs self-attention over the 64 chunk tokens. Each chunk learns which other chunks it is related to. For example, a chunk containing gut bacteria abundances learns to attend to a chunk containing dietary or metabolic features.

**Pre-training with Masked Autoencoding**

The encoder is pre-trained by randomly masking 30% of the 64 tokens and asking the model to reconstruct the original embeddings at those positions. MSE loss is computed only on the masked positions. This is done without any disease labels, so all 14,855 samples can be used for pre-training.

**Fine-tuning for Classification**

After pre-training, the encoder weights are loaded and a classification head is added. An attention pooling layer compresses the 64 encoder output tokens into a single 256-dimensional vector using a learnable query. This vector is then passed through a classification head with two linear layers separated by GELU and dropout to produce logits over 131 disease classes.

---

## File Descriptions

| File | Description |
|------|-------------|
| `model.py` | Defines the `GenomicClassifier` class. Contains the chunked feature embedding, the 4-layer transformer encoder, and the classification head. This is the main model definition file. |
| `model_pretrain.py` | Defines `MaskedChunkPretraining`. Adds a mask token and a prediction head on top of the encoder so the model can be trained with masked autoencoding before any labeled data is used. |
| `model_finetune.py` | Defines `EncoderClassifier`. Loads pre-trained encoder weights and attaches an attention pooling layer and classification head for fine-tuning on 131-class disease prediction. |
| `pretrain.py` | Training script for the masked pre-training stage. Handles data loading, the optimizer (AdamW), cosine learning rate schedule, early stopping, and checkpoint saving. |
| `finetune.py` | Training script for the fine-tuning stage. Loads a pre-trained checkpoint, trains the classification model with cross-entropy loss and label smoothing, and reports accuracy and F1 scores. |
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

This loads the pre-trained encoder weights and fine-tunes the model for 131-class classification.

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

The encoder model achieves strong performance on the held-out test set despite the heavy class imbalance across 131 diseases. The two-stage approach of masked pre-training followed by fine-tuning consistently outperforms training from scratch, especially for rare disease classes where labeled data is limited.

---

## Authors

Mohit Raiyani and Khaled Sayed
University of New Haven, April 2026
