# Transformer

Building a Transformer from scratch, based on the StatQuest Neural Networks book.

## What it covers
- Position Encoding (sin/cos curves)
- Attention (Q, K, V projections)
- Encoder + Decoder architecture
- Masked Self-Attention
- Encoder-Decoder Attention
- Residual connections

## Dataset
- Translation: "Let's go" → "vamos<EOS>"
- Translation: "to go" → "ir<EOS>"

## Tech
- PyTorch, Lightning

## Run
```bash
jupyter notebook transformer.ipynb