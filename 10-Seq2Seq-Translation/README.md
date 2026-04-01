# Seq2Seq Translation

English to Spanish translation using Seq2Seq (Encoder-Decoder) architecture, based on the StatQuest Neural Networks book.

## What it covers
- Encoder: Embedding → LSTM → hidden/cell state
- Decoder: Embedding → LSTM → linear → vocabulary
- Teacher forcing (training)
- Autoregressive generation (inference)
- <EOS> token for sequence boundaries

## Dataset
- English: lets, go, to, <EOS>
- Spanish: ir, vamos, y, <EOS>

## Tech
- PyTorch, PyTorch Lightning, nn.LSTM, nn.Embedding

## Run
```bash
python seq2seq.py