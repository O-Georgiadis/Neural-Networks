# LSTM from Scratch

A manual implementation of an LSTM (Long Short-Term Memory) network from scratch using PyTorch Lightning.

## What it covers
- LSTM cell architecture (input, forget, output gates)
- Long-term and short-term memory mechanisms
- Sequential data processing
- Sequence classification (2 companies)
- Training for 5000 epochs

## How it works
- Processes input sequences step-by-step
- Uses sigmoid for gates, tanh for candidate values
- Maintains long memory (cell state) and short memory (hidden state)
- Updates memory based on relevant patterns in sequence

## Tech
- PyTorch
- PyTorch Lightning
- TensorBoard (for logging)

## Run
```bash
python lstm_from_scratch.py
# For TensorBoard:
tensorboard --logdir=lightning_logs/