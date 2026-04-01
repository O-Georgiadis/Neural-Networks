# PyTorch LSTM

Using PyTorch's built-in `nn.LSTM` for sequence classification.

## What it covers
- `nn.LSTM` layer
- Input reshaping for LSTM
- Using last output for prediction
- Adam optimizer with lr=0.1

## Result
- Company A: predicts ~0
- Company B: predicts ~1
- 300 epochs (faster than manual)

## Run
```bash
python lstm_pytorch.py
tensorboard --logdir=lightning_logs/