import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam # we will use Adam instead of Stochastic Gradient Decent. 
import lightning as L 
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
from sklearn.model_selection import train_test_split
from pathlib import Path

df = pd.read_table(Path(__file__).parent/"iris.txt", sep=",", header=None)

print(df.head())