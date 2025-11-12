import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

dataloader = DataLoader(dataset, batch_size=16, shuffle=True)