import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import chess
import chess.pgn

class ChessPositionDataset(Dataset):
    def __init__(self, pos, move):
        self.pos = pos
        self.move = move

    def __len__(self):
        return len(self.pos)
    
    def __getitem__(self, idx):
        return self.pos[idx], self.move[idx]

