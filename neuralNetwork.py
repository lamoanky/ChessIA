import torch
from torch import nn

device = ""
if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device}")

class ChessEngine(nn.Module):
    def __init__(self):
        super().__init__()
        self.flatten = nn.Flatten()
        self.linear_relu_stack = nn.Sequential(

        )
    def forward(self):
        pass