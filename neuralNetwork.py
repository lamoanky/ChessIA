import torch
import torch.nn as nn

device = ""
if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

print(f"Using {device}")

class ChessModel(nn.Module):
    def __init__(self):
        super().__init__()

        #converts 12x8x8 into 64x8x8 by looking at 3x3 squares to help detect patterns
        self.conv1 = nn.Conv2d(in_channels = 12, out_channels=64, kernel_size = 3, padding = 1)
        self.relu1 = nn.ReLU()
        self.batchNorm1 = nn.BatchNorm2d(64)

        #second layer lets model get mroe depth into moves
        self.conv2 = nn.Conv2d(in_channels = 64, out_channels=128, kernel_size = 3, padding = 1)
        self.relu2 = nn.ReLU()
        self.batchNorm2 = nn.BatchNorm2d(128)

        
    def forward(self, x):
        pass