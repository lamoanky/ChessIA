import torch
import torch.nn as nn

device = ""
if(torch.cuda.is_available()):
    device = "cuda"
else:
    device = "cpu"

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

        #flattening layer
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(128*8*8, 1024)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.3)   
        self.fc2 = nn.Linear(1024, 4096)
  

        
    def forward(self, x): #returns predictions
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.batchNorm1(x)

        x = self.conv2(x)
        x = self.relu2(x)
        x = self.batchNorm2(x)

        x = self.flatten(x)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.dropout(x)
        x = self.fc2(x)

        return x