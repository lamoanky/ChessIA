import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from readFile import dataset
from neuralNetwork import ChessModel, device
import time

print("Starting training!")

epochs = 3
batchSize = 64

model = ChessModel().to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)
dataLoader = DataLoader(dataset, batch_size=batchSize, shuffle=True)

for epoch in range(epochs):
    startTime = time.time()

    model.train()
    averageLoss = 0
    totalBatches = len(dataLoader)
    batches = 0
    correct = 0

    for batch, (pos, move) in enumerate(dataLoader):
        pos = pos.to(device)
        move = move.to(device)

        optimizer.zero_grad()
        prediction = model(pos)
        lossValue = loss(prediction, move)
        lossValue.backward()
        optimizer.step()

        maximum, predictedMove = prediction.max(1)
        correct += predictedMove.eq(move).sum().item()

        averageLoss += lossValue.item()
    
    averageLoss = averageLoss/totalBatches
    endTime = time.time()
    totalTime = endTime-startTime
    batches += batchSize


    print("---------------------")
    print(f"Epoch #{epoch+1}:") 
    print(f"Time taken: {totalTime}")
    print(f"Average loss: {averageLoss}")
    print(f"Accuracy: {correct/len(dataset) * 100}%")
    print("---------------------")   

    



print("End of training!")

