import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from readFile import readF
from neuralNetwork import ChessModel, device
import time

print("Starting training!")

epochs = 20
chunkSize = 5000
chunkAmount = 10

model = ChessModel().to(device)
loss = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)


for epoch in range(epochs):
    startTime = time.time()
    model.train()
    averageLoss = 0
    batches = 0
    correct = 0
    batchSize = 0
    for i in range(chunkAmount):
        dataLoader = DataLoader(readF(i, chunkSize), batch_size=64, shuffle=True)
        

        for batch, (pos, move) in enumerate(dataLoader):
            batchSize = pos.size(0)
            pos = pos.to(device)
            move = move.to(device)

            optimizer.zero_grad()
            prediction = model(pos)
            lossValue = loss(prediction, move)
            lossValue.backward()
            optimizer.step()

            maximum, predictedMove = prediction.max(1)
            correct += predictedMove.eq(move).sum().item()

            averageLoss += lossValue.item() * batchSize
            batches += batchSize
        
    averageLoss = averageLoss/batches
    endTime = time.time()
    totalTime = endTime-startTime
    accuracy = correct/batches * 100


    print("---------------------")
    print(f"Epoch #{epoch+1}:") 
    print(f"Time taken: {totalTime}")
    print(f"Average loss: {averageLoss}")
    print(f"Accuracy: {accuracy}%")
    print(correct, batches)
    print("---------------------")   

    if epoch%5 ==0 and epoch != 0:
        torch.save(model.state_dict(), f"/content/drive/MyDrive/ChessIA/model{epoch+1}.pth")
        print(f"Saved model {epoch+1}!")

torch.save(model.state_dict(), f"/content/drive/MyDrive/ChessIA/modelfinal.pth")
print("End of training!")