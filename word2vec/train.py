from torch import nn
from torch.utils.data import DataLoader
import torch

def train_epoch(dataset, model, learning_rate, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(dataset, model, learning_rate = 10, batch_size = 64, epochs = 5):
    for t in range(epochs):
        print("")
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataset, model, learning_rate, batch_size)