from torch import nn
from torch.utils.data import DataLoader
import torch



def train_epoch(dataset, model, first_device, learning_rate, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0, nesterov=False)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(first_device)
        y = y.to(first_device)
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
        y = y.to(pred.device)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(dataset, model, first_device, learning_rate = 1, batch_size = 64, epochs = 15):
    for t in range(epochs):
        print("")
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataset, model, first_device, learning_rate, batch_size)