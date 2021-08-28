from torch import nn
from torch.utils.data import DataLoader
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('Using {} device'.format(device))

def train_epoch(dataset, model, learning_rate, batch_size):
    dataloader = DataLoader(dataset, batch_size=batch_size)
    loss_fn = nn.NLLLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate, momentum=0, weight_decay=0, nesterov=False)
    size = len(dataloader.dataset)
    for batch, (X, y) in enumerate(dataloader):
        X = X.to(device)
        y = y.to(device)
        optimizer.zero_grad()
        # Compute prediction and loss
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

def train(dataset, model, learning_rate = 1, batch_size = 64, epochs = 15):
    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    
    model = model.to(device)  # it's hudge important to move model to device, before creating optimizer
    
    for t in range(epochs):
        print("")
        print(f"Epoch {t+1}\n-------------------------------")
        train_epoch(dataset, model, learning_rate, batch_size)