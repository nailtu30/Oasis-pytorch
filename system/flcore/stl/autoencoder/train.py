import torch
import torchvision
from models import Autoencoder
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np
import csv

batch_size_train = 32
batch_size_test = 32

device = 'cuda' if torch.cuda.is_available else 'cpu'

train_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=True, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_train, shuffle=True)
test_loader = torch.utils.data.DataLoader(
    torchvision.datasets.MNIST('/data/', train=False, download=False,
                               transform=torchvision.transforms.Compose([
                                   torchvision.transforms.ToTensor(),
                                   torchvision.transforms.Normalize(
                                       (0.1307,), (0.3081,))
                               ])),
    batch_size=batch_size_test, shuffle=True)

epochs = 100
lr = 1e-3
model = Autoencoder().to(device)
criteon = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=lr)


def main():
    for epoch in range(epochs):
        losses = []
        for batchidx, (x, _) in enumerate(train_loader):
            x = x.to(device)
            x_hat = model(x)
            loss = criteon(x_hat, x)
            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        print(epoch, 'loss:', np.mean(losses))
        x, _ = next(iter(test_loader))
        x = x.to(device)
        with torch.no_grad():
            x_hat = model(x)
        save_image(x_hat, 'results/fake_imgs.png', normalize=True)
        with open('results/train_loss.csv', 'a') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow([np.mean(losses)])
        torch.save(model.encoder.state_dict(), 'results/enc.pth')


if __name__ == '__main__':
    main()
