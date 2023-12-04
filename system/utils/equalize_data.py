# prerequisites
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable
from torchvision.utils import save_image
import numpy as np
import random

random.seed(1)
np.random.seed(1)


def euualize_MNIST(client_num=100):

    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize([0.5], [0.5])])
    # MNIST Dataset
    train_dataset = datasets.MNIST(
        root='/data', train=True, transform=transform, download=False)
    # test_dataset = datasets.MNIST(root='/data', train=False, transform=transforms.ToTensor(), download=False)

    # Data Loader (Input Pipeline)
    train_loader = torch.utils.data.DataLoader(
        dataset=train_dataset, batch_size=1, shuffle=False)
    # test_loader = torch.utils.data.DataLoader(dataset=test_dataset, batch_size=bs, shuffle=False)

    # label_count = [0 for _ in range(10)]
    ordered_datasets = [[] for _ in range(10)]

    for idx, (data, target) in enumerate(train_loader):
        label = target[0].item()
        ordered_datasets[label].append((data, target))

    client_datasets = [[] for _ in range(client_num)]

    for ordered_dataset in ordered_datasets:
        client_quantity = len(ordered_dataset) // client_num
        last_client_quantity = len(ordered_dataset) - \
            client_quantity * (client_num - 1)
        for i in range(client_num - 1):
            client_datasets[i].extend(
                ordered_dataset[i*client_quantity:(i+1)*client_quantity])
        client_datasets[client_num -
                        1].extend(ordered_dataset[len(ordered_dataset)-last_client_quantity:])

    return client_datasets
    # for idx, client_dataset in enumerate(client_datasets):
    #     print('client: {}'.format(idx))
    #     last_y = 0
    #     count = 0
    #     for x, y in client_dataset:
    #         if last_y == y.item():
    #             count = count + 1
    #         else:
    #             last_y = y.item()
    #             print('label: {}, size: {}'.format(y.item(), count))
    #             count = 0
