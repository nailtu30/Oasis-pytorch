# celeba non iid 划分
# 根据celeba图像编号划分
# 比如，有10个client，随机将图像分配到这是个client中
import os
import numpy as np
import argparse
import random

import torch
import torchvision

from torchvision.datasets.vision import VisionDataset
from torchvision import datasets, transforms, utils
from PIL import Image

seed = 0
np.random.seed(seed)
random.seed(seed)

# def get_args():
#     parser = argparse.ArgumentParser()
#     parser.add_argument('--num_clients', '-nc', type=int)
#     parser.add_argument('--alpha', '-alpha', type=float, default=5)
#     parser.add_argument('--seed', default=0, type=int)
#     args = parser.parse_args()
#     seed = args.seed
#     np.random.seed(seed)
#     random.seed(seed)
#     return args

def split_data(num_clients, alpha=5, data_path='/data/celeba/img_align_celeba_png'):
    print('num_clients: {}'.format(num_clients))
    num_files = len(os.listdir(data_path))
    print('num_files: {}'.format(num_files))
    file_indexes = np.arange(1, num_files + 1)
    print('file_indexes: {}'.format(file_indexes))
    np.random.shuffle(file_indexes)
    print('file_indexes: {}'.format(file_indexes))
    proportions = np.random.dirichlet(
                    np.repeat(alpha, num_clients))
    proportions = proportions/proportions.sum() * num_files
    clients_file_indexes = [[] for _ in range(num_clients)]
    start_index = 0
    for client_index, proportion in enumerate(proportions):
        end_index = round(proportion) + start_index
        clients_file_indexes[client_index] = file_indexes[start_index: end_index]
    print('clients_file_indexes: {}'.format(clients_file_indexes))
    return clients_file_indexes


class Celeba(VisionDataset):

    def __init__(self, root, transform, batch_size = 60, test_mode = False, return_all = False, imsize=128):

        self.root = root
        self.transform = transform
        self.return_all = return_all
        all_files = os.listdir(self.root)
        self.all_files = all_files
        self.length = len(all_files)
        self.fixed_transform = transforms.Compose(
                [ transforms.Resize(imsize),
                    transforms.CenterCrop(imsize),
                    #transforms.RandomHorizontalFlip(),
                    #transforms.ColorJitter(brightness=0.01, contrast=0.01, saturation=0.01, hue=0.01),
                    transforms.ToTensor(),
                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                ])

        self.fixed_indices = []

        for _ in range(batch_size):
            id = np.random.randint(self.length)
            self.fixed_indices.append(id)

    def __len__(self):
        return self.length


    def fixed_batch(self):
        return torch.stack([self.random_batch(idx, True)[0].cuda() for idx in self.fixed_indices])


    def random_batch(self,index, fixed=False):

        file = str(index+1).zfill(6) + '.png'
        image_path = os.path.join(self.root, file )
        img = Image.open( image_path).convert('RGB')
        if fixed:
            img = self.fixed_transform(img)
        else:
            img = self.transform(img)

        return img, torch.zeros(1).long(), image_path
    
    def exact_batch(self, index):
        file = self.all_files[index]
        image_path = os.path.join(self.root, file)
        img = Image.open(image_path).convert('RGB')
        img = self.transform(img)
        return img, torch.zeros(1).long(), image_path

    def __getitem__(self,index):

        if self.return_all:
            return self.exact_batch(index)
        else:
            return self.random_batch(index)