import time
from flcore.clients.clientavg_unetgan_celeba import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import csv
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from flcore.servers.generate_celeba import split_data, Celeba
import shutil
from torchvision import datasets, transforms, utils
from torch.utils.data import DataLoader

class FedAvgUNetGANCeleba(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        # self.set_slow_clients()
        self.set_clients(args, clientAVG)

        # print(
        #     f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print(
            f"\nJoin clients / total clients: {self.join_clients} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def create_clients_datasets(self, split=False):
        root_path = '/data/celeba'
        if split:
            clients_file_indexes = split_data(num_clients=self.num_clients)
            for client_index, client_file_indexes in enumerate(clients_file_indexes):
                print('client: {}'.format(client_index))
                client_dir_path = os.path.join(root_path, 'train', str(client_index))
                if os.path.exists(client_dir_path):
                    shutil.rmtree(client_dir_path)
                os.mkdir(client_dir_path)
                for file_index in client_file_indexes:
                    file_index = str(file_index)
                    file_name = file_index.rjust(6, '0') + '.png'
                    target = os.path.join(client_dir_path, file_name)
                    source = os.path.join(root_path, 'img_align_celeba_png', file_name)
                    shutil.copy(source, target)
        resolution = 128
        transform = transforms.Compose(
            [
                transforms.Resize(resolution),
                transforms.CenterCrop(resolution),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
            ]
        )
        datasets = []
        for index in range(self.num_clients):
            client_root_path = os.path.join(root_path, 'train', str(index))
            dataset = Celeba(root = client_root_path, transform = transform, batch_size = self.batch_size, imsize = resolution, return_all=True)
            datasets.append(dataset)
        return datasets
    
    def set_clients(self, args, clientObj):
        datasets = self.create_clients_datasets()
        # print(datasets)
        for i in range(self.num_clients):
            train_data = datasets[i]
            client = clientObj(args,
                               id=i,
                               train_data=train_data,
                               train_samples=len(train_data))
            self.clients.append(client)

    def train(self):
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            D_losses = []
            G_losses = []
            for client in self.selected_clients:
                D_loss, G_loss = client.train()
                D_losses.append(D_loss)
                G_losses.append(G_loss)
            print(D_losses)
            print(G_losses)
            with open(os.path.join(self.results_dir, 'D_loss.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(D_losses)
            with open(os.path.join(self.results_dir, 'G_loss.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(G_losses)
            self.receive_models()
            self.aggregate_parameters()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])
            torch.save(self.global_model_G.state_dict(),
                   os.path.join(self.results_dir, 'global_netG.pth'))

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
