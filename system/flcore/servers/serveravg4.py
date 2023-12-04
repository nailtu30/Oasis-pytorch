import time
from flcore.clients.clientavg4 import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import csv
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.data_utils import read_client_data
import copy


class FedAvg4(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        # self.set_slow_clients()
        self.client_samples = dict()
        self.set_clients(args, clientAVG)

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

        lr = 0.01
        self.lr = lr
        self.loss = nn.BCELoss()
        self.global_G_optimizer = torch.optim.SGD(
            self.global_model_G.parameters(), lr=lr)
        self.global_D_optimizer = torch.optim.SGD(
            self.global_model_D.parameters(), lr=lr)

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            # test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data))
            self.clients.append(client)
            self.client_samples[i] = len(train_data)

    def send_coefficent(self):
        total_samples = 0
        coefficent_arr = []
        for client in self.selected_clients:
            total_samples += self.client_samples[client.id]
        for client in self.selected_clients:
            coefficent = self.client_samples[client.id] / total_samples
            coefficent_arr.append(coefficent)
            client.set_coefficent(coefficent)
        print(coefficent_arr)
        self.coefficent_arr = coefficent_arr

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_ids = []
        self.uploaded_models_D = []
        self.uploaded_models_G = []
        for client in self.selected_clients:
            self.uploaded_ids.append(client.id)
            self.uploaded_models_D.append(client.model_D)
            self.uploaded_models_G.append(client.model_G)

    def aggregate_parameters(self):
        # assert (len(self.uploaded_models_G) > 0)
        assert (len(self.uploaded_models_D) > 0)

        self.global_model_D = copy.deepcopy(self.uploaded_models_D[0])
        self.global_model_G = copy.deepcopy(self.uploaded_models_G[0])
        for param in self.global_model_D.parameters():
            param.data.zero_()

        for param in self.global_model_G.parameters():
            param.data.zero_()

        for w, client_model in zip(self.coefficent_arr, self.uploaded_models_D):
            self.add_parameters_D(w, client_model)

        for w, client_model in zip(self.coefficent_arr, self.uploaded_models_G):
            self.add_parameters_G(w, client_model)

    def G_eval(self):
        bs = 32
        criterian = nn.BCELoss()
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.global_model_G(z)
        D_output = self.global_model_D(G_output)
        G_loss = criterian(D_output, y)
        return G_loss.data.item()

    def global_interact(self):
        print('========global interact=======')
        self.global_model_D.train()
        self.global_model_G.train()
        self.global_model_D.zero_grad()
        self.global_model_G.zero_grad()
        # train global G
        for i in range(5):
            bs = 32
            z = Variable(torch.randn(bs, 100).to(self.device))
            y = Variable(torch.ones(bs, 1).to(self.device))

            G_output = self.global_model_G(z)
            D_output = self.global_model_D(G_output)
            G_loss = self.loss(D_output, y)
            G_loss.backward()
            self.global_G_optimizer.step()
            print(G_loss)

        self.global_model_D.zero_grad()
        self.global_model_G.zero_grad()

    def train(self):
        # first, evaluate init model
        # self.global_model_G.eval()
        # self.global_model_D.eval()
        # G_loss = self.G_eval()
        # with open(os.path.join(self.results_dir, 'train_loss.csv'), 'a') as csvfile:
        #     writer = csv.writer(csvfile)
        #     writer.writerow([G_loss] * 10)
        # self.global_model_G.train()
        # self.global_model_D.train()
        for i in range(self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            self.send_coefficent()
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
            # self.aggregate_parameters2(i)
            self.aggregate_parameters()

            self.global_interact()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        torch.save(self.global_model_G.state_dict(),
                   os.path.join(self.results_dir, 'global_netG.pth'))
