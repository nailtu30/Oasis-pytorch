import time
from flcore.clients.clientavg5 import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import csv
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
import copy


class FedAvg5(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        # self.set_slow_clients()
        self.set_clients(args, clientAVG)

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models_D = []
        self.uploaded_models_G = []
        for client in self.selected_clients:
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models_D.append(client.model_D)
            self.uploaded_models_G.append(client.model_G)

    def aggregate_parameters2(self):
        index = np.argmin(self.D_weights)
        self.global_model_D = copy.deepcopy(self.uploaded_models_D[index])
        index = np.argmax(self.G_weights)
        self.global_model_G = copy.deepcopy(self.uploaded_models_G[index])

    def aggregate_parameters(self):
        # assert (len(self.uploaded_models_G) > 0)
        assert (len(self.uploaded_models_D) > 0)

        self.global_model_D = copy.deepcopy(self.uploaded_models_D[0])
        self.global_model_G = copy.deepcopy(self.uploaded_models_G[0])
        for param in self.global_model_D.parameters():
            param.data.zero_()

        for param in self.global_model_G.parameters():
            param.data.zero_()

        for w, client_model in zip(self.G_weights, self.uploaded_models_D):
            self.add_parameters_D(w, client_model)

        for w, client_model in zip(self.D_weights, self.uploaded_models_G):
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

    def softmax(self):
        self.D_weights = []
        exps = []
        for r_D_loss in self.r_D_losses:
            exps.append(np.exp(r_D_loss))
        for e in exps:
            self.D_weights.append(e / np.sum(exps))
        print('weights: {}'.format(self.D_weights))

        self.G_weights = []
        exps = []
        for r_G_loss in self.r_G_losses:
            exps.append(np.exp(r_G_loss))
        for e in exps:
            self.G_weights.append(e / np.sum(exps))
        print('weights: {}'.format(self.G_weights))

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
            D_losses = []
            G_losses = []
            r_D_losses = []
            r_G_losses = []
            for client in self.selected_clients:
                D_loss, G_loss, r_D_loss, r_G_loss = client.train()
                D_losses.append(D_loss)
                G_losses.append(G_loss)
                r_D_losses.append(r_D_loss)
                r_G_losses.append(r_G_loss)
            print(D_losses)
            print(G_losses)
            self.r_D_losses = r_D_losses
            self.r_G_losses = r_G_losses
            self.softmax()
            with open(os.path.join(self.results_dir, 'D_loss.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(D_losses)
            with open(os.path.join(self.results_dir, 'G_loss.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(G_losses)
            self.receive_models()
            # self.aggregate_parameters2(i)
            # self.aggregate_parameters()
            self.aggregate_parameters2()

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        torch.save(self.global_model_G.state_dict(),
                   os.path.join(self.results_dir, 'global_netG.pth'))
