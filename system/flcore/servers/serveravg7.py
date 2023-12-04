import time
from flcore.clients.clientavg7 import clientAVG
from flcore.servers.serverbase import Server
from threading import Thread
import csv
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable


class FedAvg7(Server):
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

    def G_eval(self):
        bs = 32
        criterian = nn.BCELoss()
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.global_model_G(z)
        D_output = self.global_model_D(G_output)
        G_loss = criterian(D_output, y)
        return G_loss.data.item()

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

            self.Budget.append(time.time() - s_t)
            print('-'*25, 'time cost', '-'*25, self.Budget[-1])

        print("\nAverage time cost per round.")
        print(sum(self.Budget[1:])/len(self.Budget[1:]))
        torch.save(self.global_model_G.state_dict(),
                   os.path.join(self.results_dir, 'global_netG.pth'))
