import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable
import csv
import os
import copy


class clientAVG(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.loss = nn.BCELoss()
        lr = 0.001
        self.lr = lr
        momentum = 0.9
        # self.G_optimizer = torch.optim.Adam(self.model_G.parameters(), lr=lr)
        # self.D_optimizer = torch.optim.Adam(self.model_D.parameters(), lr=lr)
        self.G_optimizer = torch.optim.SGD(
            self.model_G.parameters(), lr=lr, momentum=momentum)
        self.D_optimizer = torch.optim.SGD(
            self.model_D.parameters(), lr=lr, momentum=momentum)
        self.results_dir = args.results_dir

    def set_coefficent(self, coefficent):
        self.coefficent = coefficent

    def D_train(self, x):
        #=======================Train the discriminator=======================#
        old_D = copy.deepcopy(self.model_D)
        self.model_D.zero_grad()

        # train discriminator on real
        x_real = x.view(-1, 784)
        bs = x_real.size(0)
        y_real = torch.ones(x_real.size(0), 1)
        x_real, y_real = Variable(x_real.to(self.device)), Variable(
            y_real.to(self.device))

        D_output = self.model_D(x_real)
        D_real_loss = self.loss(D_output, y_real)
        D_real_score = D_output

        # train discriminator on facke
        z = Variable(torch.randn(bs, 100).to(self.device))
        x_fake, y_fake = self.model_G(z), Variable(
            torch.zeros(bs, 1).to(self.device))

        D_output = self.model_D(x_fake)
        D_fake_loss = self.loss(D_output, y_fake)
        D_fake_score = D_output

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss

        D_loss.backward()
        for param in self.model_D.parameters():
            param.grad = param.grad * self.coefficent

        self.D_optimizer.step()

        return D_loss.data.item()

    def G_train(self, x):
        #=======================Train the generator=======================#
        old_G = copy.deepcopy(self.model_G)
        self.model_G.zero_grad()
        bs = x.size(0)
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.model_G(z)
        D_output = self.model_D(G_output)
        G_loss = self.loss(D_output, y)
        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.G_optimizer.step()

        return G_loss.data.item()

    def G_eval(self):
        bs = 32
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.model_G(z)
        D_output = self.model_D(G_output)
        G_loss = self.loss(D_output, y)
        return G_loss.data.item()

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model_D.train()
        self.model_G.train()

        max_local_steps = self.local_steps
        # if self.train_slow:
        #     max_local_steps = np.random.randint(1, max_local_steps // 2)

        for epoch in range(1, max_local_steps+1):
            D_losses, G_losses = [], []
            for batch_idx, (x, _) in enumerate(trainloader):
                D_loss = self.D_train(x)
                G_loss = self.G_train(x)
                D_losses.append(D_loss)
                G_losses.append(G_loss)

            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch),
                  max_local_steps, np.mean(D_losses), np.mean(G_losses)))
            # with open(os.path.join(self.results_dir, '{}_train_loss.csv'.format(self.id)), 'a') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow([np.mean(D_losses), np.mean(G_losses)])

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # self.model_G.eval()
        # self.model_D.eval()
        # G_loss = self.G_eval()
        return D_loss, G_loss
