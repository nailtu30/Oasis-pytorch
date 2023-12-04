import torch
import numpy as np
import time
import copy
import torch.nn as nn
from flcore.optimizers.fedoptimizer import PerturbedGradientDescent
from flcore.clients.clientbase import Client
import torch.nn.functional as F
from sklearn.preprocessing import label_binarize
from sklearn import metrics
from torch.autograd import Variable
import os


class clientDitto(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.mu = args.mu
        self.plocal_steps = args.plocal_steps

        self.pmodel_D = copy.deepcopy(self.model_D)
        self.pmodel_G = copy.deepcopy(self.model_G)

        self.loss = nn.BCELoss()
        lr = 0.0002
        self.optimizer_D = torch.optim.Adam(self.model_D.parameters(), lr=lr)
        self.optimizer_G = torch.optim.Adam(self.model_G.parameters(), lr=lr)
        self.poptimizer_D = PerturbedGradientDescent(
            self.pmodel_D.parameters(), lr=0.01, mu=self.mu)
        self.poptimizer_G = PerturbedGradientDescent(
            self.pmodel_G.parameters(), lr=0.01, mu=self.mu)
        self.results_dir = args.results_dir
    
    def save_pmodel(self):
        torch.save(self.pmodel_G.state_dict(), os.path.join(self.results_dir, 'pnetG{}.pth'.format(self.id)))

    def D_ptrain(self, x):
        #=======================Train the discriminator=======================#
        self.pmodel_D.zero_grad()

        # train discriminator on real
        x_real = x.view(-1, 784)
        bs = x_real.size(0)
        y_real = torch.ones(x_real.size(0), 1)
        x_real, y_real = Variable(x_real.to(self.device)), Variable(
            y_real.to(self.device))

        D_output = self.pmodel_D(x_real)
        D_real_loss = self.loss(D_output, y_real)
        D_real_score = D_output

        # train discriminator on facke
        z = Variable(torch.randn(bs, 100).to(self.device))
        x_fake, y_fake = self.pmodel_G(z), Variable(
            torch.zeros(bs, 1).to(self.device))

        D_output = self.pmodel_D(x_fake)
        D_fake_loss = self.loss(D_output, y_fake)
        D_fake_score = D_output

        # gradient backprop & optimize ONLY D's parameters
        D_loss = D_real_loss + D_fake_loss
        D_loss.backward()
        self.poptimizer_D.step(self.model_D.parameters(), self.device)

        return D_loss.data.item()

    def D_train(self, x):
        #=======================Train the discriminator=======================#
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
        self.optimizer_D.step()

        return D_loss.data.item()

    def G_ptrain(self, x):
        #=======================Train the generator=======================#
        self.pmodel_G.zero_grad()
        bs = x.size(0)
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.pmodel_G(z)
        D_output = self.pmodel_D(G_output)
        G_loss = self.loss(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.poptimizer_G.step(self.model_G.parameters(), self.device)

        return G_loss.data.item()

    def G_train(self, x):
        #=======================Train the generator=======================#
        self.model_G.zero_grad()
        bs = x.size(0)
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.model_G(z)
        D_output = self.model_D(G_output)
        G_loss = self.loss(D_output, y)

        # gradient backprop & optimize ONLY G's parameters
        G_loss.backward()
        self.optimizer_G.step()

        return G_loss.data.item()

    # def G_eval(self):
    #     bs = 32
    #     z = Variable(torch.randn(bs, 100).to(self.device))
    #     y = Variable(torch.ones(bs, 1).to(self.device))

    #     G_output = self.model_G(z)
    #     D_output = self.model_D(G_output)
    #     G_loss = self.loss(D_output, y)
    #     return G_loss.data.item()

    def train(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model_D.train()
        self.model_G.train()

        max_local_steps = self.local_steps
        # if self.train_slow:
        #     max_local_steps = np.random.randint(1, max_local_steps // 2)

        for step in range(max_local_steps):
            D_losses, G_losses = [], []
            for batch_idx, (x, _) in enumerate(trainloader):
                D_loss = self.D_train(x)
                G_loss = self.G_train(x)
                D_losses.append(D_loss)
                G_losses.append(G_loss)

            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((step),
                  max_local_steps, np.mean(D_losses), np.mean(G_losses)))

        # self.model.cpu()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # self.model_G.eval()
        # self.model_D.eval()
        # G_loss = self.G_eval()
        return D_loss, G_loss

    def ptrain(self):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.pmodel_D.train()
        self.pmodel_G.train()

        max_local_steps = self.plocal_steps

        for step in range(max_local_steps):
            D_losses, G_losses = [], []
            for batch_idx, (x, _) in enumerate(trainloader):
                D_loss = self.D_ptrain(x)
                G_loss = self.G_ptrain(x)
                D_losses.append(D_loss)
                G_losses.append(G_loss)

            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((step),
                  max_local_steps, np.mean(D_losses), np.mean(G_losses)))

        # self.model.cpu()

        self.train_time_cost['total_cost'] += time.time() - start_time
        return D_loss, G_loss
