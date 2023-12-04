import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable
import csv
import os
import copy
from torch.optim import Optimizer


class clientAVG(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.loss = nn.BCELoss()
        lr = 0.0002
        self.lr = lr
        self.D_optimizer = torch.optim.Adam(
            self.model_D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(
            self.model_G.parameters(), lr=lr, betas=(0.5, 0.999))
        self.init_model_G = copy.deepcopy(self.model_G)
        self.init_model_D = copy.deepcopy(self.model_D)

        self.results_dir = args.results_dir

    @torch.no_grad()
    def helper1(self, init_model, model, T):
        if T == 1:
            return
        for init_param, param in zip(init_model.parameters(), model.parameters()):
            diff_param = init_param - param
            norm_param = torch.norm(diff_param, p=2, dim=0)
            # extra_grad = norm_param / (2 * self.lr * (T-1))
            norm_param_square = torch.pow(norm_param, 2)
            extra_grad = norm_param_square / (2 * self.lr * (T-1))
            param.grad.data = param.grad.data + extra_grad

    @torch.no_grad()
    def helper2(self, init_model, model, T):
        proximal_item = 0.0
        for init_param, param in zip(init_model.parameters(), model.parameters()):
            diff_param = init_param - param
            proximal_item += diff_param.norm(2)
        return proximal_item

    def D_train(self, x, T):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        self.model_D.zero_grad()
        x = x.to(self.device)
        batch_size = x.size(0)
        label = torch.full((batch_size,), 1.0, device=self.device)

        output = self.model_D(x)
        errD_real = self.loss(output, label)
        # errD_real.backward()
        # train with fake
        noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
        fake = self.model_G(noise)
        label = torch.full((batch_size,), 0.0, device=self.device)
        output = self.model_D(fake.detach())
        errD_fake = self.loss(output, label)

        # errD_fake.backward()
        D_loss = errD_real + errD_fake

        # if T > 1:
        #     proximal_item = self.helper2(self.init_model_D, self.model_D, T)
        #     D_loss = D_loss + proximal_item / (2 * self.lr * (T - 1))

        D_loss.backward()
        self.helper1(self.init_model_D, self.model_D, T)
        self.D_optimizer.step()

        return D_loss.item()

    def G_train(self, x, T):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        self.model_G.zero_grad()
        bs = x.size(0)
        # fake labels are real for generator cost
        noise = torch.randn(bs, 100, 1, 1, device=self.device)
        fake = self.model_G(noise)
        label = torch.full((bs,), 1.0, device=self.device)
        output = self.model_D(fake)
        G_loss = self.loss(output, label)

        # if T > 1:
        #     proximal_item = self.helper2(self.init_model_G, self.model_G, T)
        #     G_loss = G_loss + proximal_item / (2 * self.lr * (T - 1))

        G_loss.backward()
        self.helper1(self.init_model_G, self.model_G, T)
        self.G_optimizer.step()

        return G_loss.item()

    def train(self, T):
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
                D_loss = self.D_train(x, T)
                G_loss = self.G_train(x, T)
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
