import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable
import csv
import os
import copy
from utils.kmeans import Encoder
from sklearn.decomposition import FastICA


class clientSim(Client):
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
        self.data_size = self.get_data_size()
        # self.gamma = args.gamma
        self.orginal_gamma = args.gamma

    def get_data_size(self):
        loader = self.load_full_batch_data()
        for batch_idx, (x, _) in enumerate(loader):
            data_size = x.size(0)
            break
        return data_size

    def extract_representation(self):
        encoder = Encoder()
        encoder.load_state_dict(torch.load('flcore/clients/enc.pth'))
        encoder.to(device=self.device)
        trainloader = self.load_full_batch_data()
        for batch_idx, (x, _) in enumerate(trainloader):
            x = x.to(self.device)
            with torch.no_grad():
                x = encoder(x)
            x = x.reshape(-1)
            break
        x = x.cpu().numpy()
        return x

    def discriminator_regularizer(self, critic, D1_args, D2_args):
        '''
        JS-Regularizer

        A Methode that regularize the gradient when discriminate real and fake. 
        This methode was proposed to deal with the problem of the choice of the 
        careful choice of architecture, paremeter intializaton and selction 
        of hyperparameters.

        GAN is highly sensitive to the choice of the latters. 
        According to "Stabilizing Training of Generative Adversarial Networks 
        through Regularization", Roth and al., This fragility is due to the mismathch or 
        non-overlapping support between the model distribution and the data distribution.
        :param critic : Discriminator network,
        :param D1_args : real value
        :param D2_args : fake value
        '''
        BATCH_SIZE, *others = D1_args.shape
        DEVICE = D1_args.device

        D1_args = Variable(D1_args, requires_grad=True)
        D2_args = Variable(D2_args, requires_grad=True)
        D1_logits, D2_logits = critic(D1_args), critic(D2_args)
        D1, D2 = torch.sigmoid(D1_logits), torch.sigmoid(D2_logits)

        grad_D1_logits = torch.autograd.grad(outputs=D1_logits, inputs=D1_args,
                                             create_graph=True, retain_graph=True,
                                             grad_outputs=torch.ones(D1_logits.size()).to(DEVICE))[0]

        grad_D2_logits = torch.autograd.grad(outputs=D2_logits, inputs=D2_args,
                                             create_graph=True, retain_graph=True,
                                             grad_outputs=torch.ones(D2_logits.size()).to(DEVICE))[0]

        grad_D1_logits_norm = torch.norm(torch.reshape(grad_D1_logits, (BATCH_SIZE, -1)),
                                         dim=-1, keepdim=True)

        grad_D2_logits_norm = torch.norm(torch.reshape(grad_D2_logits, (BATCH_SIZE, -1)),
                                         dim=-1, keepdim=True)

        assert grad_D1_logits_norm.shape == D1.shape
        assert grad_D2_logits_norm.shape == D2.shape

        reg_D1 = torch.multiply(torch.square(
            1. - D1), torch.square(grad_D1_logits_norm))
        reg_D2 = torch.multiply(torch.square(
            D2), torch.square(grad_D2_logits_norm))
        discriminator_regularizer = torch.sum(reg_D1 + reg_D2).mean()
        return discriminator_regularizer

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

        disc_reg = self.discriminator_regularizer(self.model_D, x_real, x_fake)

        if D_loss.data.item() > 10:
            print('SSSSSSs: {}'.format(D_loss.data.item()))
            print('reg loss: {}'.format(disc_reg))

        D_loss = D_loss + (self.gamma / 2.0) * disc_reg

        D_loss.backward()

        # if D_loss.data.item() > 10:
        #     print('disc reg: {}'.format(disc_reg))

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
        self.able_avg = True
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model_D.train()
        self.model_G.train()

        max_local_steps = self.local_steps
        # if self.train_slow:
        #     max_local_steps = np.random.randint(1, max_local_steps // 2)

        print('client {} starts training'.format(self.id))

        for epoch in range(1, max_local_steps+1):
            D_losses, G_losses = [], []
            self.gamma = self.orginal_gamma * \
                np.power(0.01, (epoch - 1) / max_local_steps)
            for batch_idx, (x, _) in enumerate(trainloader):
                D_loss = self.D_train(x)
                G_loss = self.G_train(x)
                D_losses.append(D_loss)
                G_losses.append(G_loss)
                # if D_loss > 10 or G_loss > 10:
                #     self.able_avg = False
                #     break
            if not self.able_avg:
                print('Unable Avg!')
                break
            print('gamma: {}'.format(self.gamma))
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
        return D_loss, G_loss, self.able_avg
