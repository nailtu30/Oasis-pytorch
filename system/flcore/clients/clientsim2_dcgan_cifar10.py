import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable
import csv
import os
import copy
from utils.kmeans import AutoencoderCifar10
from sklearn.decomposition import FastICA
from torchvision import transforms

class clientSim(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        self.loss = nn.BCELoss()
        lr = 0.002
        self.lr = lr
        momentum = 0.9
        # self.G_optimizer = torch.optim.Adam(self.model_G.parameters(), lr=lr)
        # self.D_optimizer = torch.optim.Adam(self.model_D.parameters(), lr=lr)
        self.D_optimizer = torch.optim.Adam(
            self.model_D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(
            self.model_G.parameters(), lr=lr, betas=(0.5, 0.999))
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
        autoencoder = AutoencoderCifar10().to(device=self.device)
        autoencoder.load_state_dict(torch.load('flcore/clients/autoencoder_cifar10.pth'))
        encoder = autoencoder.encoder
        trainloader = self.load_full_batch_data()
        for batch_idx, (x, _) in enumerate(trainloader):
            x = x.to(self.device)
            with torch.no_grad():
                x = encoder(x)
        x = x.cpu().numpy()
        x = np.mean(x, axis=0)
        x = np.mean(x, axis=0)
        return x.reshape(-1)

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
        
        D1 = D1.reshape(D1.size(0), 1)
        D2 = D2.reshape(D2.size(0), 1)

        assert grad_D1_logits_norm.shape == D1.shape
        assert grad_D2_logits_norm.shape == D2.shape

        reg_D1 = torch.multiply(torch.square(
            1. - D1), torch.square(grad_D1_logits_norm))
        reg_D2 = torch.multiply(torch.square(
            D2), torch.square(grad_D2_logits_norm))
        discriminator_regularizer = torch.sum(reg_D1 + reg_D2).mean()
        return discriminator_regularizer

    def D_train(self, x):
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
        errD_real.backward()
        # train with fake
        noise = torch.randn(batch_size, 100, 1, 1, device=self.device)
        fake = self.model_G(noise)
        label = torch.full((batch_size,), 0.0, device=self.device)
        output = self.model_D(fake.detach())
        errD_fake = self.loss(output, label)
        errD_fake.backward()
        D_loss = errD_real + errD_fake
        disc_reg = self.discriminator_regularizer(self.model_D, x, fake)
        D_loss = D_loss + (self.gamma / 2.0) * disc_reg
        self.D_optimizer.step()
        return D_loss.item()

    def G_train(self, x):
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
        G_loss.backward()
        self.G_optimizer.step()
        return G_loss.item()


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
            self.gamma = self.orginal_gamma
            # self.gamma = self.orginal_gamma * \
            #     np.power(0.01, (epoch - 1) / max_local_steps)
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
