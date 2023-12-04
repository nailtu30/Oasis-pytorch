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
        self.agent_D = copy.deepcopy(args.model_D)
        self.agent_G = copy.deepcopy(args.model_G)
        lr = 0.0002
        self.lr = lr
        self.D_optimizer = torch.optim.Adam(
            self.model_D.parameters(), lr=lr, betas=(0.5, 0.999))
        self.G_optimizer = torch.optim.Adam(
            self.model_G.parameters(), lr=lr, betas=(0.5, 0.999))

        self.results_dir = args.results_dir

    def set_agent_D(self, model_D):
        for new_param, old_param in zip(model_D.parameters(), self.agent_D.parameters()):
            old_param.data = new_param.data.clone()

    def set_agent_G(self, model_G):
        for new_param, old_param in zip(model_G.parameters(), self.agent_G.parameters()):
            old_param.data = new_param.data.clone()

    def D_train(self, x):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real
        old_D = copy.deepcopy(self.model_D)
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

        K = -1.1 * self.lr
        extra_loss = 0
        num_layer = 0
        for p, ap in zip(old_D.parameters(), self.agent_D.parameters()):
            num_layer = num_layer + 1
            tmp = p - ap
            if len(tmp.shape) == 1:
                tmp = tmp.reshape(tmp.size(0), 1)
                K_eye = K * torch.eye(tmp.size(0)).to(self.device)
                extra_loss = extra_loss + torch.mean(K_eye.mm(tmp))
            elif len(tmp.shape) == 4:
                num_channel = tmp.size(1)
                channel_loss = 0
                for i in range(num_channel):
                    channel = tmp[:, i, :, :]
                    channel = channel.reshape(channel.size(0), -1)
                    K_eye = K * torch.eye(channel.size(0)).to(self.device)
                    channel_loss += torch.mean(K_eye.mm(channel))
                channel_loss = channel_loss / num_channel
                extra_loss = extra_loss + channel_loss
            else:
                raise RuntimeError(
                    'No implement of Handling Shape: {}'.format(tmp.shape))
            del K_eye
            del tmp
        extra_loss = extra_loss / num_layer
        D_loss = D_loss + extra_loss
        D_loss.backward()
        self.D_optimizer.step()

        P = 0.3 * self.lr
        for p, ap in zip(old_D.parameters(), self.agent_D.parameters()):
            # ap = ap + P * (p - ap)
            tmp = p - ap
            if len(tmp.shape) == 1:
                tmp = tmp.reshape(tmp.size(0), 1)
                P_eye = P * torch.eye(tmp.size(0)).to(self.device)
                ap = ap + P_eye.mm(tmp).reshape(ap.shape)
            elif len(tmp.shape) == 4:
                num_channel = tmp.size(1)
                delta = None
                for i in range(num_channel):
                    channel = tmp[:, i, :, :]
                    channel = channel.reshape(channel.size(0), -1)
                    P_eye = P * torch.eye(channel.size(0)).to(self.device)
                    res = P_eye.mm(channel)
                    res = res.reshape(tmp.size(0), 1, tmp.size(2), tmp.size(3))
                    if delta is None:
                        delta = res
                    else:
                        delta = torch.cat((delta, res), dim=1)

                ap = ap + delta
            else:
                raise RuntimeError(
                    'No implement of Handling Shape: {}'.format(tmp.shape))

            del P_eye
            del tmp

        return D_loss.item()

    def G_train(self, x):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################
        old_G = copy.deepcopy(self.model_G)
        self.model_G.zero_grad()
        bs = x.size(0)
        # fake labels are real for generator cost
        noise = torch.randn(bs, 100, 1, 1, device=self.device)
        fake = self.model_G(noise)
        label = torch.full((bs,), 1.0, device=self.device)
        output = self.model_D(fake)
        G_loss = self.loss(output, label)

        K = -1.1 * self.lr
        extra_loss = 0
        num_layer = 0
        for p, ap in zip(old_G.parameters(), self.agent_G.parameters()):
            # G_loss = G_loss + K * torch.mean(p - ap)
            num_layer = num_layer + 1
            tmp = p - ap
            if len(tmp.shape) == 1:
                tmp = tmp.reshape(tmp.size(0), 1)
                K_eye = K * torch.eye(tmp.size(0)).to(self.device)
                extra_loss = extra_loss + torch.mean(K_eye.mm(tmp))
            elif len(tmp.shape) == 4:
                num_channel = tmp.size(1)
                channel_loss = 0
                for i in range(num_channel):
                    channel = tmp[:, i, :, :]
                    channel = channel.reshape(channel.size(0), -1)
                    K_eye = K * torch.eye(channel.size(0)).to(self.device)
                    channel_loss += torch.mean(K_eye.mm(channel))
                channel_loss = channel_loss / num_channel
                extra_loss = extra_loss + channel_loss
            else:
                raise RuntimeError(
                    'No implement of Handling Shape: {}'.format(tmp.shape))
            del K_eye
            del tmp
        extra_loss = extra_loss / num_layer
        G_loss = G_loss + extra_loss
        G_loss.backward()
        self.G_optimizer.step()

        P = 0.3 * self.lr
        for p, ap in zip(old_G.parameters(), self.agent_G.parameters()):
            # ap = ap + P * (p - ap)
            tmp = p - ap
            if len(tmp.shape) == 1:
                tmp = tmp.reshape(tmp.size(0), 1)
                P_eye = P * torch.eye(tmp.size(0)).to(self.device)
                ap = ap + P_eye.mm(tmp).reshape(ap.shape)
            elif len(tmp.shape) == 4:
                num_channel = tmp.size(1)
                delta = None
                for i in range(num_channel):
                    channel = tmp[:, i, :, :]
                    channel = channel.reshape(channel.size(0), -1)
                    P_eye = P * torch.eye(channel.size(0)).to(self.device)
                    res = P_eye.mm(channel)
                    res = res.reshape(tmp.size(0), 1, tmp.size(2), tmp.size(3))
                    if delta is None:
                        delta = res
                    else:
                        delta = torch.cat((delta, res), dim=1)
                ap = ap + delta
            else:
                raise RuntimeError(
                    'No implement of Handling Shape: {}'.format(tmp.shape))
            del P_eye
            del tmp

        return G_loss.item()

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
