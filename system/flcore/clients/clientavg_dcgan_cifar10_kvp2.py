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

    def set_parameters_D(self, model_D):
        # self.model_D = copy.deepcopy(model_D)
        for new_param, old_param in zip(model_D.parameters(), self.model_D.parameters()):
            old_param.data = new_param.data.clone()

    def set_parameters_G(self, model_G):
        # self.model_G = copy.deepcopy(model_G)
        for new_param, old_param in zip(model_G.parameters(), self.model_G.parameters()):
            old_param.data = new_param.data.clone()

    def set_agent_D(self, model_D):
        # self.agent_D = copy.deepcopy(model_D)
        for new_param, old_param in zip(model_D.parameters(), self.agent_D.parameters()):
            old_param.data = new_param.data.clone()

    def set_agent_G(self, model_G):
        # self.agent_G = copy.deepcopy(model_G)
        for new_param, old_param in zip(model_G.parameters(), self.agent_G.parameters()):
            old_param.data = new_param.data.clone()

    @torch.no_grad()
    def helper1(self, K, old_model, agent_model, current_model):
        for old_param, agent_param, current_param in zip(old_model.parameters(), agent_model.parameters(), current_model.parameters()):
            diff_param = old_param.data - agent_param.data
            current_param.data = current_param.data + K * diff_param
            del diff_param

    @torch.no_grad()
    def helper11(self, K, old_model, agent_model, current_model):
        for old_param, agent_param, current_param in zip(old_model.parameters(), agent_model.parameters(), current_model.parameters()):
            diff_param = old_param.data - agent_param.data
            if len(diff_param.shape) == 1:
                diff_param = diff_param.reshape(diff_param.size(0), 1)
                K_eye = K * torch.eye(diff_param.size(0)).to(self.device)
                diff_param = K_eye.mm(diff_param)
                diff_param = diff_param.reshape(-1)
            elif len(diff_param.shape) == 4:
                new_diff_param = None
                n_channel = diff_param.size(1)
                # print(diff_param.shape)
                for c in range(n_channel):
                    c_diff_param = diff_param[:, c, :, :]
                    c_diff_param = c_diff_param.reshape(diff_param.size(0), -1)
                    K_eye = K * torch.eye(c_diff_param.size(0)).to(self.device)
                    c_diff_param = K_eye.mm(c_diff_param)
                    c_diff_param = c_diff_param.reshape(diff_param.size(
                        0), 1, diff_param.size(2), diff_param.size(3))
                    if new_diff_param is None:
                        new_diff_param = c_diff_param
                    else:
                        new_diff_param = torch.concat(
                            (new_diff_param, c_diff_param), dim=1)
                diff_param = new_diff_param
            current_param.data = current_param.data + diff_param
            del diff_param

    @torch.no_grad()
    def helper2(self, P, old_model, agent_model):
        for old_param, agent_param in zip(old_model.parameters(), agent_model.parameters()):
            diff_param = old_param.data - agent_param.data
            agent_param.data = agent_param.data + P * diff_param
            del diff_param

    @torch.no_grad()
    def helper21(self, P, old_model, agent_model):
        for old_param, agent_param in zip(old_model.parameters(), agent_model.parameters()):
            diff_param = old_param.data - agent_param.data
            if len(diff_param.shape) == 1:
                diff_param = diff_param.reshape(diff_param.size(0), 1)
                P_eye = P * torch.eye(diff_param.size(0)).to(self.device)
                diff_param = P_eye.mm(diff_param)
                diff_param = diff_param.reshape(-1)
            elif len(diff_param.shape) == 4:
                new_diff_param = None
                n_channel = diff_param.size(1)
                for c in range(n_channel):
                    c_diff_param = diff_param[:, c, :, :]
                    c_diff_param = c_diff_param.reshape(diff_param.size(0), -1)
                    P_eye = P * torch.eye(c_diff_param.size(0)).to(self.device)
                    c_diff_param = P_eye.mm(c_diff_param)
                    c_diff_param = c_diff_param.reshape(diff_param.size(
                        0), 1, diff_param.size(2), diff_param.size(3))
                    if new_diff_param is None:
                        new_diff_param = c_diff_param
                    else:
                        new_diff_param = torch.concat(
                            (new_diff_param, c_diff_param), dim=1)
                diff_param = new_diff_param
            agent_param.data = agent_param.data + diff_param
            del diff_param

    def D_train(self, x):
        ############################
        # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        ###########################
        # train with real

        self.model_D.zero_grad()
        self.agent_D.zero_grad()
        old_D = copy.deepcopy(self.model_D)
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
        D_loss.backward()
        self.D_optimizer.step()

        K = self.lr * (-1.1)
        self.helper11(K, old_D, self.agent_D, self.model_D)
        # for old_param, agent_param, current_param in zip(old_D.parameters(), self.agent_D.parameters(), self.model_D.parameters()):
        #     diff_param = old_param.data - agent_param.data
        #     current_param.data = current_param.data + K * diff_param
        #     del diff_param

        P = self.lr * 0.3
        self.helper21(P, old_D, self.agent_D)
        # for old_param, agent_param in zip(old_D.parameters(), self.agent_D.parameters()):
        #     diff_param = old_param.data - agent_param.data
        #     agent_param.data = agent_param.data + P * diff_param
        #     del diff_param
        del old_D
        self.model_D.zero_grad()
        self.agent_D.zero_grad()
        return D_loss.item()

    def G_train(self, x):
        ############################
        # (2) Update G network: maximize log(D(G(z)))
        ###########################

        self.model_G.zero_grad()
        self.agent_G.zero_grad()
        old_G = copy.deepcopy(self.model_G)
        bs = x.size(0)
        # fake labels are real for generator cost
        noise = torch.randn(bs, 100, 1, 1, device=self.device)
        fake = self.model_G(noise)
        label = torch.full((bs,), 1.0, device=self.device)
        output = self.model_D(fake)
        G_loss = self.loss(output, label)
        G_loss.backward()
        self.G_optimizer.step()

        K = self.lr * (-1.1)
        self.helper11(K, old_G, self.agent_G, self.model_G)
        # for old_param, agent_param, current_param in zip(old_G.parameters(), self.agent_G.parameters(), self.model_G.parameters()):
        #     diff_param = old_param.data - agent_param.data
        #     current_param.data = current_param.data + K * diff_param
        #     del diff_param

        P = self.lr * 0.3
        self.helper21(P, old_G, self.agent_G)
        # for old_param, agent_param in zip(old_G.parameters(), self.agent_G.parameters()):
        #     diff_param = old_param.data - agent_param.data
        #     agent_param.data = agent_param.data + P * diff_param
        #     del diff_param
        del old_G
        self.model_G.zero_grad()
        self.agent_G.zero_grad()
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
