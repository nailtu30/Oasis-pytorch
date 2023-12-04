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
        self.agent_D = copy.deepcopy(args.model_D)
        self.agent_G = copy.deepcopy(args.model_G)

    def set_agent_D(self, model_D):
        for new_param, old_param in zip(model_D.parameters(), self.agent_D.parameters()):
            old_param.data = new_param.data.clone()

    def set_agent_G(self, model_G):
        for new_param, old_param in zip(model_G.parameters(), self.agent_G.parameters()):
            old_param.data = new_param.data.clone()

    @torch.no_grad()
    def helper1(self, K, current_model, agent_model):
        res = 0.0
        for agent_param, current_param in zip(agent_model.parameters(), current_model.parameters()):
            # diff_param = old_param.data - agent_param.data
            diff_param = current_param.data - agent_param.data
            # print(diff_param.shape)
            if len(diff_param.shape) == 1:
                diff_param = diff_param.reshape(diff_param.size(0), 1)
                K_eye = K * torch.eye(diff_param.size(0)).to(self.device)
                diff_param = K_eye.mm(diff_param)
                diff_param = diff_param.reshape(-1)
            elif len(diff_param.shape) == 2:
                K_eye = K * torch.eye(diff_param.size(0)).to(self.device)
                diff_param = K_eye.mm(diff_param)
            else:
                raise RuntimeError('No Handling Parameter Shape!')
            res += torch.norm(diff_param, p=2)
        return res

    @torch.no_grad()
    def helper2(self, P, current_model, agent_model):
        for current_param, agent_param in zip(current_model.parameters(), agent_model.parameters()):
            diff_param = current_param.data - agent_param.data
            if len(diff_param.shape) == 1:
                diff_param = diff_param.reshape(diff_param.size(0), 1)
                P_eye = P * torch.eye(diff_param.size(0)).to(self.device)
                diff_param = P_eye.mm(diff_param)
                diff_param = diff_param.reshape(-1)
            elif len(diff_param.shape) == 2:
                P_eye = P * torch.eye(diff_param.size(0)).to(self.device)
                diff_param = P_eye.mm(diff_param)
            else:
                raise RuntimeError('No Handling Parameter Shape!')
            agent_param.data = agent_param.data + diff_param
            del diff_param

    def D_train(self, x, gr):
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
        if gr > 0:
            D_loss += self.helper1(K=self.lr * (-1.1), current_model=self.model_D,
                                   agent_model=self.agent_D)
        D_loss.backward()

        self.D_optimizer.step()

        return D_loss.data.item()

    def G_train(self, x, gr):
        #=======================Train the generator=======================#
        self.model_G.zero_grad()
        bs = x.size(0)
        z = Variable(torch.randn(bs, 100).to(self.device))
        y = Variable(torch.ones(bs, 1).to(self.device))

        G_output = self.model_G(z)
        D_output = self.model_D(G_output)
        G_loss = self.loss(D_output, y)
        if gr > 0:
            G_loss += self.helper1(K=self.lr * (-1.1), current_model=self.model_G,
                                   agent_model=self.agent_G)
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

    def train(self, gr):
        trainloader = self.load_train_data()

        start_time = time.time()

        # self.model.to(self.device)
        self.model_D.train()
        self.model_G.train()

        max_local_steps = self.local_steps
        # if self.train_slow:
        #     max_local_steps = np.random.randint(1, max_local_steps // 2)
        agent_train = True
        for epoch in range(1, max_local_steps+1):
            D_losses, G_losses = [], []
            for batch_idx, (x, _) in enumerate(trainloader):
                # if epoch == max_local_steps and batch_idx == len(trainloader) - 1:
                #     agent_train = True
                D_loss = self.D_train(x, gr)
                G_loss = self.G_train(x, gr)
                D_losses.append(D_loss)
                G_losses.append(G_loss)

            print('[%d/%d]: loss_d: %.3f, loss_g: %.3f' % ((epoch),
                  max_local_steps, np.mean(D_losses), np.mean(G_losses)))
            # with open(os.path.join(self.results_dir, '{}_train_loss.csv'.format(self.id)), 'a') as csvfile:
            #     writer = csv.writer(csvfile)
            #     writer.writerow([np.mean(D_losses), np.mean(G_losses)])

        # self.model.cpu()
        # print('train agent')
        self.helper2(P=1, current_model=self.model_D,
                     agent_model=self.agent_D)
        self.helper2(P=1, current_model=self.model_G,
                     agent_model=self.agent_G)
        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        # self.model_G.eval()
        # self.model_D.eval()
        # G_loss = self.G_eval()
        return D_loss, G_loss
