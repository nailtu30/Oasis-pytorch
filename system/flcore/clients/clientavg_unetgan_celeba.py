import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable
import csv
import os
import copy
from torch.utils.data import DataLoader
from flcore.trainmodel.BigGAN import Generator, G_D
import sys
sys.path.append('/home/nailtu/codes/FLGAN/system/flcore/clients')
import unetgan_utils
import train_fns
from mixup import CutMix

class clientAVG(Client):
    def __init__(self, args, id, train_samples, **kwargs):
        super().__init__(args, id, train_samples, **kwargs)

        # self.loss = nn.BCELoss()
        # lr = 0.001
        # self.lr = lr
        # momentum = 0.9
        # self.G_optimizer = torch.optim.Adam(self.model_G.parameters(), lr=lr)
        # self.D_optimizer = torch.optim.Adam(self.model_D.parameters(), lr=lr)
        # self.G_optimizer = torch.optim.SGD(
        #     self.model_G.parameters(), lr=lr, momentum=momentum)
        # self.D_optimizer = torch.optim.SGD(
        #     self.model_D.parameters(), lr=lr, momentum=momentum)
        self.results_dir = args.results_dir
        self.train_data = kwargs['train_data']
        self.config = args.config
        print('clientAVG UNetGAN')#
        # print(self.id, self.train_samples)
        # print(self.id, self.train_data)
    
    def run(self):
        self.model_D.to(self.device)
        self.model_G.to(self.device)
        G_ema = Generator(**{**self.config, 'skip_init':True, 'no_optim': True}).to(self.device)
        # for key in self.model_G.state_dict():
        #     print(key)
        # for key in G_ema.state_dict():
        #     print(key)
        ema = unetgan_utils.ema(self.model_G, G_ema, self.config['ema_decay'], self.config['ema_start'])
        state_dict = {'itr': 0, 'epoch': 0, 'save_num': 0, 'save_best_num': 0, 'best_IS': 0,'best_FID': 999999,'config': self.config}
        GD = G_D(self.model_G, self.model_D, self.config)
        G_batch_size = max(self.config['G_batch_size'], self.config['batch_size'])
        G_batch_size = int(G_batch_size*self.config["num_G_accumulations"])
        z_, y_ = unetgan_utils.prepare_z_y(G_batch_size, self.model_G.dim_z, self.config['n_classes'],
                             device=self.device, fp16=self.config['G_fp16'])
        if self.config['parallel']:
            GD = nn.DataParallel(GD)
        
            # Prepare a fixed z & y to see individual sample evolution throghout training
        fixed_z, fixed_y = unetgan_utils.prepare_z_y(G_batch_size, self.model_G.dim_z,
                                        self.config['n_classes'], device=self.device,
                                        fp16=self.config['G_fp16'])
        fixed_z.sample_()
        fixed_y.sample_()

        train = train_fns.GAN_training_function(self.model_G, self.model_D, GD, z_, y_,
                                                ema, state_dict, self.config)
        batch_size = self.config['batch_size']
        for epoch in range(self.local_steps):
            pbar = self.load_train_data()
            target_map = None
            D_losses = []
            G_losses = []
            start_time = time.time()
            for i, batch_data in enumerate(pbar):
                # print('=======batch {}======='.format(i))
                x = batch_data[0]
                y = batch_data[1]
                state_dict['itr'] += 1
                self.model_G.train()
                self.model_D.train()
                G_ema.train()
                x, y = x.to(self.device), y.to(self.device).view(-1)
                x.requires_grad = False
                y.requires_grad = False

                # Here we load cutmix masks for every image in the batch
                n_mixed = int(x.size(0)/self.config["num_D_accumulations"])
                target_map = torch.cat([CutMix(self.config["resolution"]).cuda().view(1,1,self.config["resolution"],self.config["resolution"]) for _ in range(n_mixed) ],dim=0)

                r_mixup = 0.0
                metrics = train(x, y, state_dict["epoch"], batch_size , target_map = target_map, r_mixup = r_mixup)
                # print('metrics: {}'.format(metrics))
                D_loss_real = metrics['D_loss_real']
                D_loss_fake = metrics['D_loss_fake']
                D_loss = D_loss_real + D_loss_fake
                G_loss = metrics['G_loss']
                D_losses.append(D_loss)
                G_losses.append(G_loss)
            state_dict['epoch'] += 1
        print('[{}/{}]  D_loss: {}, G_loss: {}. Comsuming Time(s): {}'.format(epoch, self.local_steps, np.mean(D_losses), np.mean(G_losses), time.time() - start_time))
        self.model_D.to('cpu')
        self.model_G.to('cpu')
        return D_loss, G_loss
    
    def load_train_data(self, batch_size=None):
        if batch_size == None:
            batch_size = self.batch_size
        return DataLoader(self.train_data, batch_size, shuffle=True, drop_last=True)

    def train(self):
        start_time = time.time()
        print('client: {} train'.format(self.id))
        D_loss, G_loss = self.run()

        self.train_time_cost['num_rounds'] += 1
        self.train_time_cost['total_cost'] += time.time() - start_time
        return D_loss, G_loss
