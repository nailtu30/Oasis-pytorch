import time
from flcore.clients.clientsim import clientSim
from flcore.servers.serverbase import Server
from threading import Thread
import csv
import numpy as np
import os
import torch
import torch.nn as nn
from torch.autograd import Variable
from utils.kmeans import KMeans
import copy
from sklearn.decomposition import FastICA


class FedSim(Server):
    def __init__(self, args, times):
        super().__init__(args, times)
        # select slow clients
        # self.set_slow_clients()
        self.set_clients(args, clientSim)

        print(
            f"\nJoin ratio / total clients: {self.join_ratio} / {self.num_clients}")
        print("Finished creating server and clients.")

        # self.load_model()
        self.Budget = []
        self.cluser_size = args.cluster_size
        self.model_repository = [[copy.deepcopy(self.global_model_D), copy.deepcopy(
            self.global_model_G)] for _ in range(self.cluser_size)]
        # print(self.model_repository)
        self.group_selected_times = [0] * self.cluser_size
        self.group_unable_avg_times = [0] * self.cluser_size

    def create_similarity_table(self):
        representations = []
        max_len = 0
        for idx, client in enumerate(self.clients):
            print('========{}========'.format(idx))
            representation = client.extract_representation()
            representations.append(representation)
            max_len = max(max_len, len(representation))

        for idx, representation in enumerate(representations):
            while len(representation) < max_len:
                representation = np.append(representation, 0.0)
            representations[idx] = representation

        representations = np.array(representations)
        # print('ICA')
        # ICA = FastICA(n_components=100)
        # representations = ICA.fit_transform(representations)
        print('Kmeans')
        cluster_size = self.cluser_size
        cp, cluster = KMeans(representations, cluster_size)
        groups = [[] for _ in range(cluster_size)]
        self.cluster = []
        for id, (group_id, _) in enumerate(cluster):
            group_id = group_id.astype(int)
            groups[group_id].append(id)
            self.cluster.append(group_id)
        for group_idx, client_ids in enumerate(groups):
            data_size = 0
            for client in client_ids:
                data_size += self.clients[client].data_size
            print('group{}: {}, group size: {}, data size: {}'.format(
                group_idx, client_ids, len(client_ids), data_size))
        self.groups = groups
        # print(self.clutser)

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            c_id = client.id
            g_id = self.cluster[c_id]
            client.set_parameters_D(self.model_repository[g_id][0])
            client.set_parameters_G(self.model_repository[g_id][1])

    def receive_models(self):
        assert (len(self.selected_clients) > 0)
        self.uploaded_weights = [[] for _ in range(self.cluser_size)]
        tot_samples = [0] * self.cluser_size
        self.uploaded_models_D = [[] for _ in range(self.cluser_size)]
        self.uploaded_models_G = [[] for _ in range(self.cluser_size)]
        self.uploaded_group_ids = set()
        for client in self.selected_clients:
            if not client.able_avg:
                continue
            c_id = client.id
            g_id = self.cluster[c_id]
            print('group {} is selected'.format(g_id))
            self.group_selected_times[g_id] = self.group_selected_times[g_id] + 1
            self.uploaded_group_ids.add(g_id)
            self.uploaded_weights[g_id].append(client.train_samples)
            tot_samples[g_id] += client.train_samples
            self.uploaded_models_D[g_id].append(client.model_D)
            self.uploaded_models_G[g_id].append(client.model_G)
        for g_id, uw in enumerate(self.uploaded_weights):
            for i, w in enumerate(uw):
                uw[i] = w / tot_samples[g_id]

    def aggregate_parameters(self):
        # assert (len(self.uploaded_models_G) > 0)
        # assert (len(self.uploaded_models_D) > 0)
        for g_id, model_pair in enumerate(self.model_repository):
            if g_id not in self.uploaded_group_ids:
                continue
            model_pair[0] = copy.deepcopy(self.uploaded_models_D[g_id][0])
            model_pair[1] = copy.deepcopy(self.uploaded_models_G[g_id][0])
        # self.global_model_D = copy.deepcopy(self.uploaded_models_D[0])
        # self.global_model_G = copy.deepcopy(self.uploaded_models_G[0])
        for g_id, model_pair in enumerate(self.model_repository):
            if g_id not in self.uploaded_group_ids:
                continue
            for param in model_pair[0].parameters():
                param.data.zero_()
            for param in model_pair[1].parameters():
                param.data.zero_()

        # for param in self.global_model_D.parameters():
        #     param.data.zero_()

        # for param in self.global_model_G.parameters():
        #     param.data.zero_()
        for g_id, model_pair in enumerate(self.model_repository):
            if g_id not in self.uploaded_group_ids:
                continue
            uw = self.uploaded_weights[g_id]
            g_Ds = self.uploaded_models_D[g_id]
            g_Gs = self.uploaded_models_G[g_id]
            for w, client_model in zip(uw, g_Ds):
                for server_param, client_param in zip(model_pair[0].parameters(), client_model.parameters()):
                    server_param.data += client_param.data.clone() * w
            for w, client_model in zip(uw, g_Gs):
                for server_param, client_param in zip(model_pair[1].parameters(), client_model.parameters()):
                    server_param.data += client_param.data.clone() * w
        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models_D):
        #     self.add_parameters_D(w, client_model)

        # for w, client_model in zip(self.uploaded_weights, self.uploaded_models_G):
        #     self.add_parameters_G(w, client_model)

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
        unable_client_ids = []
        for i in range(1, self.global_rounds+1):
            s_t = time.time()
            self.selected_clients = self.select_clients()
            self.send_models()
            D_losses = []
            G_losses = []
            for client in self.selected_clients:
                D_loss, G_loss, able_avg = client.train()
                if able_avg:
                    D_losses.append(D_loss)
                    G_losses.append(G_loss)
                else:
                    unable_client_ids.append((i, client.id))
                    self.group_unable_avg_times[self.cluster[client.id]
                                                ] = self.group_unable_avg_times[self.cluster[client.id]] + 1
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
        # torch.save(self.global_model_G.state_dict(),
        #            os.path.join(self.results_dir, 'global_netG.pth'))
        for g_id, model_pair in enumerate(self.model_repository):
            torch.save(model_pair[1].state_dict(), os.path.join(
                self.results_dir, 'global_netG{}.pth'.format(g_id)))

        for g_id, times in enumerate(self.group_selected_times):
            info = 'group{}: {} times'.format(g_id, times)
            print(info)
            with open(os.path.join(self.results_dir, 'group_select.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([times])
            # with open(os.path.join(self.results_dir, 'intro.txt'), 'a') as file:
            #     file.write(info)
            #     file.write('\n')

        for g_id, times in enumerate(self.group_unable_avg_times):
            info = 'unable avg, group{}: {} times'.format(g_id, times)
            print(info)
            with open(os.path.join(self.results_dir, 'group_unable_select.csv'), 'a') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([times])

        for (global_epoch, client_id) in unable_client_ids:
            info = 'global epoch: {}, client id: {}'.format(
                global_epoch, client_id)
            with open(os.path.join(self.results_dir, 'intro.txt'), 'a') as file:
                file.write(info)
                file.write('\n')
