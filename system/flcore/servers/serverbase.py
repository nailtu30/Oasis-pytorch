import torch
import os
import numpy as np
import h5py
import copy
import time
import random
import csv
from torch.autograd import Variable
from scipy.special import rel_entr
import scipy.stats
import torch.nn.functional as F

from utils.data_utils import read_client_data


class Server(object):
    def __init__(self, args, times):
        # Set up the main attributes
        self.device = args.device
        self.dataset = args.dataset
        self.global_rounds = args.global_rounds
        self.local_steps = args.local_steps
        self.batch_size = args.batch_size
        # self.global_model = copy.deepcopy(args.model)
        self.global_model_D = copy.deepcopy(args.model_D)
        self.global_model_G = copy.deepcopy(args.model_G)
        self.num_clients = args.num_clients
        self.join_ratio = args.join_ratio
        self.join_clients = args.join_clients
        # self.join_clients = int(self.num_clients * self.join_ratio)
        self.algorithm = args.algorithm
        self.goal = args.goal
        self.save_folder_name = args.save_folder_name
        self.top_cnt = 100

        self.clients = []
        self.selected_clients = []
        self.train_slow_clients = []
        self.send_slow_clients = []

        self.uploaded_weights = []
        self.uploaded_ids = []
        self.uploaded_models = []

        self.rs_test_acc = []
        self.rs_test_auc = []
        self.rs_train_loss = []

        self.times = times
        self.results_dir = args.results_dir

    def set_clients(self, args, clientObj):
        for i in range(self.num_clients):
            train_data = read_client_data(self.dataset, i, is_train=True)
            # test_data = read_client_data(self.dataset, i, is_train=False)
            client = clientObj(args,
                               id=i,
                               train_samples=len(train_data))
            self.clients.append(client)

    # random select slow clients
    def select_slow_clients(self, slow_rate):
        slow_clients = [False for i in range(self.num_clients)]
        idx = [i for i in range(self.num_clients)]
        idx_ = np.random.choice(idx, int(slow_rate * self.num_clients))
        for i in idx_:
            slow_clients[i] = True

        return slow_clients

    def set_slow_clients(self):
        self.train_slow_clients = self.select_slow_clients(
            self.train_slow_rate)
        self.send_slow_clients = self.select_slow_clients(
            self.send_slow_rate)

    def select_clients(self):
        selected_clients = list(np.random.choice(
            self.clients, self.join_clients, replace=False))

        return selected_clients

    def send_models(self):
        assert (len(self.clients) > 0)

        for client in self.clients:
            client.set_parameters_D(self.global_model_D)
            client.set_parameters_G(self.global_model_G)

    def receive_models(self):
        assert (len(self.selected_clients) > 0)

        self.uploaded_weights = []
        tot_samples = 0
        self.uploaded_ids = []
        self.uploaded_models_D = []
        self.uploaded_models_G = []
        for client in self.selected_clients:
            self.uploaded_weights.append(client.train_samples)
            tot_samples += client.train_samples
            self.uploaded_ids.append(client.id)
            self.uploaded_models_D.append(client.model_D)
            self.uploaded_models_G.append(client.model_G)
        for i, w in enumerate(self.uploaded_weights):
            self.uploaded_weights[i] = w / tot_samples

    def aggregate_parameters(self):
        # assert (len(self.uploaded_models_G) > 0)
        assert (len(self.uploaded_models_D) > 0)

        self.global_model_D = copy.deepcopy(self.uploaded_models_D[0])
        self.global_model_G = copy.deepcopy(self.uploaded_models_G[0])
        for param in self.global_model_D.parameters():
            param.data.zero_()

        for param in self.global_model_G.parameters():
            param.data.zero_()

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models_D):
            self.add_parameters_D(w, client_model)

        for w, client_model in zip(self.uploaded_weights, self.uploaded_models_G):
            self.add_parameters_G(w, client_model)

    def add_parameters_D(self, w, client_model):
        for server_param, client_param in zip(self.global_model_D.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def add_parameters_G(self, w, client_model):
        for server_param, client_param in zip(self.global_model_G.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w

    def add_parameters_G2(self, w, client_model):
        index = 0
        for server_param, client_param in zip(self.global_model_G.parameters(), client_model.parameters()):
            server_param.data += client_param.data.clone() * w[0]
            index = index + 1

    # def save_global_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     if not os.path.exists(model_path):
    #         os.makedirs(model_path)
    #     model_path = os.path.join(
    #         model_path, self.algorithm + "_server" + ".pt")
    #     torch.save(self.global_model, model_path)

    # def load_model(self):
    #     model_path = os.path.join("models", self.dataset)
    #     model_path = os.path.join(
    #         model_path, self.algorithm + "_server" + ".pt")
    #     assert (os.path.exists(model_path))
    #     self.global_model = torch.load(model_path)

    # def model_exists(self):
    #     model_path = os.path.join("models", self.dataset)
    #     model_path = os.path.join(
    #         model_path, self.algorithm + "_server" + ".pt")
    #     return os.path.exists(model_path)

    # def save_results(self):
    #     algo = self.dataset + "_" + self.algorithm
    #     result_path = "../results/"
    #     if not os.path.exists(result_path):
    #         os.makedirs(result_path)

    #     if (len(self.rs_test_acc)):
    #         algo = algo + "_" + self.goal + "_" + str(self.times)
    #         file_path = result_path + "{}.h5".format(algo)
    #         print("File path: " + file_path)

    #         with h5py.File(file_path, 'w') as hf:
    #             hf.create_dataset('rs_test_acc', data=self.rs_test_acc)
    #             hf.create_dataset('rs_test_auc', data=self.rs_test_auc)
    #             hf.create_dataset('rs_train_loss', data=self.rs_train_loss)

    # def save_item(self, item, item_name):
    #     if not os.path.exists(self.save_folder_name):
    #         os.makedirs(self.save_folder_name)
    #     torch.save(item, os.path.join(
    #         self.save_folder_name, "server_" + item_name + ".pt"))

    # def load_item(self, item_name):
    #     return torch.load(os.path.join(self.save_folder_name, "server_" + item_name + ".pt"))

    # def test_metrics(self):
    #     num_samples = []
    #     tot_correct = []
    #     tot_auc = []
    #     for c in self.clients:
    #         ct, ns, auc = c.test_metrics()
    #         tot_correct.append(ct*1.0)
    #         tot_auc.append(auc*ns)
    #         num_samples.append(ns)

    #     ids = [c.id for c in self.clients]

    #     return ids, num_samples, tot_correct, tot_auc

    # def test_select_metrics(self):
    #     num_samples = []
    #     tot_correct = []
    #     tot_auc = []
    #     for c in self.selected_clients:
    #         ct, ns, auc = c.test_metrics()
    #         tot_correct.append(ct*1.0)
    #         tot_auc.append(auc*ns)
    #         num_samples.append(ns)

    #     ids = [c.id for c in self.clients]

    #     return ids, num_samples, tot_correct, tot_auc

    def train_metrics(self):
        num_samples = []
        losses = []
        for c in self.clients:
            cl, ns = c.train_metrics()
            num_samples.append(ns)
            losses.append(cl*1.0)

        ids = [c.id for c in self.clients]

        return ids, num_samples, losses

    # evaluate selected clients
    # def evaluate(self, acc=None, loss=None):
    #     stats = self.test_metrics()
    #     stats_train = self.train_metrics()

    #     stats_selected = self.test_select_metrics()

    #     test_acc = sum(stats[2])*1.0 / sum(stats[1])
    #     test_auc = sum(stats[3])*1.0 / sum(stats[1])
    #     train_loss = sum(stats_train[2])*1.0 / sum(stats_train[1])
    #     accs = [a / n for a, n in zip(stats[2], stats[1])]
    #     aucs = [a / n for a, n in zip(stats[3], stats[1])]

    #     accs_selected = [a / n for a,
    #                      n in zip(stats_selected[2], stats_selected[1])]

    #     if acc == None:
    #         self.rs_test_acc.append(test_acc)
    #     else:
    #         acc.append(test_acc)

    #     if loss == None:
    #         self.rs_train_loss.append(train_loss)
    #     else:
    #         loss.append(train_loss)

    #     print("Averaged Train Loss: {:.4f}".format(train_loss))
    #     print("Averaged Test Accurancy: {:.4f}".format(test_acc))
    #     print("Averaged Test AUC: {:.4f}".format(test_auc))
    #     # self.print_(test_acc, train_acc, train_loss)
    #     print("Std Test Accurancy: {:.4f}".format(np.std(accs)))
    #     print("Std Test AUC: {:.4f}".format(np.std(aucs)))

    #     # accs = [acc for acc in accs if acc != 0]
    #     file_name = os.path.join(
    #         self.save_folder_name, self.algorithm + '_' + self.goal + '.csv')
    #     with open(file_name, 'a') as csvfile:
    #         writer = csv.writer(csvfile)
    #         writer.writerow(accs_selected)
    #     print('save clients test accs!')

    # def print_(self, test_acc, test_auc, train_loss):
    #     print("Average Test Accurancy: {:.4f}".format(test_acc))
    #     print("Average Test AUC: {:.4f}".format(test_auc))
    #     print("Average Train Loss: {:.4f}".format(train_loss))

    # def check_done(self, acc_lss, top_cnt=None, div_value=None):
    #     for acc_ls in acc_lss:
    #         if top_cnt != None and div_value != None:
    #             find_top = len(
    #                 acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
    #             find_div = len(acc_ls) > 1 and np.std(
    #                 acc_ls[-top_cnt:]) < div_value
    #             if find_top and find_div:
    #                 pass
    #             else:
    #                 return False
    #         elif top_cnt != None:
    #             find_top = len(
    #                 acc_ls) - torch.topk(torch.tensor(acc_ls), 1).indices[0] > top_cnt
    #             if find_top:
    #                 pass
    #             else:
    #                 return False
    #         elif div_value != None:
    #             find_div = len(acc_ls) > 1 and np.std(
    #                 acc_ls[-top_cnt:]) < div_value
    #             if find_div:
    #                 pass
    #             else:
    #                 return False
    #         else:
    #             raise NotImplementedError
    #     return True
