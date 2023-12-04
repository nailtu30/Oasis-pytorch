import copy
import torch
import argparse
import os
import time
import warnings
import numpy as np
from torch.nn.functional import dropout
import torchvision

from flcore.servers.serveravg import FedAvg
from flcore.servers.serverprox import FedProx
from flcore.servers.serverditto import Ditto

from flcore.servers.serverkvp import FedKVP
from flcore.servers.serverkvp2 import FedKVP2

from flcore.servers.serveravg_dcgan_cifar10 import FedAvgDCGANCifar10
from flcore.servers.serveravg_dcgan_cifar10_kvp import FedAvgDCGANCifar10KVP

from flcore.servers.serveravg_dcgan_cifar10_way2 import FedAvgDCGANCifar10Way2

from flcore.servers.serveravg_dcgan_cifar10_kvp2 import FedAvgDCGANCifar10KVP2

from flcore.trainmodel.mlp_model import *
from flcore.trainmodel.dcgan_cifar10_model import *

from flcore.servers.serveravg2 import FedAvg2
from flcore.servers.serveravg3 import FedAvg3
from flcore.servers.serveravg4 import FedAvg4
from flcore.servers.serveravg5 import FedAvg5
from flcore.servers.serveravg6 import FedAvg6
from flcore.servers.serveravg7 import FedAvg7
from flcore.servers.serveravg8 import FedAvg8
from flcore.servers.serveravg9 import FedAvg9
from flcore.servers.serveravg10 import FedAvg10
from flcore.servers.serveravg11 import FedAvg11
from flcore.servers.serveravg12 import FedAvg12

from flcore.servers.serversim import FedSim

from flcore.servers.serversim_ISODATA import FedSimISODATA

from flcore.servers.server_moti_cluser import FedAvgMotiCluster

from flcore.servers.serversim2 import FedSim2

from flcore.servers.serveravg_unetgan_celeba import FedAvgCeleba

from flcore.servers.serverprox_dcgan_cifar10 import FedProxDCGANCifar10
from flcore.servers.serverditto_dcgan_cifar10 import DittoDCGANCifar10

torch.manual_seed(0)


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def run(args):
    time_list = []
    for i in range(args.prev, args.times):
        print(f"\n============= Running time: {i}th =============")
        print("Creating server and clients ...")
        start = time.time()
        if args.dataset == 'mnist':
            # args.model_G = MNISTGenerator(
            #     g_input_dim=100, g_output_dim=784).to(args.device)
            # args.model_D = MNISTDiscriminator(784).to(args.device)
            args.model_G = MNISTGenerator(
                g_input_dim=100, g_output_dim=784).cuda(0)
            args.model_D = MNISTDiscriminator(784).cuda(0)
            if args.algorithm == 'FedAvg':
                server = FedAvg(args, i)
            elif args.algorithm == 'FedProx':
                server = FedProx(args, i)
            elif args.algorithm == 'Ditto':
                server = Ditto(args, i)
            elif args.algorithm == 'kvp':
                server = FedKVP(args, i)
            elif args.algorithm == 'kvp2':
                server = FedKVP2(args, i)
            elif args.algorithm == 'FedAvg2':
                server = FedAvg2(args, i)
            elif args.algorithm == 'FedAvg3':
                server = FedAvg3(args, i)
            elif args.algorithm == 'FedAvg4':
                server = FedAvg4(args, i)
            elif args.algorithm == 'FedAvg5':
                server = FedAvg5(args, i)
            elif args.algorithm == 'FedAvg6':
                server = FedAvg6(args, i)
            elif args.algorithm == 'FedAvg7':
                server = FedAvg7(args, i)
            elif args.algorithm == 'FedAvg8':
                server = FedAvg8(args, i)
            elif args.algorithm == 'FedAvg9':
                server = FedAvg9(args, i)
            elif args.algorithm == 'FedAvg10':
                server = FedAvg10(args, i)
            elif args.algorithm == 'FedAvg11':
                server = FedAvg11(args, i)
            elif args.algorithm == 'FedAvg12':
                server = FedAvg12(args, i)
            elif args.algorithm == 'FedSim':
                server = FedSim(args, i)
            elif args.algorithm == 'FedSim2':
                server = FedSim2(args, i)
            elif args.algorithm == 'FedSimISODATA':
                server = FedSimISODATA(args, i)
            elif args.algorithm == 'FedAvgMC':
                server = FedAvgMotiCluster(args, i)
        elif args.dataset == 'cifar10':
            args.model_G = Generator().to(args.device)
            args.model_G.apply(weights_init)
            args.model_D = Discriminator().to(args.device)
            args.model_D.apply(weights_init)
            if args.algorithm == 'kvp':
                server = FedAvgDCGANCifar10KVP(args, i)
            elif args.algorithm == 'way2':
                server = FedAvgDCGANCifar10Way2(args, i)
            elif args.algorithm == 'kvp2':
                server = FedAvgDCGANCifar10KVP2(args, i)
            elif args.algorithm == 'FedProx':
                print('FedProx  Cifar10')
                server = FedProxDCGANCifar10(args, i)
            elif args.algorithm == 'Ditto':
                print('Ditto  Cifar10')
                server = DittoDCGANCifar10(args, i)
            else:
                print('use Fedavg')
                server = FedAvgDCGANCifar10(args, i)
        elif args.dataset == 'celeba':
            args.model_G = Generator().to(args.device)
            args.model_D = Discriminator().to(args.device)
            if args.algorithm == 'FedAvg':
                server = FedAvgCeleba(args, i)

        if args.algorithm == 'FedSim' or args.algorithm == 'FedSim2':
            server.create_similarity_table()
            # server.train()
        elif args.dataset == 'celeba':
            continue
        else:
            server.train()
        time_list.append(time.time()-start)

    print(f"\nAverage time cost: {round(np.average(time_list), 2)}s.")

    print("All done!")


if __name__ == "__main__":
    total_start = time.time()

    parser = argparse.ArgumentParser()
    # general
    parser.add_argument('--intro', type=str, default='FL Experiment')
    parser.add_argument('--results_dir', '-rd', type=str)
    parser.add_argument('-go', "--goal", type=str, default="test",
                        help="The goal for this experiment")
    parser.add_argument('-dev', "--device", type=str, default="cuda",
                        choices=["cpu", "cuda"])
    parser.add_argument('-did', "--device_id", type=str, default="0")
    parser.add_argument('-data', "--dataset", type=str, default="mnist")
    parser.add_argument('-nb', "--num_classes", type=int, default=10)
    parser.add_argument('-lbs', "--batch_size", type=int, default=32)
    parser.add_argument('-gr', "--global_rounds", type=int, default=200)
    parser.add_argument('-ls', "--local_steps", type=int, default=5)
    parser.add_argument('-algo', "--algorithm", type=str, default="FedAvg")
    parser.add_argument('-jr', "--join_ratio", type=float, default=0.1,
                        help="Ratio of clients per round")
    parser.add_argument('-jc', "--join_clients", type=int, default=10,
                        help="join of clients per round")
    parser.add_argument('-nc', "--num_clients", type=int, default=100,
                        help="Total number of clients")
    parser.add_argument('-pv', "--prev", type=int, default=0,
                        help="Previous Running times")
    parser.add_argument('-t', "--times", type=int, default=1,
                        help="Running times")
    parser.add_argument('-eg', "--eval_gap", type=int, default=1,
                        help="Rounds gap for evaluation")
    parser.add_argument('-sfn', "--save_folder_name",
                        type=str, default='models')
    parser.add_argument('-mu', "--mu", type=float, default=0,
                        help="Proximal rate for FedProx")
    parser.add_argument('-pls', "--plocal_steps", type=int, default=1)
    parser.add_argument('-cs', "--cluster_size", type=int, default=10)
    parser.add_argument('-ga', "--gamma", type=float, default=1)

    args = parser.parse_args()

    os.environ["CUDA_VISIBLE_DEVICES"] = args.device_id

    print("=" * 50)

    with open(os.path.join(args.results_dir, 'intro.txt'), 'a') as file:
        # file.write('{}: {}\n'.format(
        #     'intro', 'FLGAN SGD experiment'))
        for key, value in vars(args).items():
            file.write('{}: {}'.format(key, value))
            file.write('\n')
    run(args)
