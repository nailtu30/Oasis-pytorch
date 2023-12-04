import torch
import torch.nn as nn
import numpy as np
import time
from flcore.clients.clientbase import Client
from torch.autograd import Variable
import csv
import os
import copy

# from hessian_penalty import hessian_penalty


def hessian_penalty(G, z, k=2, epsilon=0.1, reduction=torch.max, return_separately=False, G_z=None, **G_kwargs):
    """
    Official PyTorch Hessian Penalty implementation.

    Note: If you want to regularize multiple network activations simultaneously, you need to
    make sure the function G you pass to hessian_penalty returns a list of those activations when it's called with
    G(z, **G_kwargs). Otherwise, if G returns a tensor the Hessian Penalty will only be computed for the final
    output of G.

    :param G: Function that maps input z to either a tensor or a list of tensors (activations)
    :param z: Input to G that the Hessian Penalty will be computed with respect to
    :param k: Number of Hessian directions to sample (must be >= 2)
    :param epsilon: Amount to blur G before estimating Hessian (must be > 0)
    :param reduction: Many-to-one function to reduce each pixel/neuron's individual hessian penalty into a final loss
    :param return_separately: If False, hessian penalties for each activation output by G are automatically summed into
                              a final loss. If True, the hessian penalties for each layer will be returned in a list
                              instead. If G outputs a single tensor, setting this to True will produce a length-1
                              list.
    :param G_z: [Optional small speed-up] If you have already computed G(z, **G_kwargs) for the current training
                iteration, then you can provide it here to reduce the number of forward passes of this method by 1
    :param G_kwargs: Additional inputs to G besides the z vector. For example, in BigGAN you
                     would pass the class label into this function via y=<class_label_tensor>

    :return: A differentiable scalar (the hessian penalty), or a list of hessian penalties if return_separately is True
    """
    if G_z is None:
        G_z = G(z, **G_kwargs)
    rademacher_size = torch.Size((k, *z.size()))  # (k, N, z.size())
    xs = epsilon * rademacher(rademacher_size, device=z.device)
    second_orders = []
    for x in xs:  # Iterate over each (N, z.size()) tensor in xs
        central_second_order = multi_layer_second_directional_derivative(
            G, z, x, G_z, epsilon, **G_kwargs)
        # Appends a tensor with shape equal to G(z).size()
        second_orders.append(central_second_order)
    loss = multi_stack_var_and_reduce(
        second_orders, reduction, return_separately)  # (k, G(z).size()) --> scalar
    return loss


def rademacher(shape, device='cpu'):
    """Creates a random tensor of size [shape] under the Rademacher distribution (P(x=1) == P(x=-1) == 0.5)"""
    x = torch.empty(shape, device=device)
    x.random_(0, 2)  # Creates random tensor of 0s and 1s
    x[x == 0] = -1  # Turn the 0s into -1s
    return x


def multi_layer_second_directional_derivative(G, z, x, G_z, epsilon, **G_kwargs):
    """Estimates the second directional derivative of G w.r.t. its input at z in the direction x"""
    G_to_x = G(z + x, **G_kwargs)
    G_from_x = G(z - x, **G_kwargs)

    G_to_x = listify(G_to_x)
    G_from_x = listify(G_from_x)
    G_z = listify(G_z)

    eps_sqr = epsilon ** 2
    sdd = [(G2x - 2 * G_z_base + Gfx) / eps_sqr for G2x,
           G_z_base, Gfx in zip(G_to_x, G_z, G_from_x)]
    return sdd


def stack_var_and_reduce(list_of_activations, reduction=torch.max):
    """Equation (5) from the paper."""
    second_orders = torch.stack(list_of_activations)  # (k, N, C, H, W)
    var_tensor = torch.var(second_orders, dim=0, unbiased=True)  # (N, C, H, W)
    penalty = reduction(var_tensor)  # (1,) (scalar)
    return penalty


def multi_stack_var_and_reduce(sdds, reduction=torch.max, return_separately=False):
    """Iterate over all activations to be regularized, then apply Equation (5) to each."""
    sum_of_penalties = 0 if not return_separately else []
    for activ_n in zip(*sdds):
        penalty = stack_var_and_reduce(activ_n, reduction)
        sum_of_penalties += penalty if not return_separately else [penalty]
    return sum_of_penalties


def listify(x):
    """If x is already a list, do nothing. Otherwise, wrap x in a list."""
    if isinstance(x, list):
        return x
    else:
        return [x]


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

        D_loss.backward()

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

        hp_loss = hessian_penalty(self.model_G, z, epsilon=0.5)
        hp_loss.backward()

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
