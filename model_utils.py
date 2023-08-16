import torch
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F

import os
from datetime import datetime
import time
import math
from utilities import *

from itertools import repeat
from torch.nn.parameter import Parameter
import collections
import matplotlib

matplotlib.use('Agg')

# Activation functions
def my_sigmoid(x):
    return 1 / (1 + torch.exp(-4 * (x - 0.5)))

def hard_sigmoid(x):
    return (1 + F.hardtanh(2 * x - 1)) * 0.5

def my_hard_sig(x):
    return (1 + F.hardtanh(x - 1)) * 0.5

# Some helper functions
def grad_or_zero(x):
    if x.grad is None:
        return torch.zeros_like(x).to(x.device)
    else:
        return x.grad

def neurons_zero_grad(neurons):
    for idx in range(len(neurons)):
        if neurons[idx].grad is not None:
            neurons[idx].grad.zero_()

def copy(neurons):
    copy = []
    for n in neurons:
        copy.append(torch.empty_like(n).copy_(n.data).requires_grad_())
    return copy

# Specifically for the SNLI dataset
class P_MLP(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh, seqLen=25):
        super(P_MLP, self).__init__()
        self.sq_len = seqLen
        self.enc_len = 300   #Change this if you change the word_embed dimension
        self.activation = activation
        self.archi = archi
        self.nc = self.archi[-1]    #Number of classes for the classification problem SNLI -> 3
        #We call the connections as synapses and the nodes as neurons
        self.synapses = torch.nn.ModuleList()
        #First we add four linear projection connections
        self.synapses.append(torch.nn.Linear(self.enc_len, self.enc_len, bias=False))
        self.synapses.append(torch.nn.Linear(self.enc_len, self.enc_len, bias=False))
        self.synapses.append(torch.nn.Linear(self.enc_len, self.enc_len, bias=False))
        self.synapses.append(torch.nn.Linear(self.enc_len, self.enc_len, bias=False))
        # Fully connected connections
        for idx in range(len(archi) - 1):
            self.synapses.append(torch.nn.Linear(archi[idx], archi[idx + 1], bias=True))
        self.qk_val = None

    def Phi(self, x, y, neurons, beta, criterion):
        batch_size = x.size(0)
        x = x.squeeze()
        y = y.long()
        layers = [x] + neurons  # Adding the input as s^0 as described in paper
        # Scalar Primitive function computation
        phi = 0.0
        # Implementation of Phi is as described in paper
        phi += torch.sum(self.synapses[0](layers[0][:,:self.sq_len,:]) * layers[1], dim=(1,2)).squeeze()
        phi += torch.sum(self.synapses[1](layers[0][:,self.sq_len:,:]) * layers[2], dim=(1,2)).squeeze()
        phi += torch.sum(self.synapses[2](layers[0][:,:self.sq_len,:]) * layers[3], dim=(1,2)).squeeze()
        phi += torch.sum(self.synapses[3](layers[0][:,self.sq_len:,:]) * layers[4], dim=(1,2)).squeeze()
        phi += torch.sum(self.synapses[4]((torch.cat([layers[1], layers[2], layers[3], layers[4]], dim=1)).view(batch_size, -1)) * layers[5],
                         dim=1).squeeze()
        for idx in range(5,len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layers[idx].view(batch_size, -1)) * layers[idx + 1], dim=1).squeeze()  # Scalar product s_n.W.s_n-1
        if beta != 0.0:  # Nudging the output layer when beta is non zero
            y = F.one_hot(y, num_classes=self.nc)
            L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False,
                type='train'):
        not_mse = (criterion.__class__.__name__.find('MSE') == -1)
        x = x.squeeze()
        batch_size = x.size(0)
        device = x.device
        scale = 0.16 #1. / math.sqrt(self.enc_len)
        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion)  # Computing Phi
            init_grads = torch.tensor([1 for i in range(batch_size)], dtype=torch.float, device=device,
                                      requires_grad=True)  # Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm)  # dPhi/ds

            for idx in range(len(neurons) - 1):
                if idx == 0:
                    query = grads[idx]
                    key = x[:,self.sq_len:,:].detach()
                    value = x[:,self.sq_len:,:].detach()
                    self.qk_val = torch.softmax(scale * torch.bmm(query, torch.transpose(key, 1, 2)), dim=-1)
                    attention_output = torch.bmm(self.qk_val, value)
                    neurons[idx] = attention_output#self.activation(attention_output)
                elif idx == 1:
                    query = grads[idx]
                    key = x[:,:self.sq_len,:].detach()
                    value = x[:,:self.sq_len,:].detach()
                    self.qk_val = torch.softmax(scale * torch.bmm(query, torch.transpose(key, 1, 2)), dim=-1)
                    attention_output = torch.bmm(self.qk_val, value)
                    neurons[idx] = attention_output#self.activation(attention_output)
                elif idx == 2:
                    query = grads[idx]
                    key = x[:,:self.sq_len,:].detach()
                    value = x[:,:self.sq_len,:].detach()
                    self.qk_val = torch.softmax(scale * torch.bmm(query, torch.transpose(key, 1, 2)), dim=-1)
                    attention_output = torch.bmm(self.qk_val, value)
                    neurons[idx] = attention_output#self.activation(attention_output)
                elif idx == 3:
                    query = grads[idx]
                    key = x[:,self.sq_len:,:].detach()
                    value = x[:,self.sq_len:,:].detach()
                    self.qk_val = torch.softmax(scale * torch.bmm(query, torch.transpose(key, 1, 2)), dim=-1)
                    attention_output = torch.bmm(self.qk_val, value)
                    neurons[idx] = attention_output#self.activation(attention_output)
                else:
                    neurons[idx] = self.activation(grads[idx])  # s_(t+1) = sigma( dPhi/ds )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True

            if not_mse:
                neurons[-1] = grads[-1]
            else:
                neurons[-1] = self.activation(grads[-1])

            if check_thm:
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        if type == 'test' or (type == 'train' and beta != 0.0):
            self.qk_val = None

        return neurons

    def init_neurons(self, batch_size, device, data=None):
        # Initializing the neurons
        neurons = []
        append = neurons.append
        append(torch.zeros((batch_size, self.sq_len, self.enc_len), requires_grad=True, device=device))
        append(torch.zeros((batch_size, self.sq_len, self.enc_len), requires_grad=True, device=device))
        append(torch.zeros((batch_size, self.sq_len, self.enc_len), requires_grad=True, device=device))
        append(torch.zeros((batch_size, self.sq_len, self.enc_len), requires_grad=True, device=device))
        for size in self.archi[1:]:
            append(torch.zeros((batch_size, size), requires_grad=True, device=device))
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        # Computing the EP update given two steady states neurons_1 and neurons_2, static input x, label y
        beta_1, beta_2 = betas
        #x = torch.nn.functional.layer_norm(x, normalized_shape=[self.sq_len, self.enc_len], weight=None, bias=None)
        #print(self.synapses[0].weight)
        self.zero_grad()  # p.grad is zero
        if not (check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # p.grad = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1) ----> dL/dp  by the theorem


class IMDB_model(torch.nn.Module):
    def __init__(self, archi, activation=torch.tanh, seqLen = 600):
        super(IMDB_model, self).__init__()
        self.sq_len = seqLen
        self.enc_len = 300 #Change this if you change the word_embed dimension
        self.activation = activation
        self.archi = archi
        self.nc = self.archi[-1] #Number of classes for the classification problem IMDB -> 2
        # Synapses
        self.synapses = torch.nn.ModuleList()
        self.synapses.append(torch.nn.Linear(self.enc_len, self.enc_len, bias=False))           #Linear Projection layer added
        for i in range(len(archi) - 1):        #Weights for the fully connected layers.
            self.synapses.append(torch.nn.Linear(archi[i], archi[i + 1], bias=True))
        self.qk_val = None

    def Phi(self, x, y, neurons, beta, criterion):
        batch_size = x.size(0)
        x = x.squeeze()
        y = y.long()
        layers = [x] + neurons  # Adding the input as s^0 as described in paper
        # Scalar Primitive function computation
        phi = 0.0
        phi += torch.sum(self.synapses[0](layers[0]) * layers[1], dim=(1,2)).squeeze()  # Scalar dot product between 2 tensors for the linear projection layer
        for idx in range(1,len(self.synapses)):
            phi += torch.sum(self.synapses[idx](layers[idx].view(batch_size, -1)) * layers[idx + 1], dim=1).squeeze()
        # phi += torch.sum(self.synapses[1](layers[1].view(batch_size, -1)) * layers[2],dim=1).squeeze()
        # phi += torch.sum(self.synapses[2](layers[2].view(batch_size, -1)) * layers[3],dim=1).squeeze()
        #phi += torch.sum(self.synapses[3](layers[3].view(batch_size, -1)) * layers[4],dim=1).squeeze()

        if beta != 0.0:  # Nudging the output layer when beta is non zero
            y = F.one_hot(y, num_classes=self.nc)
            L = 0.5 * criterion(layers[-1].float(), y.float()).sum(dim=1).squeeze()
            phi -= beta * L

        return phi

    def forward(self, x, y, neurons, T, beta=0.0, criterion=torch.nn.MSELoss(reduction='none'), check_thm=False):
        x = x.squeeze()
        batch_size = x.size(0)
        device = x.device
        scale = 1. / math.sqrt(self.enc_len)
        for t in range(T):
            phi = self.Phi(x, y, neurons, beta, criterion)  # Computing Phi
            init_grads = torch.tensor([1 for i in range(batch_size)], dtype=torch.float, device=device,
                                      requires_grad=True)  # Initializing gradients
            grads = torch.autograd.grad(phi, neurons, grad_outputs=init_grads, create_graph=check_thm)  # dPhi/ds
            for idx in range(len(neurons) - 1):
                if idx == 0:
                    query = grads[idx]
                    key = x
                    value = x
                    self.qk_val = torch.softmax(scale * torch.bmm(query, torch.transpose(key, 1, 2)), dim=-1)
                    attention_output = torch.bmm(self.qk_val, value)
                    neurons[idx] = attention_output
                else:
                    neurons[idx] = self.activation(grads[idx])  # s_(t+1) = sigma( dPhi/ds )
                if check_thm:
                    neurons[idx].retain_grad()
                else:
                    neurons[idx].requires_grad = True

            neurons[-1] = self.activation(grads[-1])

            if check_thm:       #If theorem check is enabled
                neurons[-1].retain_grad()
            else:
                neurons[-1].requires_grad = True

        return neurons

    def init_neurons(self, batch_size, device, data=None):
        # Neurons are initialised with 0's
        neurons = []
        neurons.append(torch.zeros((batch_size, self.sq_len, self.enc_len), requires_grad=True, device=device))
        for size in self.archi[1:]:
            neurons.append(torch.zeros((batch_size, size), requires_grad=True, device=device))
        return neurons

    def compute_syn_grads(self, x, y, neurons_1, neurons_2, betas, criterion, check_thm=False):
        # Computing EP estimates
        beta_1, beta_2 = betas
        self.zero_grad()  # p.grad is zero
        if not (check_thm):
            phi_1 = self.Phi(x, y, neurons_1, beta_1, criterion)
        else:
            phi_1 = self.Phi(x, y, neurons_1, beta_2, criterion)
        phi_1 = phi_1.mean()

        phi_2 = self.Phi(x, y, neurons_2, beta_2, criterion)
        phi_2 = phi_2.mean()

        delta_phi = (phi_2 - phi_1) / (beta_1 - beta_2)
        delta_phi.backward()  # gradient = -(d_Phi_2/dp - d_Phi_1/dp)/(beta_2 - beta_1)


def check_gdu(model, x, y, T1, T2, betas, criterion, alg='EP'):
    #This function verifies the GDU algorithm as mentioned in the paper

    BPTT, EP = {}, {}

    for name, p in model.named_parameters():
        BPTT[name], EP[name] = [], []

    neurons = model.init_neurons(x.size(0), x.device, x)
    for idx in range(len(neurons)):
        BPTT['neurons_' + str(idx)], EP['neurons_' + str(idx)] = [], []

    # We first compute BPTT gradients
    # First phase up to T1-T2
    beta_1, beta_2 = betas
    neurons = model(x, y, neurons, T1 - T2, beta=beta_1, criterion=criterion)
    ref_neurons = copy(neurons)

    # Last steps of the first phase
    for K in range(T2 + 1):

        neurons = model(x, y, neurons, K, beta=beta_1, criterion=criterion)  # Running K time step

        # detach data and neurons from the graph
        x = x.detach()
        x.requires_grad = True
        leaf_neurons = []
        for idx in range(len(neurons)):
            neurons[idx] = neurons[idx].detach()
            neurons[idx].requires_grad = True
            leaf_neurons.append(neurons[idx])

        neurons = model(x, y, neurons, T2 - K, beta=beta_1, criterion=criterion, check_thm=True)  # T2-K time step
        # Loss
        y = y.long()
        loss = (1 / (2.0 * x.size(0))) * criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(dim=1).squeeze()

        # setting gradients field to zero before backward
        neurons_zero_grad(leaf_neurons)
        model.zero_grad()

        # BPTT
        loss.backward(
            torch.tensor([1 for i in range(x.size(0))], dtype=torch.float, device=x.device, requires_grad=True))

        # Collecting BPTT gradients : for parameters they are partial sums over T2-K time steps
        if K != T2:
            for name, p in model.named_parameters():
                update = torch.empty_like(p).copy_(grad_or_zero(p))
                BPTT[name].append(update.unsqueeze(0))  # unsqueeze for time dimension
                neurons = copy(ref_neurons)  # Resetting the neurons to T1-T2 step
        if K != 0:
            for idx in range(len(leaf_neurons)):
                update = torch.empty_like(leaf_neurons[idx]).copy_(grad_or_zero(leaf_neurons[idx]))
                BPTT['neurons_' + str(idx)].append(update.mul(-x.size(0)).unsqueeze(0))  # unsqueeze for time dimension

    # Differentiating partial sums to get elementary parameter gradients
    for name, p in model.named_parameters():
        for idx in range(len(BPTT[name]) - 1):
            BPTT[name][idx] = BPTT[name][idx] - BPTT[name][idx + 1]

    # Reverse the time
    for key in BPTT.keys():
        BPTT[key].reverse()

    # Now we compute EP gradients forward in time
    # Second phase done step by step
    for t in range(T2):
        neurons_pre = copy(neurons)  # neurons at time step t
        neurons = model(x, y, neurons, 1, beta=beta_2, criterion=criterion)  # neurons at time step t+1

        model.compute_syn_grads(x, y, neurons_pre, neurons, betas, criterion,
                                check_thm=True)  # compute the EP parameter update

        # Collect the EP updates forward in time
        for n, p in model.named_parameters():
            update = torch.empty_like(p).copy_(grad_or_zero(p))
            EP[n].append(update.unsqueeze(0))  # unsqueeze for time dimension
        for idx in range(len(neurons)):
            update = (neurons[idx] - neurons_pre[idx]) / (beta_2 - beta_1)
            EP['neurons_' + str(idx)].append(update.unsqueeze(0))  # unsqueeze for time dimension

    # Concatenating with respect to time dimension
    for key in BPTT.keys():
        BPTT[key] = torch.cat(BPTT[key], dim=0).detach()
        EP[key] = torch.cat(EP[key], dim=0).detach()


    return BPTT, EP


def RMSE(BPTT, EP):
    # root mean square error, and sign error between EP and BPTT gradients
    for key in BPTT.keys():
        K = BPTT[key].size(0)
        f_g = (EP[key] - BPTT[key]).pow(2).sum(dim=0).div(K).pow(0.5)
        f = EP[key].pow(2).sum(dim=0).div(K).pow(0.5)
        g = BPTT[key].pow(2).sum(dim=0).div(K).pow(0.5)
        comp = f_g / (1e-10 + torch.max(f, g))
        sign = torch.where(EP[key] * BPTT[key] < 0, torch.ones_like(EP[key]), torch.zeros_like(EP[key]))

def train(model, optimizer, train_loader, test_loader, T1, T2, betas, device, epochs, criterion, alg='EP',
          save=False, check_thm=False, path='', checkpoint=None, thirdphase=False, scheduler=None):
    batch_size = train_loader.batch_size
    start = time.time()
    iter_per_epochs = math.ceil(len(train_loader.dataset) / batch_size)
    beta_1, beta_2 = betas

    if checkpoint is None:
        train_acc = [0.0]
        test_acc = [0.0]
        best = 0.0
        epoch_sofar = 0
    else:
        train_acc = checkpoint['train_acc']
        test_acc = checkpoint['test_acc']
        best = checkpoint['best']
        epoch_sofar = checkpoint['epoch']

    for epoch in range(epochs):
        run_correct = 0
        run_total = 0
        model.train()

        for idx, (x, y) in enumerate(train_loader):
            x, y = x.to(device), y.to(device)

            neurons = model.init_neurons(x.size(0), device, x)
            if alg == 'EP':
                # First phase
                neurons = model(x, y, neurons, T1, beta=beta_1, criterion=criterion)
                neurons_1 = copy(neurons)
            elif alg == 'BPTT':
                neurons = model(x, y, neurons, T1 - T2, beta=0.0, criterion=criterion)
                # detach data and neurons from the graph
                x = x.detach()
                x.requires_grad = True
                for k in range(len(neurons)):
                    neurons[k] = neurons[k].detach()
                    neurons[k].requires_grad = True

                neurons = model(x, y, neurons, T2, beta=0.0, criterion=criterion, check_thm=True)  # T2 time step

            with torch.no_grad():
                prediction = torch.argmax(neurons[-1], dim=1).squeeze()
                run_correct += (y == prediction).sum().item()
                run_total += x.size(0)

            if alg == 'EP':
                # Run Second phase
                neurons = model(x, y, neurons, T2, beta=beta_2, criterion=criterion)
                neurons_2 = copy(neurons)
                # Run third phase
                if thirdphase:
                    # we use the steady state after "free" phase as initial state as mentioned in the paper
                    neurons = copy(neurons_1)
                    neurons = model(x, y, neurons, T2, beta=- beta_2, criterion=criterion)
                    neurons_3 = copy(neurons)
                    model.compute_syn_grads(x, y, neurons_2, neurons_3, (beta_2, - beta_2), criterion)
                else:
                    model.compute_syn_grads(x, y, neurons_1, neurons_2, betas, criterion)

                optimizer.step()

            elif alg == 'BPTT':

                # final loss
                y = y.long()
                loss = 0.5 * criterion(neurons[-1].float(), F.one_hot(y, num_classes=model.nc).float()).sum(
                        dim=1).mean().squeeze()
                model.zero_grad()
                # Backpropagation through time
                loss.backward()
                optimizer.step()

            if ((idx % (iter_per_epochs // 20) == 0) or (idx == iter_per_epochs - 1)):
                train_accuracy = run_correct / run_total
                print('Epoch :', round(epoch_sofar + epoch + (idx / iter_per_epochs), 2),
                      '\tRun train acc :', round(train_accuracy, 3), '\t(' + str(run_correct) + '/' + str(run_total) + ')\t',
                      timeSince(start, ((idx + 1) + epoch * iter_per_epochs) / (epochs * iter_per_epochs)))
                if check_thm and alg != 'BPTT':
                    BPTT, EP = check_gdu(model, x[0:5, :], y[0:5], T1, T2, betas, criterion, alg=alg)
                    RMSE(BPTT, EP)

        if scheduler is not None:  # learning rate decay step
            if epoch + epoch_sofar < scheduler.T_max:
                scheduler.step()

        test_correct = test_model(model, test_loader, T1, device)
        test_accuracy = test_correct / (len(test_loader.dataset))
        if save:
            test_acc.append(100 * test_accuracy)
            train_acc.append(100 * train_accuracy)
            if test_correct > best:
                best = test_correct
                save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                            'train_acc': train_acc, 'test_acc': test_acc,
                            'best': best, 'epoch': epoch_sofar + epoch + 1}
                save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
                torch.save(save_dic, path + '/checkpoint.tar')
                torch.save(model, path + '/model.pt')
            plot_acc(train_acc, test_acc, path)

    if save:
        save_dic = {'model_state_dict': model.state_dict(), 'opt': optimizer.state_dict(),
                    'train_acc': train_acc, 'test_acc': test_acc,
                    'best': best, 'epoch': epochs}
        save_dic['scheduler'] = scheduler.state_dict() if scheduler is not None else None
        torch.save(save_dic, path + '/final_checkpoint.tar')
        torch.save(model, path + '/final_model.pt')

# Compute the accuracy of a trained model run for T time steps.
def test_model(model, data_loader, T, device):

    model.eval()
    correct = 0
    for x, y in data_loader:
        x, y = x.to(device), y.to(device)
        neurons = model.init_neurons(x.size(0), device, x)
        neurons = model(x, y, neurons, T)  # dynamics for T time steps
        prediction = torch.argmax(neurons[-1], dim=1).squeeze()
        correct += (y == prediction).sum().item()

    acc = correct / len(data_loader.dataset)
    print('Accuracy :\t', acc)
    return correct