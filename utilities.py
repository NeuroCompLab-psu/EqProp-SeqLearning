import torch
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import torchvision
import torch.nn.functional as F
import matplotlib
from matplotlib.pyplot import *
#rcParams['mathtext.rm'] = 'sans-serif'

import os
from datetime import datetime
import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return 'elapsed time : %s \t (will finish in %s)' % (asMinutes(s), asMinutes(rs))


def get_estimate(dic):
    estimates = {}
    for key in dic.keys():
        if key.find('weight')!=-1:
            estimate = integrate(dic[key])
            estimates[key] = estimate[-1,:]
    return estimates
    

def half_sum(dic1, dic2):
    dic3 = {}
    for key in dic1.keys():
        dic3[key] = (dic1[key] + dic2[key])/2
    return dic3

def compare_estimate(bptt, ep_1, ep_2, path):
    heights=[]
    abscisse=[]
    plt.figure(figsize=(16,9))
    for key in bptt.keys():
        
        ep_3 = (ep_1[key]+ep_2[key])/2
        
        ep1_bptt = (ep_1[key] - bptt[key]).abs()
        ep2_bptt = (ep_2[key] - bptt[key]).abs()
        ep3_bptt = (ep_3 - bptt[key]).abs()

        comp = torch.where( (ep1_bptt + ep2_bptt)==0, torch.ones_like(ep1_bptt), (2*ep3_bptt)/(ep1_bptt + ep2_bptt) )
        comp = comp.mean().item()

        if key.find('weight')!=-1:
            heights.append(comp)
            abscisse.append(int(key[9])+1)

    plt.bar(abscisse, heights)
    plt.ylim((0.,1.))
    plt.title('Euclidian distance between EP symmetric and BPTT, divided by mean distance between EP one-sided and BPTT\n 1.0 means EP symmetric is as close to BPTT as EP one-sided, 0.5 means EP symmetric twice closer to BPTT than EP one-sided')
    plt.ylabel('Relative distance to BPTT')
    plt.xlabel('Layer index')
    plt.savefig(path+'/bars.png', dpi=300)
    plt.close()

def integrate(x):
    y = torch.empty_like(torch.tensor(x))
    with torch.no_grad():
        for j in reversed(range(x.shape[0])):
            integ=0.0
            for i in range(j):
                integ += x[i]
            y[j] = integ
    return y


def plot_gdu(BPTT, EP, path, EP_2=None, alg='EP'):

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    plt.rcParams.update({'font.size': 18})
    for key in EP.keys():
        if 'synapse' in key:
            plt.rcParams.update({'font.family': 'sans-serif'})
            fig = plt.figure(figsize=(10,7))
            #ax = fig.add_subplot(1, 1, 1)
            sns.set_theme()
            sns.axes_style({'font.family': ['sans-serif'],
                            'font.sans-serif': ['Arial']
                            })
            for idx in range(3):
                if len(EP[key].size())==3:
                    i, j = np.random.randint(EP[key].size(1)), np.random.randint(EP[key].size(2))
                    ep = EP[key][:,i,j].cpu().detach()
                    if EP_2 is not None:
                        ep_2 = EP_2[key][:,i,j].cpu().detach()
                    bptt = BPTT[key][:,i,j].cpu().detach()
                elif len(EP[key].size())==2:
                    i = np.random.randint(EP[key].size(1))
                    ep = EP[key][:,i].cpu().detach()
                    if EP_2 is not None:
                        ep_2 = EP_2[key][:,i].cpu().detach()
                    bptt = BPTT[key][:,i].cpu().detach()
                elif len(EP[key].size())==5:
                    i, j = np.random.randint(EP[key].size(1)), np.random.randint(EP[key].size(2))
                    k, l = np.random.randint(EP[key].size(3)), np.random.randint(EP[key].size(4))
                    ep = EP[key][:,i,j,k,l].cpu().detach()
                    if EP_2 is not None:
                        ep_2 = EP_2[key][:,i,j,k,l].cpu().detach()
                    bptt = BPTT[key][:,i,j,k,l].cpu().detach()
                ep, bptt = integrate(ep), integrate(bptt)
                ep, bptt = ep.numpy().flatten(), bptt.numpy().flatten()
                #plt.plot(ep, linestyle=':', linewidth=2, color=colors[idx], alpha=0.7, label=alg+' with B ')
                plt.plot(bptt, color=colors[idx], linewidth=2, alpha=0.7, label='BPTT')
                if EP_2 is not None:
                    ep_2 = integrate(ep_2)
                    ep_2 = ep_2.numpy().flatten()
                    #plt.plot(ep_2, linestyle='-.', linewidth=2, color=colors[idx], alpha=0.7, label=alg+' one-sided left')
                    plt.plot((ep + ep_2)/2, linestyle='--', linewidth=2, color=colors[idx], alpha=0.7, label=alg+' symmetric')
                #plt.title(key.replace('.',' '))
            major_ticks = np.arange(0, 21, 4)
            plt.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
            plt.xticks(major_ticks)
            #plt.grid()
            #hfont = {'fontname': 'Helvetica'}
            plt.xlabel('Time Step in Second Phase', fontsize=16)
            plt.ylabel('Gradient Estimate', fontsize=16)
            plt.legend(fontsize=14)
            fig.savefig(path+'/'+key.replace('.','_')+'.png', dpi=300)
            plt.close()

def plot_synapses(model, path):   
    N = len(model.synapses)
    fig = plt.figure(figsize=(4*N,3))
    for idx in range(N):
        fig.add_subplot(1, N, idx+1)
        nrn = model.synapses[idx].weight.cpu().detach().numpy().flatten()
        plt.hist(nrn, 50)
        plt.title('synapses of layer '+str(idx+1))
    fig.savefig(path)
    plt.close()

def plot_acc(train_acc, test_acc, path):
    fig = plt.figure(figsize=(16,9))
    x_axis = [i for i in range(len(train_acc))]
    plt.plot(x_axis, train_acc, label='train')
    plt.plot(x_axis, test_acc, label='test')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid()
    fig.savefig(path + '/train-test_acc.png')
    plt.close()









 
