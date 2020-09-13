import h5py
import numpy as np
import json

from .saliency.attribution_methods import *
from .models import SimpleCNNDeconv

import matplotlib.pyplot as plt
import numpy
import time
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np


import model_io
import data_io
import render
import statsmodels.api as sm
import torch
# from torchvision import models
from torchvision import models
from torchsummary import summary
from config import *


def visualize_ROARnKAR(targets, ratio_lst, eval_method, methods=None, attention=None, savedir=None, **kwargs):
    '''
    Visualize ROAR or KAR 
    Args:
        dataset: target dataset. ['mnist','cifar10']
        ratio_lst: pixel ratio list
        eval_method: ['ROAR','KAR']
        methods: attribution methods 
        attention: attention method
        savedir: save path and save name
    '''
    if methods==None:
        assert attention!=None, 'If methods is None, attention should not be None'
        methods = attention
    elif attention==None:
        assert methods!=None, 'If methods is None, attention should not be None'
    else:
        t_methods = methods + attention
        methods = t_methods.copy()

    # if attention is not None, define methods list
    for i in range(len(methods)):
        if methods[i] == 'LRP':
            methods[i] = 'LRP'
        else:
            methods[i] = 'proposed_method'

    # initialize
    fontsize = 10 if 'fontsize' not in kwargs.keys() else kwargs['fontsize']
    size = (5,5) if 'size' not in kwargs.keys() else kwargs['size']
    color = ['b', 'g', 'r', 'c', 'm', 'y', 'k', 'w'] if 'color' not in kwargs.keys() else kwargs['color']
    marker = ['o','v','^','s','x','*','p','d'] if 'marker' not in kwargs.keys() else kwargs['marker']
    random_state = 223 if 'random_state' not in kwargs.keys() else kwargs['random_state']
    dpi = None if 'dpi' not in kwargs.keys() else kwargs['dpi']

    # initialize methods acc by targets [mnist, cifar10]
    test_acc = {target: {} for target in targets}
    for target in targets:
        test_acc[target] = {m: [] for m in methods}

    # load test accuracy
    def get_test_acc(methods, targets, test_acc):
        for m in methods:
            for target in targets:
                model_name = '{}_{}'.format('simple_cnn', target)

                acc = np.mean(np.argmax(nn.forward(X), axis=1) == np.argmax(Y, axis=1))
                # 가장 높은 값의 평균을 가져오고, 예측값 진행(argmax) forward는 pytorch로 코딩한 것.
                if not np == numpy: # np=cupy
                    acc = np.asnumpy(acc)
                test_acc[target][m].append(acc)
        return test_acc


    # load roar/kar accuracy
    def get_roar_kar_test_acc(methods, targets, test_acc):
        for target in targets:
            for m in methods:
                if (('RAN' in m) or ('WARN' in m)) & (m != 'RANDOM'):
                    model_name = '{}_{}'.format(target, m)
                else:
                    model_name = '{}_{}_{}'.format('simple_cnn', target, m)

                for ratio in ratio_lst[1:-1]:
                    f = open('../logs/{0:}_{1:}{2:.1f}_logs.txt'.format(model_name, eval_method, ratio),'r')
                    test_acc[target][m].append(json.load(f)['test_result'])
        return test_acc

    # insert 0 
    def get_0_test_acc(methods, targets, test_acc):
        for target in targets:
            for m in methods:
                test_acc[target][m].append(0)
        return test_acc

    # ROAR or KAR
    if eval_method=='ROAR':
        test_acc = get_test_acc(methods, targets, test_acc)
        test_acc = get_roar_kar_test_acc(methods, targets, test_acc)
        test_acc = get_0_test_acc(methods, targets, test_acc)
    elif eval_method=='KAR':
        test_acc = get_0_test_acc(methods, targets, test_acc)
        test_acc = get_roar_kar_test_acc(methods, targets, test_acc)
        test_acc = get_test_acc(methods, targets, test_acc)

    # plotting
    f, ax = plt.subplots(1,2,figsize=size)
    for i in range(len(targets)):
        for j in range(len(methods)):
            if methods[j] == 'RANDOM':
                ax[i].plot(ratio_lst, test_acc[targets[i]][methods[j]], label=methods[j], color='black', linestyle='--', linewidth=3, alpha=0.5)
            else:
                ax[i].plot(ratio_lst, test_acc[targets[i]][methods[j]], label=methods[j], color=color[j], marker=marker[j])
        ax[i].set_title(f'{eval_method} : {targets[i].upper()}', size=fontsize)
        ax[i].set_ylabel('Test accuracy', size=fontsize)
        ax[i].set_xlabel(f'% of input features replaced', size=fontsize)
        ax[i].set_xlim([0,1])
        ax[i].set_ylim([0,1])
        ax[i].legend(loc='upper right')
    if savedir:
        plt.tight_layout()
        plt.savefig(savedir, dpi=dpi)