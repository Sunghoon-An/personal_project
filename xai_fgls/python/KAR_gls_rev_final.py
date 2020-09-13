import matplotlib.pyplot as plt
import numpy
import time
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np
    print ("ok")

from tootorch.dataload import mnist_load
from tootorch.utils import seed_everything, get_samples, ModelTrain
import model_io
import data_io
import render
import statsmodels.api as sm
import torch
# from torchvision import models
from torchvision import models
from torchsummary import summary
from config import *
from tqdm import tqdm
import pickle
import modules


import os, math, collections

import tensorflow as tf
logger = get_logger(os.path.basename(__file__))
np.random.seed(42)
def unpickle(file):
    
    with open(file, 'rb') as fo:
        
        dict = pickle.load(fo, encoding='bytes')
    
    return dict

def save(dataset, file):
    
    with open(file, 'wb') as fo:
        
        pickle.dump(dataset, fo)
        

def scale(x):
    
    return x / 127.5 - 1.0        

def scaling(x):
    xr = 2 * ((x - x.min())/(x.max() - x.min()))-1
    return xr


def random_remove(images, percent, keep=False):
    im = np.copy(images)
    img_size = im.shape[1:]
    mask = np.random.binomial(1, (1 - percent), size=im.shape[:-1])

    if keep:
        im[mask == 1] = train_mean
    else:
        im[mask == 0] = train_mean

    return im
    
def remove(images, attributions, percent, keep=False, random=False):
    '''
    images       : tensor of shape [N,H,W,C]
    attributions : tensor of shape [N,H,W]
    percentile   : scalar between 0 and 100, inclusive
    keep         : if true keep q percent; otherwise remove q percent
    '''
    im = np.copy(images)
    img_size = im.shape[1:]

    nb_pixel = np.prod(img_size)
    threshold = int(nb_pixel * (1 - percent))
    
    re_sal_maps = attributions.reshape(attributions.shape[0], -1)
    indice = re_sal_maps.argsort().argsort()
    
    if keep:
        mask = indice >= threshold
    else:
        mask = indice < threshold
        
    mask = mask.reshape(im.shape)
#     logger.info("num threshold is : {}".format(threshold))
#     logger.info("num mask is : {}".format(np.sum(mask)))

    images = (im * mask).reshape(images.shape)

#     logger.info("image max is : {}".format(images.max()))
#     logger.info("image min is : {}".format(images.min()))

    return images


def occlude_dataset(DNN, attribution, percentiles, test=False, keep=False, random=False, batch_size= 64, savedir=''):
    print("Condition of test : {}".format(test))
    if test:
        Xs = Xtest
        ys = Ytest
    else:
        Xs = Xtrain
        ys = Ytrain
    
    print("initial batch_size is : {}".format(batch_size))
    total_batch = math.ceil(len(Xs) / batch_size)
    print("batch size is :{}".format(total_batch))
    hmaps = []
    data = []
    label = []
    hmaps_00 = []
    hmaps_01 = []
    hmaps_02 = []
    hmaps_03 = []
    hmaps_04 = []
    hmaps_05 = []
    hmaps_06 = []
    hmaps_07 = []
    hmaps_08 = []
    hmaps_09 = []
    hmaps_10 = []
    data_00 = []
    data_01 = []
    data_02 = []
    data_03 = []
    data_04 = []
    data_05 = []
    data_06 = []
    data_07 = []
    data_08 = []
    data_09 = []
    data_10 = []
    label_00 = []
    label_01 = []
    label_02 = []
    label_03 = []
    label_04 = []
    label_05 = []
    label_06 = []
    label_07 = []
    label_08 = []
    label_09 = []
    label_10 = []
    for i in tqdm(range(total_batch)):
        if 'LRP' in attribution:
            x = Xs[i:i+1,...]
            y = ys[i:i+1,...]
            ypred = DNN.forward(x)
            m = np.zeros_like(ypred)
            m[:,np.argmax(ypred)] = 1
            Rinit = ypred*m
            Rinit.astype(np.float)
            R = DNN.lrp(Rinit,'epsilon',1.)
            R = R.sum(axis=3)
            if not np == numpy:
                R = np.asnumpy(R)
            
            attrs = R
            data = x
            
            attrs = scaling(attrs)
            attrs *= 255
            attrs = attrs.astype(np.uint8)
            attrs = scale(attrs)



        elif 'proposed_method' in attribution:
            x = Xs[i:i+1,...]
            y = ys[i:i+1,...]
            ypred = DNN.forward(x)

            m = np.zeros_like(ypred)
            m[:,np.argmax(ypred)] = 1
            Rinit = ypred*m
            Rinit.astype(np.float)
            R = DNN.lrp(Rinit,'epsilon',1.)
            R = R.sum(axis=3)
            if not np == numpy: 
                xs = np.asnumpy(x)
                R = np.asnumpy(R)
            xs = x
            tar = xs
            a = np.load('../r_array/convolution.npy')
            a = np.reshape(a,[a.shape[1]*a.shape[2],1])
            b = np.load('../r_array/rect.npy')    
            b = np.pad(b,((0,0),(2,2),(2,2),(0,0)))
            b = np.reshape(b,[b.shape[1]*b.shape[2],b.shape[0]*b.shape[3]])
            c = np.load('../r_array/sumpoll.npy')
            c = np.pad(c,((0,0),(2,2),(2,2),(0,0)))
            c = np.reshape(c,[c.shape[1]*c.shape[2],c.shape[3]])
            new_b = np.hstack((b, c))
            new = np.hstack((a, new_b))
            tar = np.reshape(tar, [tar.shape[0]*tar.shape[1]*tar.shape[2]])
            y_tran = tar.transpose()
            new = sm.add_constant(new)

            model = sm.GLSAR(y_tran, new, rho = 2)
            result = model.iterative_fit()
            find = result.resid
            check = np.reshape(find,[1,32,32])
            
            attrs = check
            data = x
            
            attrs = scaling(attrs)
            attrs *= 255
            attrs = attrs.astype(np.uint8)
            attrs = scale(attrs)

            
        else:
            x = Xs[i:i+1,...]
            y = ys[i:i+1,...]
            if not np == numpy:
                xs = np.asnumpy(x)
            xs = x
            
            attrs = xs
            data = x
            
#             attrs = scaling(attrs)
            attrs *= 255
            attrs = attrs.astype(np.uint8)
            attrs = scale(attrs)


        for percent in tqdm(percentiles):
            batch_attrs = attrs
            data = data
            if attribution == 'normal':
                if percent == 0.0:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_00.append(occluded_images)
                    data_00.append(data)
                    label_00.append(y)
                elif percent == 0.1:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_01.append(occluded_images)
                    data_01.append(data)
                    label_01.append(y)
                elif percent == 0.2:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_02.append(occluded_images)
                    data_02.append(data)
                    label_02.append(y)
                elif percent == 0.3:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_03.append(occluded_images)
                    data_03.append(data)
                    label_03.append(y)
                elif percent == 0.4:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_04.append(occluded_images)
                    data_04.append(data)
                    label_04.append(y)
                elif percent == 0.5:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_05.append(occluded_images)
                    data_05.append(data)
                    label_05.append(y)
                elif percent == 0.6:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_06.append(occluded_images)
                    data_06.append(data)
                    label_06.append(y)
                elif percent == 0.7:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_07.append(occluded_images)
                    data_07.append(data)
                    label_07.append(y)
                elif percent == 0.8:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_08.append(occluded_images)
                    data_08.append(data)
                    label_08.append(y)
                elif percent == 0.9:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_09.append(occluded_images)
                    data_09.append(data)
                    label_09.append(y)
                elif percent == 1:
                    occluded_images = random_remove(data, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_10.append(occluded_images)
                    data_10.append(data)
                    label_10.append(y)
                else:
                    raise ValueError("attribution?")
            else:
                if percent == 0.0:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_00.append(occluded_images)
                    data_00.append(data)
                    label_00.append(y)
                elif percent == 0.1:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_01.append(occluded_images)
                    data_01.append(data)
                    label_01.append(y)
                elif percent == 0.2:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_02.append(occluded_images)
                    data_02.append(data)
                    label_02.append(y)
                elif percent == 0.3:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_03.append(occluded_images)
                    data_03.append(data)
                    label_03.append(y)
                elif percent == 0.4:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_04.append(occluded_images)
                    data_04.append(x)
                    label_04.append(y)
                elif percent == 0.5:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_05.append(occluded_images)
                    data_05.append(x)
                    label_05.append(y)
                elif percent == 0.6:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_06.append(occluded_images)
                    data_06.append(x)
                    label_06.append(y)
                elif percent == 0.7:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_07.append(occluded_images)
                    data_07.append(x)
                    label_07.append(y)
                elif percent == 0.8:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_08.append(occluded_images)
                    data_08.append(x)
                    label_08.append(y)
                elif percent == 0.9:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_09.append(occluded_images)
                    data_09.append(x)
                    label_09.append(y)
                elif percent == 1:
                    print(" percent : {}".format(percent))
                    occluded_images = remove(data, batch_attrs, percent, keep)
                    raw_image = render.digit_to_rgb(data, scaling = 1)
                    prdigit = render.digit_to_rgb(occluded_images, scaling = 1)
                    test_image = render.save_image([raw_image, prdigit],'../{}_KAR_check_point_{}.png'.format(attribution, percent))
                    hmaps_10.append(occluded_images)
                    data_10.append(x)
                    label_10.append(y)
                else:
                    raise ValueError("Error : {}".format(percent))

        
        
        
#         hmaps.append(attrs)
#         data.append(x)
#         label.append(y)
#         print("print final : {}".format(len(hmaps)))
    print("Interpretation is done, concatenate...")
    hmaps_00 = np.concatenate(hmaps_00, axis=0)
    hmaps_01 = np.concatenate(hmaps_01, axis=0)
    hmaps_02 = np.concatenate(hmaps_02, axis=0)
    hmaps_03 = np.concatenate(hmaps_03, axis=0)
    hmaps_04 = np.concatenate(hmaps_04, axis=0)
    hmaps_05 = np.concatenate(hmaps_05, axis=0)
    hmaps_06 = np.concatenate(hmaps_06, axis=0)
    hmaps_07 = np.concatenate(hmaps_07, axis=0)
    hmaps_08 = np.concatenate(hmaps_08, axis=0)
    hmaps_09 = np.concatenate(hmaps_09, axis=0)
    hmaps_10 = np.concatenate(hmaps_10, axis=0)
    data_00 = np.concatenate(data_00, axis = 0)
    data_01 = np.concatenate(data_01, axis = 0)
    data_02 = np.concatenate(data_02, axis = 0)
    data_03 = np.concatenate(data_03, axis = 0)
    data_04 = np.concatenate(data_04, axis = 0)
    data_05 = np.concatenate(data_05, axis = 0)
    data_06 = np.concatenate(data_06, axis = 0)
    data_07 = np.concatenate(data_07, axis = 0)
    data_08 = np.concatenate(data_08, axis = 0)
    data_09 = np.concatenate(data_09, axis = 0)
    data_10 = np.concatenate(data_10, axis = 0)
    label_00 = np.concatenate(label_00, axis = 0)
    label_01 = np.concatenate(label_01, axis = 0)
    label_02 = np.concatenate(label_02, axis = 0)
    label_03 = np.concatenate(label_03, axis = 0)
    label_04 = np.concatenate(label_04, axis = 0)
    label_05 = np.concatenate(label_05, axis = 0)
    label_06 = np.concatenate(label_06, axis = 0)
    label_07 = np.concatenate(label_07, axis = 0)
    label_08 = np.concatenate(label_08, axis = 0)
    label_09 = np.concatenate(label_09, axis = 0)
    label_10 = np.concatenate(label_10, axis = 0)
    
#     hmaps = np.concatenate(hmaps, axis=0)
#     data = np.concatenate(data, axis = 0)
    
    print("concatenate is done...")
#     print("print final : {}".format(hmaps.shape))
#     percentiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]
    for percent in tqdm(percentiles):
        if percent == 0.0:
            save(hmaps_00, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_00, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.1:
            save(hmaps_01, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_01, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.2:
            save(hmaps_02, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_02, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.3:
            save(hmaps_03, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_03, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.4:
            save(hmaps_04, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_04, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.5:
            save(hmaps_05, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_05, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.6:
            save(hmaps_06, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_06, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.7:
            save(hmaps_07, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_07, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.8:
            save(hmaps_08, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_08, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 0.9:
            save(hmaps_09, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_09, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        elif percent == 1:
            save(hmaps_10, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
            save(label_10, savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
            print("Occlude image {} percentile...".format(percent))
        else:
            print("error")
    del hmaps_00,hmaps_01,hmaps_02,hmaps_03,hmaps_04,hmaps_05,hmaps_06,hmaps_07,hmaps_08,hmaps_09,hmaps_10,data_00,data_01,data_02,data_03,data_04,data_05,data_06,data_07,data_08,data_09,data_10,label_00,label_01,label_02,label_03,label_04,label_05,label_06,label_07,label_08,label_09,label_10
#     for percent in tqdm(percentiles):
#         batch_attrs = hmaps
#         occluded_images = remove(data, batch_attrs, percent, keep)
#         print("save start")
#         print("Save directory is {}".format(savedir))
#         save(occluded_images, savedir + '{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent))
#         save(np.concatenate(label, axis = 0), savedir + '{}_{}_{}_{}.pickle'.format('test' if test else 'train', attribution, percent, 'label'))
#         print("Occlude image {} percentile...".format(percent))
        
def roar_kar(keep, random = False, train_only=False):
    
    logdir = 'tf_logs/standard/'
    
    def get_savedir():
        
        savedir = logdir.replace('tf_logs', 'KAR' if keep else 'ROAR')
        
        if not os.path.exists(savedir):
            
            os.makedirs(savedir)
        
        return savedir
#     ratio = 0.1
    percentiles = [0.0, 0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]
    attribution_methods = ['normal', 'LRP', 'proposed_method']
    
    if not train_only:
        DNN = model_io.read('../models/MNIST/LeNet-5.nn')
        for v in attribution_methods:
            batch_size = 64
            print("{} Step is start".format(v))
            if v == 'normal':
                print("{} Random Remove".format(v))
                occlude_dataset(DNN = DNN, attribution = v, percentiles = percentiles, random = False, keep = keep, batch_size = batch_size, savedir = get_savedir())
            else:
                print("{} percentile Remove".format(v))
                occlude_dataset(DNN = DNN, attribution = v, percentiles = percentiles, random = True, keep = keep, batch_size = batch_size, savedir = get_savedir())
            print("{} : occlude step is done".format(v))
        print("ress record")
    ress = {k : [] for k in attribution_methods}
    
    for _ in range(3):
    
        for v in attribution_methods:
            logger.info("=============================================")
            logger.info("[{} KAR start] : {}".format(v, str(datetime.datetime.now())))
            st = time.time()
            res = []

            for p in percentiles:
                logger.info("=============================================")
                logger.info("[{} KAR percentile] : {}".format(v, p))
                start = time.time()
                occdir = get_savedir() + '{}_{}_{}.pickle'.format('{}', v, p)
                occdir_y = get_savedir() + '{}_{}_{}_{}.pickle'.format('{}', v, p,'label')

                Xtrain = unpickle(occdir.format('train'))
                Xtrain = np.array(Xtrain)
                Ytrain = unpickle(occdir_y.format('train'))
                Ytrain = np.array(Ytrain)
                
                Xtrain = scale(Xtrain)
                
                print("check : {}".format(Ytrain.shape))
                
                DNN = model_io.read('../models/MNIST/LeNet-5.nn')
# DNN.drop_softmax_output_layer()

#                 DNN = modules.Sequential([
#                                 modules.Convolution(filtersize=(5,5,1,10),stride = (1,1)),\
#                                 modules.Rect(),\
#                                 modules.SumPool(pool=(2,2),stride=(2,2)),\
#                                 modules.Convolution(filtersize=(5,5,10,25),stride = (1,1)),\
#                                 modules.Rect(),\
#                                 modules.SumPool(pool=(2,2),stride=(2,2)),\
#                                 modules.Convolution(filtersize=(4,4,25,100),stride = (1,1)),\
#                                 modules.Rect(),\
#                                 modules.SumPool(pool=(2,2),stride=(2,2)),\
#                                 modules.Convolution(filtersize=(1,1,100,10),stride = (1,1)),\
#                                 modules.Flatten()
#                             ])
                
                print("training...")
                DNN.train(X=Xtrain,\
                    Y=Ytrain,\
                    Xval=Xtest,\
                    Yval=Ytest,\
                    iters=100,\
                    lrate=0.0001,\
                    status = 25,\
                    batchsize = 64
                         )
#                 ypred = DNN.forward(Xtest)
                DNN.drop_softmax_output_layer()

                acc = np.mean(np.argmax(DNN.forward(Xtest), axis=1) == np.argmax(Ytest, axis=1))
                del DNN
                logger.info('metric model test accuracy is: {:0.4f}'.format(acc))

                res.append(acc)
                sec = time.time() - start
                h = int(sec//(60*60))
                m = int((sec-(h*60*60))//(60))
                s = int((sec-(h*60*60)-(m*60)))

                logger.info('percent : {} Done, total time [{}:{}:{}]'.format(p,h,m,s))
            logger.info("End of {}:training, accuracy...".format(_))
            sec = time.time() - st
            h = int(sec//(60*60))
            m = int((sec-(h*60*60))//(60))
            s = int((sec-(h*60*60)-(m*60)))

            logger.info('method : {} Done, total time [{}:{}:{}]'.format(v,h,m,s))

            ress[v].append(res)
    logger.info("metric...")
    res_mean = {v: np.mean(v, axis=0) for v in ress}
    
    logger.info(res_mean)
    
    return res_mean
logger.info("=======================================================")
logger.info("[KAR start] : {}".format(str(datetime.datetime.now())))

Xtrain = data_io.read('../data/MNIST/train_images.npy')
Ytrain = data_io.read('../data/MNIST/train_labels.npy')

Xtest = data_io.read('../data/MNIST/test_images.npy')
Ytest = data_io.read('../data/MNIST/test_labels.npy')

Xtrain = scale(Xtrain)
Xtest = scale(Xtest)

Xtrain = np.reshape(Xtrain,[Xtrain.shape[0],28,28,1])
Xtest = np.reshape(Xtest,[Xtest.shape[0],28,28,1])
Xtrain = np.pad(Xtrain,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))
Xtest = np.pad(Xtest,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

train_mean = np.mean(Xtrain, axis=(0,1,2))

It = Ytrain[:,0].astype(int)
Ytrain = np.zeros([Xtrain.shape[0],np.unique(Ytrain).size])
Ytrain[np.arange(Ytrain.shape[0]),It] = 1
Ix = Ytest[:,0].astype(int)
Ytest = np.zeros([Xtest.shape[0],np.unique(Ytest).size])
Ytest[np.arange(Ytest.shape[0]),Ix] = 1

DNN = model_io.read('../models/MNIST/LeNet-5.nn')
DNN.drop_softmax_output_layer()


results = roar_kar(keep = True, random = False, train_only = True)

logger.info("================final check KAR is above================")
logger.info(results)