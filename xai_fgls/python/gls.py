import matplotlib.pyplot as plt
import numpy
import time
import numpy as np
import importlib.util as imp
if imp.find_spec("cupy"): #use cupy for GPU support if available
    import cupy
    import cupy as np

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


#load a neural network, as well as the MNIST test data and some labels
nn = model_io.read('../models/MNIST/LeNet-5.nn') # 99.23% prediction accuracy

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# nn = nn().to(device)

# nn.summary()
nn.drop_softmax_output_layer() #drop softnax output layer for analyses
X = data_io.read('../data/MNIST/test_images.npy')
Y = data_io.read('../data/MNIST/test_labels.npy')


# transfer pixel values from [0 255] to [-1 1] to satisfy the expected input / training paradigm of the model
# 픽셀 벨류를 0 ~ 255인 애들에서 -1과 1의 값으로 변경
X =  X / 127.5 - 1.

#reshape the vector representations in X to match the requirements of the CNN input
# CNN에 들어갈 수 있도록 Vector를 Reshape(array로)
X = np.reshape(X,[X.shape[0],28,28,1])
# 패딩 진행
X = np.pad(X,((0,0),(2,2),(2,2),(0,0)), 'constant', constant_values = (-1.,))

# transform numeric class labels to vector indicator for uniformity. assume presence of all classes within the label set
# 클래스를 Label화 시키기 위해 Unique 값 지정 및 0으로 초기화
I = Y[:,0].astype(int)
Y = np.zeros([X.shape[0],np.unique(Y).size])
Y[np.arange(Y.shape[0]),I] = 1
print("check {}: :.".format(X.shape))
acc = np.mean(np.argmax(nn.forward(X), axis=1) == np.argmax(Y, axis=1))
# 가장 높은 값의 평균을 가져오고, 예측값 진행(argmax) forward는 pytorch로 코딩한 것.
if not np == numpy: # np=cupy
    acc = np.asnumpy(acc)
print('model test accuracy is: {:0.4f}'.format(acc))

I = np.arange(X.shape[0])

for i in I[:10]:
    # i의 순번을 지정하고
    x = X[i:i+1,...]
    print("x shape is : {}".format(x.shape))

    #forward pass and prediction
    # 예측하고
    ypred = nn.forward(x)
    print('True Class:     ', np.argmax(Y[i]))
    print('Predicted Class:', np.argmax(ypred),'\n')

    #prepare initial relevance to reflect the model's dominant prediction (ie depopulate non-dominant output neurons)
    # zeros_like를 사용해서 다른 배열과 같은 크기의 배열을 0로 생성하고
    mask = np.zeros_like(ypred)
    print("lrp mask shape {}: ".format(mask.shape))

    # argmax를 사용해서 가장 큰 값 가져오고
    mask[:,np.argmax(ypred)] = 1
    print("mask shape is : {}".format(mask.shape))
    # 초기 Relevance Score 지정하고
    Rinit = ypred*mask
    print("Lrp R shape {} : ".format(Rinit.shape))
    #compute first layer relevance according to prediction
    #R = nn.lrp(Rinit)                   #as Eq(56) from DOI: 10.1371/journal.pone.0130140
    R = nn.lrp(Rinit,'epsilon',1.)

    R = R.sum(axis=3)
    xs = ((x+1.)/2.).sum(axis=3)

    if not np == numpy: 
        xs = np.asnumpy(xs)
        R = np.asnumpy(R)

    digit = render.digit_to_rgb(xs, scaling = 3)
    hm = render.hm_to_rgb(R, X = xs, scaling = 3, sigma = 2)
    digit_hm = render.save_image([digit,hm],'../heatmap.png')
    data_io.write(R,'../heatmap.npy')
    data_io.write(xs,'../xs.npy')
    print(xs.shape)
    y = xs
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
    y = np.reshape(y, [y.shape[0]*y.shape[1]*y.shape[2]])
    y_tran = y.transpose()
    new = sm.add_constant(new)
#     new = np.nan_to_num(new)
#     y = np.nan_to_num(y)
#     model = sm.GLSAR(y_tran, new, rho = 2, missing = "raise")
    print("y_tran shape : ")
    print(y_tran.shape)
    print("X shape : ")
    print(new.shape)
    model = sm.GLSAR(y_tran, new, rho = 2)
    result = model.iterative_fit(maxiter = 5000)
#     for i in range(30):
#         result = model.fit()
#         print("AR coefficients : {0}".format(model.rho))
#         rho, sigma = sm.regression.yule_walker(result.resid, order = model.order)
#         print("{}step is done".format(i))
#         model = sm.GLSAR(y_tran, new, rho)
    find = result.resid
    check = np.reshape(find,[1,32,32])
    print(xs.shape)
    digit = render.digit_to_rgb(xs, scaling = 3)
    LRP_test = render.digit_to_rgb(R, scaling = 3)
    proposed_test = render.digit_to_rgb(check, scaling = 3)
    hm_digit = render.hm_to_rgb(xs, X = xs, scaling = 3, sigma = 2)
    hm_R = render.hm_to_rgb(check, X = xs, scaling = 3, sigma = 2)
    print("raw shape is {}".format(xs.shape))
    print("LRP R shape is {}".format(R.shape))
    print("Propose shape is {}".format(check.shape))
    print("digit shape is {}".format(digit.shape))
    print("remaid shape is {}".format(hm_digit.shape))
    print("lrp shape is {}".format(hm.shape))
    print("f-lrp shape is {}".format(hm_R.shape))
    digit_hm = render.save_image([hm_digit,hm, hm_R],'../re_heatmap.png')
    
    data_io.write(check,'../re_heatmap.npy')
    
    compare_images(digit, LRP_test, "Original vs LRP")
    compare_images(digit, proposed_test, "Original vs feasible LRP")
    test_image = render.save_image([digit, LRP_test, proposed_test],'../checking_image.png')

    plt.imshow(test_image, interpolation = 'none')
    plt.axis('off')
    plt.show()

if True:
    N = 256
    t_start = time.time()
    x = X[:N,...]
    y = nn.forward(x)
    R = nn.lrp(y)
    data_io.write(R,'../Rbatch.npy')
    print('Computation of {} heatmaps using {} in {:.3f}s'.format(N, np.__name__, time.time() - t_start))