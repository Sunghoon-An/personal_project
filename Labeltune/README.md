# Probabilistic Iterative Correction (PICO)
- - - - - - - - - - - - - -

## reference
 * https://deview.kr/2019/schedule
     * https://tv.naver.com/v/11212650/list/534045

## install
wheel 파일을 이용한 설치 예시
```sh
pip install gruls-0.1.0-py3-none-any.whl
```

## Usage
사용법

```python
# import
from grulc import IterativeCorrector

corrector = IterativeCorrector(n_cheaker=2, test_prob=0.15, verbose=2
                        , niter=4, logfile='corrector.log', save_probs='probs'
                        , epoch_reduction=14, min_epoch=2)
```
* n_cheaker(int) : cheaker의 개수 
* test_prob(float) : test 데이터의 비율 (0 ~ 0.5)
* niter(int) : 반복횟수 (epoch 아님)
* verbose(int) : 출력 옵션 0:출력없음, 1:model당 loss 출력, 2:각 모델의 학습과정까지 출력
* logfile(str) : (default : None) 로그파일명, None 일경우 로그를 남기지 않음
* epoch_reduction(float) : (default : 1) 첫번째 iteration 이후 epoch 감소량 
        1 iteration epoch : 지정된 epoch 수
        1 iteration 이후 epoch : 지정된 epoch 수 / epoch_reduction
* save_probs(str) : (default : None) 확률값을 numpy array 형태로 저장할 디렉토리 None 일경우 저장하지 않음
* min_epoch(int) : (default : 0) : 최소 epoch

```python
## train corrector
corrector.fit(model, (x, y), epoch=100, batch_size=128, monitor='accuracy'
                    , use_best=True, mode='max', augmentation=process_image
                    ,careful_update=0.8, lr_scheduler=lrscheduler
                    ,data_sampler=None)

## get new label
new_label = corrector.new_label()
```
* model(class) : (tensorflow 2.0.0 | tensorflow keras Model) 학습모형
* data(array or list) : (x_data, y_data) numpy array y_data의 경우 최소 2개 이상의 category를 필요로 함
* epoch(int) : (default : 3) 각 모델당 학습 에폭
* batch_size(int) : (default : 64) 학습 베치사이즈
* monitor(str) : (default : 'loss') 모니터링할 지표명
* use_best(bool) : (default : True) 각 iteration내 model당 가장 좋은 성능의 가중치 사용
* augmentation(func) : (default : None) 데이터 전처리 함수 ex) process_image = lambda x : np.flip(x, -1)
* lr_scheduler(func) : (default : None) 현재 epoch을 받아 learning rate을 return 하는 함수 ex) lrscheduler = lambda e : 1e-3 if e >10 else 1-e4
* careful_update(float) : (default : None) .0 ~ 1. 각 category별 최대 확률값이 지정된 값을 넘지 않으면 업데이트 하지 않음
* data_sampler(class) : (default : None) under/over sampling class, imblearn>=0.6.1 support, "fit_resample" method를 가진 RandomUnderSampler, Pipeline, ... 등 만 지원합니다.


checker의 개수만큼의 모형을 iteration만큼 반복학습하기 때문에 전체 시간은 **[epoch당 시간 x epoch x n_checker x niter]** 이상 소요됩니다. 시간을 고려하여 적절한 파라미터를 선택하십시오.

## Guide
### 1. Mnist 예제
```python
import os
import random

import numpy as np
from sklearn.metrics import confusion_matrix
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Input, Dense, Activation, Flatten
from tensorflow.keras.layers import BatchNormalization, AveragePooling2D, Conv2D, MaxPooling2D
from tensorflow.keras import Model
from tensorflow.keras.losses import categorical_crossentropy
from tensorflow.keras.regularizers import l2
from grulc import IterativeCorrector
```

```python
## define functions
def load_mnist(noise=True, noise_size=0.2, normalize=True):
    mnist = tf.keras.datasets.mnist
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    x_data = np.concatenate([x_train, x_test], axis=0)
    y_data = np.concatenate([y_train, y_test], axis=0)
    
    n = y_data.shape[0]
    random_idx = random.sample(range(n), int(n*noise_size)) 
    y_true = y_data.copy()
    
    ## add random noise
    for i in random_idx:
        tmp = list(range(10)) # [1,2,3, ...,10]
        tmp.remove(y_data[i])
        y_data[i] = random.choice(tmp)
        
    if normalize:
        x_data = x_data / 255.0
    
    return x_data, y_data, y_true


def random_flip_image(img, horizon=True, vertical=True):
    if img.ndim != 3:
        raise ValueError(f'exception flip_image: expected dim 3 but got {img.shape}')
    if horizon:
        if random.choice([True, False]):
            img = np.flip(img, 0)
    if vertical:
        if random.choice([True, False]):
            img = np.flip(img, 1)
    return img


def process_image(x):
    x = random_flip_image(x, horizon=True, vertical=True)
    return x 


def acc(m):
    return np.sum(np.diag(m))/ m.sum()


def confusion_matrix_report(true, pred):
    [[tn, fp], [fn, tp]] = confusion_matrix(true, pred)
    precision = tp/(fn+ tp)
    recall = tp/(fp+ tp)
    print(f"\t\tT\tF")
    print(f"\t1 [{tp:5d}, {fp:5d}]")
    print(f"\t0 [{fn:5d}, {tn:5d}]")
    print(f"accuracy : \t{ (tp+tn)/(tn + fp + fn + tp) :2.5f}")
    print(f"precision : \t{precision:2.5f}")
    print(f"recall : \t{recall:2.5f}")
    print(f"f1score : \t{(2*recall*precision)/(recall+precision):2.5f}")
    

class CnnModel():
    def __init__(self, name='CNN_model'):
        self.name = name
    
    
    def pooling(self, x, pool_size=3, pool_type = "max"):
        if pool_type == "max":
            x = MaxPooling2D(pool_size=pool_size, strides=(1, 1), padding='valid')(x)
        elif pool_type == "avg":
            x = AveragePooling2D(pool_size=pool_size, strides=(1, 1), padding='valid')(x)
        return x
    
    
    def conv_block(self,x, filter_size, kernels=3, stride=1 ,batch=True ,acti=True ):
        x = Conv2D(filters=filter_size,
                   kernel_size=kernels,
                   padding="SAME", strides=stride,
                   kernel_initializer='he_normal',
                   kernel_regularizer=l2(1e-4))(x)
        if batch:
            x = BatchNormalization()(x)
        if acti:
            x = Activation('relu')(x)
        return x

    
    def build(self, input_shape, out_shape) :
        
        inputs = Input(input_shape)  
        x = self.conv_block(inputs, 32, kernels=3, stride=1 ,batch=True ,acti=True )
        x = self.pooling(x, pool_type = "max")
        x = self.conv_block(x, 18, kernels=3, stride=1 ,batch=True ,acti=True )
        x = self.pooling(x, pool_type = "max")
        x = Flatten()(x)
        out = Dense(units=out_shape, activation='softmax',kernel_initializer ='he_normal')(x)
        model = Model(inputs=inputs, outputs=out)
        
        return model

```

```python
## loading data with noise
x_data, y_data, y_true = load_mnist()
x_data = x_data[:,:,:,np.newaxis] # (70000, 28, 28) -> (70000, 28, 28, 1)
y_data = to_categorical(y_data) # (70000,) -> (70000, 10)
```

```python
model = CnnModel().build((28, 28,1), 10)
adadelta = tf.keras.optimizers.Adadelta(learning_rate=1., rho=0.95, epsilon=1e-06)
model.compile(optimizer = adadelta, loss = categorical_crossentropy, metrics=['accuracy'])
model.summary()
```

```python
corrector = IterativeCorrector(n_checker=2, test_prob=0.15, verbose=1
                                , niter=4, logfile=None, save_probs='./probs'
                                , epoch_reduction=0.1, min_epoch=5)

corrector.fit(model, (x_data, y_data), epoch=10, batch_size=512, monitor='accuracy'
             , use_best=True, mode='max',careful_update=0.8
             , augmentation=process_image)
```
![출력예시](./image/output_example.PNG)

```python
new_label = corrector.new_label()
confm0 = confusion_matrix(y_true ,np.argmax(y_data, axis =1))
confm1 = confusion_matrix(y_true ,np.argmax(new_label, axis =1))
```
![결과예시](./image/result.PNG)


### 2. Generater 사용 예제
```python
import os
import h5py

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Input, Activation, BatchNormalization, multiply
from sklearn.preprocessing import MinMaxScaler ,StandardScaler
from grulc.utils.binary_dump import to_hdf5
```

```python
## csv -> h5py
to_hdf5(os.path.join('dirctory', 'preprocessed.csv'), target_column='Class', chunk_size=10000, ncategory=2, scaler=StandardScaler())
```

```python
## loading
infile = os.path.join('dirctory', 'preprocessed.h5')
data = h5py.File(infile, mode="r")
```

```python
with open("/home/gruds/projects/chan/pico/dataset/credit_card/scaler.pkl", 'rb') as f:
    standardscaler = pickle.load(f)
```

```python
## model
def self_attention(x):
    n = int(x.get_shape()[1])
    a = Dense(n, activation='sigmoid' #, name = f'attention_{i}'
              ,kernel_initializer='glorot_uniform')(x)
    return multiply([x, a])


def fc(x, n, batch = False, attention = False, acti = 'relu', kinit='he_uniform'):
    x = Dense(n, activation = acti, use_bias=True, kernel_initializer=kinit)(x)
    x = self_attention(x) if attention else x
    x = BatchNormalization()(x) if batch else x
    return x


def build(input_dim, out_dim, node=32, repeat_layer=10, use_attenton=False):
    inputs = Input(shape=(input_dim,))
    
    x = fc(inputs, node, batch=True, attention=use_attenton, acti = 'relu')
    for i in range(repeat_layer):
        x = fc(x, node, batch=True, attention=use_attenton, acti = 'relu')
    out = fc(x, out_dim, batch=False, attention=False, acti='softmax', kinit='glorot_uniform')
    
    return Model(inputs, out, name='classifier')
```

```python
model = build(len(columns), 2, node=256, repeat_layer=15, use_attenton=True)
model.compile(optimizer = tf.keras.optimizers.Adam(lr=0.001)
             ,loss = tf.keras.losses.categorical_crossentropy
             ,metrics=['accuracy'])
```

```python
corrector = IterativeCorrector(n_checker=1, test_prob=0.05, verbose=1
                                 , niter=4, logfile=None, save_probs='./probs'
                                 , epoch_reduction=0.3, min_epoch=10)

corrector.fit(model, data, epoch=10, batch_size=256, monitor='f1-score',
                      use_best=True, mode='max', scaler=standardscaler,
                      lr_scheduler=None, careful_update=0.9, data_sampler=None
                     )
```


[TODO] 
- [ ] pandas DataFrame support
- [x] generator support
- [x] install 방법