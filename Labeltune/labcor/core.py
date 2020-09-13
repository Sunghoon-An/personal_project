#-*- coding:utf-8 -*-

import sys
import os
import time
import random
import logging
import h5py
from collections import Counter
from multiprocessing import Pool

import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import to_categorical

from .utils import logging_format


class IterativeCorrector():
    """Probabilistic Iterative Correction (PICO)
    
    [usage]
    ```python
    # import
    from grulc import IterativeCorrector
    
    corrector = IterativeCorrector(n_cheaker=2, test_prob=0.15, verbose=2
                            , niter=4, logfile='corrector.log'
                            , epoch_reduction=14, min_epoch=2)
    ```
    * n_cheaker(int) : cheaker의 개수 
    * test_prob(float) : test 데이터의 비율 (0 ~ 0.5)
    * niter(int) : 반복횟수 (epoch 아님)
    * verbose(int) : 출력 옵션 0:출력없음, 1:model당 loss 출력, 2:각 모델의 학습과정까지 출력
    * logfile(str) : (default : None) 로그파일명, None 일경우 로그를 남기지 않음
    * save_probs(str) : (default : None) 확률값을 numpy array 형태로 저장할 디렉토리 None 일경우 저장하지 않음
    * epoch_reduction(float) : (default : 1) 첫번째 iteration 이후 epoch 감소량 
            1 iteration epoch : 지정된 epoch 수
            1 iteration 이후 epoch : 지정된 epoch 수 / epoch_reduction
    * min_epoch(int) : (default : 0) : 최소 epoch

    ```python
    ## train corrector
    corrector.fit(model, x, y, epoch=100, batch_size=128, monitor='accuracy'
                        , use_best=True, mode='max', augmentation=process_image
                        ,careful_update=0.8, lr_scheduler=lrscheduler
                        ,data_sampler=None)
    ```
    * model(class) : (tensorflow 2.0.0 | tensorflow keras Model) 학습모형
    * x_data(array) : numpy array 
    * y_data(array) : numpy array(2 dim array) 최소 2개 이상의 category를 필요로 함
    * epoch(int) : (default : 3) 각 모델당 학습 에폭
    * batch_size(int) : (default : 64) 학습 베치사이즈
    * monitor(str) : (default : 'loss') 모니터링할 지표명
    * use_best(bool) : (default : True) 각 iteration내 model당 가장 좋은 성능의 가중치 사용
    * augmentation(func) : (default : None) 데이터 전처리 함수 ex) process_image = lambda x : np.flip(x, -1)
    * lr_scheduler(func) : (default : None) 현재 epoch을 받아 learning rate을 return 하는 함수 ex) lrscheduler = lambda e : 1e-3 if e >10 else 1-e4
    * careful_update(float) : (default : None) .0 ~ 1. 각 category별 최대 확률값이 지정된 값을 넘지 않으면 업데이트 하지 않음
    * data_sampler(class) : (default : None) under/over sampling class, imblearn>=0.6.1 support, "fit_resample" method를 가진 RandomUnderSampler, Pipeline, ... 등 만 지원합니다.

    ```python
    ## get new label
    new_label = corrector.new_label()
    ```
    """
    def __init__(self, n_checker, test_prob, logfile=None, save_probs=None, verbose=1, niter=10, epoch_reduction=0.5, min_epoch=0):
        self.n_checker = n_checker
        if type(self.n_checker) != int:
            raise ValueError(f"n_checker require int type object not {type(self.n_checker)}")

        self.test_prob = test_prob
        if (self.test_prob > 0.) and (self.test_prob < 0.5) == False:
            raise ValueError(f"test_prob should in 0 ~ 0.5 but {test_prob}")
        
        self.niter = niter
        if type(self.niter) != int:
            raise ValueError(f"niter require int type object not {type(niter)}")

        self.verbose = verbose
        self.epoch_reduction = epoch_reduction
        self.min_epoch = min_epoch 
        if type(self.min_epoch) != int:
            raise ValueError(f"min_epoch require int type object not {type(self.min_epoch)}")

        self._logger = logging_format(logfile, self.verbose)

        if save_probs is not None: ## save_probs 옵션이 None 일 경우 iteration 당 확률값을 저장하지 않음
            if os.path.isdir(save_probs) == False: ## save_probs에 지정된 경로가 존재하지 않을경우 에러
                raise ValueError(f"directory for save probabilities does not exist, save_probs: {save_probs}")
            else :
                self._save_probs = save_probs

        self.n_model = int(np.ceil(100/(100 * test_prob) * n_checker))
        
        
    def _get_chunk(self, n_data, shuffle = False):
        
        data_idx = list(range(n_data))
        data_size = len(data_idx)
        ## each size of chunk = test size / number of checker
        n_chunk = np.int(np.round(data_size* self.test_prob / self.n_checker))
        
        if shuffle:
            ## shuffling index
            random.shuffle(data_idx)
        
        chunk = []
        for i in range(np.int(np.round(data_size / n_chunk))):
            ## 데이터를 인덱스를 나누어 저장
            chunk.append(data_idx[i*n_chunk : (i+1)*n_chunk])
        
        return chunk
    
    
    def _train_test_gen(self, chunk):
        for i in range(len(chunk)):
            ## 나누어진 인덱스를 checker 개수만큼 가져와 test용 인덱스로 사용
            ## 나머지는 train용 인덱스
            test = chunk[i : (i + self.n_checker)]
            
            if len(test) < self.n_checker:
                ## 인덱스가 부족할 경우 앞에서 추가
                train = chunk[self.n_checker - len(test) : i]            
                test.extend(chunk[: self.n_checker - len(test)])

            else :
                train = chunk[:i] + chunk[(i + self.n_checker):]

            test_idx = []
            train_idx = []
            [test_idx.extend(t) for t in test]
            [train_idx.extend(t) for t in train]
            
            yield test_idx, train_idx
    
    
    def _get_trans_porb(self, label, nlabel):
        
        label = np.argmax(label, axis=-1)[:,np.newaxis]
        nlabel = np.argmax(nlabel, axis=-1)[:,np.newaxis]
        trans_porb = np.zeros((self.n_cat, self.n_cat))
        
        for c in range(self.n_cat):
            uniq, cnt = np.unique(nlabel[np.where(label==c)[0]]
                                  ,return_counts=True)
            for u, t in zip(uniq, cnt):
                trans_porb[c,u] = t/cnt.sum()
                
        return trans_porb


    def _compare(self, best, current, mode):
        
        if mode == 'max':
            return best <= current
        else :
            return best >= current
    
    
    def _resampler(self, data_sampler, x_train, y_train):
        start_sampling = time.time()
        y_train = np.argmax(y_train, axis =-1)
        label_cnt = Counter(y_train)
        
        x_train, y_train = data_sampler.fit_resample(x_train, y_train)
        
        sample_cnt = Counter(y_train)
        self._logger.info(f"(resample) {label_cnt} -> {sample_cnt}, samping time : {time.time()-start_sampling:0.3f} sec")
        
        return x_train, to_categorical(y_train)
    
    
    def _batch_generator(self, x_train, y_train, batch_size):
        length = x_train.shape[0]
        steps = int(length/batch_size)+1
        data_idx = list(range(length))

        random.shuffle(data_idx)
        for i in range(steps):
            idx = data_idx[i*batch_size : (i+1)*batch_size]        

            yield x_train[idx], y_train[idx]
    

    def _probabilistic_update(self, prior, joint_likelihood, trans_porb, iteration, careful_update):
        ## update Posterior
        if prior is None:
            # prior가 없음으로
            posterior = joint_likelihood
        else :
            # hidden markov model
            # t_prior = trans_porb * t-1_Prior
            # p(Ct|q) = p(Ct | Ct-1) * p(Ct-1 | q)
            prior = np.dot(trans_porb, np.transpose(prior))
            prior = np.transpose(prior)
            
            # posterior = t_prior * joint_likelihood
            # p(Ct+1|q) = p(Ct|q) * p(q|Ct+1)
            posterior = prior * joint_likelihood
        
        ### bayes rule update
        # [normalization] posterior = 1/Z * posterior
        tmp = np.sum(posterior, axis = -1)
        for i in range(posterior.shape[-1]):
            posterior[:,i] /= tmp
            
        # posterior become next step prior
        # p(Ct|q) <- p(Ct+1|q)
        prior = posterior.copy()
        
        if careful_update is not None: # (iteration == 0) and 
            unsure = np.max(posterior,axis = 1) < careful_update
            posterior[unsure] = self.t_label[unsure]
        
        return posterior, prior
    
    
    @tf.function
    def __train_on_batch(self, model, x, y):
        return model.train_on_batch(x, y)
    

    def __train_generator(self, train_idx, x_test, y_test, scaler, k, iteration
                         , epoch, batch_size, use_best, mode, augmentation, aug_poolsize, lr_scheduler):
        
        best_weights = self.model.get_weights()
        
        ## init best
        best_metric = 0 if mode == "max" else 1

        # 이전 학습상태를 가져와 이어서 학습함
        if iteration > 0:
            self.model.set_weights(self.save_checker[f"model_{k}"])
            first_epoch = epoch
            epoch = int(max(epoch*self.epoch_reduction, self.min_epoch))
        else:
            first_epoch = 0

        for e in range(epoch):
            ########## epoch start ##########
            start = time.time()
            loss, metric = [], []

            steps = int(len(train_idx)/batch_size)+1
            for i in range(steps):
                ########## setp start ##########
                step = train_idx[i*batch_size : (i+1)*batch_size].copy()
                step.sort() ## h5py 인덱싱은 오름차순 이어야함

                x_train = self.x_data[step].copy()
                if augmentation is not None:
                    with Pool(aug_poolsize) as p:
                        x_train = p.map(augmentation, x_train) ## pool.map은 결과를 list로 return함
                    x_train = np.concatenate([r[np.newaxis,:,:,:] for r in x_train]) ## list -> numpy array
                
                ## scaling
                x_train = x_train if scaler is None else scaler.transform(x_train)

                step_loss, *step_metric = self.__train_on_batch(self.model, x_train, self.y_data[step])
                loss.append(step_loss)
                metric.append(step_metric[self._ind])
                ########## setp end ##########

            if self.verbose >= 2:
                sys.stdout.write("\r {:2d}/{} ({:2d}/{})[{:3d}/{}] | train loss : {:2.5f}, train {} : {:2.5f}".format(
                    iteration+1, self.niter, k+1 , self.n_model, e+1, epoch, np.mean(loss), self.monitor, np.mean(metric))
                    )
            
            val_loss, *val_metric = self.__evaluate(x_test, y_test)
            val_metric = val_metric[self._ind]
            if self.verbose >= 2:
                print("| val loss : {:2.5f}, val {} : {:2.5f} | best {:2.5f} | time : {:0.3f} sec".format(
                    val_loss, self.monitor, val_metric, best_metric, time.time() - start)
                     )
            
            if use_best:
                if self._compare(best_metric, val_metric, mode):
                    ## mode가 max일 경우 best_metric =< val_metric 가 True, min일 경우 best_metric => val_metric 가 True
                    best_metric = val_metric
                    best_weights = self.model.get_weights()
            
            if lr_scheduler is not None:
                ## learning rate scheduler 가 있을 경우
                tf.keras.backend.set_value(self.model.optimizer.lr, lr_scheduler(e + first_epoch))
            
            self.model.reset_metrics()
        
        if use_best:
            ## use best model
            self.model.set_weights(best_weights)
        
        # 학습이 끝난후 모형의 학습상태를 저장하여 다음 iteration에 넘겨줌
        self.save_checker[f"model_{k}"] = self.model.get_weights()

        return np.mean(loss), np.mean(metric)
    

    def __train(self, x_train, y_train, x_test, y_test, k, iteration
                , epoch, batch_size, use_best, mode, augmentation, aug_poolsize, lr_scheduler):
        
        best_weights = self.model.get_weights()
        
        ## init best
        best_metric = 0 if mode == "max" else 1
        
        # 이전 학습상태를 가져와 이어서 학습함
        if iteration > 0:
            self.model.set_weights(self.save_checker[f"model_{k}"])
            first_epoch = epoch
            epoch = int(max(epoch*self.epoch_reduction, self.min_epoch))
        else:
            first_epoch = 0
        
        for e in range(epoch):
            ########## epoch start ##########
            start = time.time()
            loss, metric = [], []
            
            for x, y in self._batch_generator(x_train, y_train, batch_size):
                ########## setp start ##########
                step_loss, *step_metric = self.__train_on_batch(self.model, x, y)
                loss.append(step_loss)
                metric.append(step_metric[self._ind])
                ########## setp end ##########

            if self.verbose >= 2:
                sys.stdout.write("\r {:2d}/{} ({:2d}/{})[{:3d}/{}] | train loss : {:2.5f}, train {} : {:2.5f}".format(
                    iteration+1, self.niter, k+1 , self.n_model, e+1, epoch, np.mean(loss), self.monitor, np.mean(metric))
                    )
            
            
            val_loss, *val_metric = self.__evaluate(x_test, y_test)
            val_metric = val_metric[self._ind]
            if self.verbose >= 2:
                print("| val loss : {:2.5f}, val {} : {:2.5f} | best {:2.5f} | time : {:0.3f} sec".format(
                    val_loss, self.monitor, val_metric, best_metric, time.time() - start)
                     )
            
            if use_best:
                if self._compare(best_metric, val_metric, mode):
                    ## mode가 max일 경우 best_metric =< val_metric 가 True, min일 경우 best_metric => val_metric 가 True
                    best_metric = val_metric
                    best_weights = self.model.get_weights()
            
            if lr_scheduler is not None:
                tf.keras.backend.set_value(self.model.optimizer.lr, lr_scheduler(e+first_epoch))
            
            if augmentation is not None:
                with Pool(aug_poolsize) as p:
                    x_train = p.map(augmentation, x_train) ## pool.map은 결과를 list로 return함
                x_train = np.concatenate([r[np.newaxis,:,:,:] for r in x_train]) ## list -> numpy array
            
            self.model.reset_metrics()
            
        if use_best:
            ## use best model
            self.model.set_weights(best_weights)
        
        # 학습이 끝난후 모형의 학습상태를 저장하여 다음 iteration에 넘겨줌
        self.save_checker[f"model_{k}"] = self.model.get_weights()
        
        return np.mean(loss), np.mean(metric)
    
    
    def __evaluate(self, x_test, y_test):
        return self.model.evaluate(x_test, y_test, batch_size=32, verbose = 0)
    
    
    def __predict(self, x_test):
        return self.model.predict(x_test, batch_size=32, verbose=0)
    
    
    def fit(self, model, data, scaler=None, epoch=3, batch_size=128, use_best=True, monitor='loss'
           , mode='max', augmentation=None, aug_poolsize=1, lr_scheduler=None, careful_update=None
           , data_sampler=None):
        
        self.model = model
        
        metrics = [m.get_config()['name'] for m in self.model.metrics]
        if monitor not in metrics:
            print(f"[{monitor}] not in metrics, available metrics is {metrics}, use first")
            monitor = metrics[0]
        self._ind = metrics.index(monitor)
        self.monitor = monitor
        
        if isinstance(data, h5py.File):
            ## h5py 일때
            self.is_generator = True
            self.x_data, self.y_data = data["x"], data["y"][:]
            
        else :
            ## h5py 아닐때
            self.is_generator = False
            self.x_data, self.y_data = data
            if isinstance(self.x_data, np.ndarray) == False:
                raise ValueError(f"x_data require numpy array not {type(self.x_data)}")
            
            if isinstance(self.y_data, np.ndarray) == False:
                raise ValueError(f"y_data require numpy array not {type(self.y_data)}")

        if len(self.y_data.shape) != 2 :
            raise ValueError(f"y_data require 2 dim array but got {self.y_data.shape}")

        self.n_cat = self.y_data.shape[-1]
        if self.n_cat < 2:
            raise ValueError(f"y_data need at least 2 category value but got {self.n_cat}") 
        
        if (isinstance(careful_update, float) == False) and (careful_update is not None):
            raise ValueError(f"careful_update should in float or None not {careful_update}")
        
        if (data_sampler is not None) & ("fit_resample" not in dir(data_sampler)):
            raise ValueError(f"data_sampler has no fit_resample method ")

        if mode not in ['max', 'min']:
            raise ValueError(f"available mode [max, min] not {mode}")
        
        self.t_label = self.y_data.copy()
        self.n_data = self.x_data.shape[0]
        # 각 checker 의 학습내용을 다음 iteration으로 전달하기 위해 보관
        self.save_checker = {}
        prior = None
        trans_porb = None
        
        # 데이터의 인덱스를 [data_size* test 비율 / n_checker] 만큼 분리해둠
        chunk = self._get_chunk(self.n_data, shuffle = True)
        
        for iteration in range(self.niter):
            ########## iteration start ##########
            self._logger.info(f"============ iteration : {iteration+1} / {self.niter} ============")
            
            ## 확률(likelihood) [Posterior likelihood] (shape : n data * n checker * n category)
            joint_likelihood = np.ones((self.n_data, self.n_model, self.n_cat ))
            
            for k, (test_idx, train_idx) in enumerate(self._train_test_gen(chunk)):
                ########## k start ##########
                initial_weights = self.model.get_weights()
                
                ## train test split
                if self.is_generator:
                    ## test data는 memory에 로딩
                    test_idx.sort() ## h5py 인덱싱은 오름차순 이어야함
                    x_test, y_test = self.x_data[test_idx], self.t_label[test_idx]

                    train_loss, train_metric = self.__train_generator(train_idx, x_test, y_test, scaler, k, iteration, epoch, batch_size,
                                                        use_best, mode, augmentation, aug_poolsize, lr_scheduler)
                else : 
                    x_train, y_train = self.x_data[train_idx], self.t_label[train_idx]
                    x_test, y_test = self.x_data[test_idx], self.t_label[test_idx]
                
                    if data_sampler is not None:
                        x_train, y_train = self._resampler(data_sampler, x_train, y_train)
                    ## train
                    train_loss, train_metric = self.__train(x_train, y_train, x_test, y_test, k, iteration, epoch, batch_size, 
                                                        use_best, mode, augmentation, aug_poolsize, lr_scheduler)
                
                ## validate
                val_loss, *val_metric = self.__evaluate(x_test, y_test)
                self._logger.info('model : {:3d}/{}, train_loss : {:2.5f}, train {} : {:2.5f}, val loss : {:2.5f}, val {} : {:2.5f}'.format(
                                    k+1, self.n_model, train_loss, self.monitor, train_metric, val_loss, self.monitor, val_metric[self._ind]))
                ## predict
                y_pred = self.__predict(x_test)
                joint_likelihood[test_idx, k, :] = y_pred
                
                ## reset model
                self.model.set_weights(initial_weights)
                ########## k end ##########
            
            ## t-1 label <- t label
            self.t1_label = self.t_label.copy()
            
            ## product checkers result = p(q|Ct+1) 
            np.save(os.path.join("prob", f"joint_likelihood_{iteration+1}"), joint_likelihood) if self._save_probs is not None else None
            joint_likelihood = np.prod(joint_likelihood, axis = 1)
            
            ## probabilistic update (bayes rule, markov chain)
            posterior, prior = self._probabilistic_update(prior, joint_likelihood, trans_porb, iteration, careful_update)
            
            ## new label
            posterior = np.argmax(posterior, axis =-1)
            self._logger.info(f"N modifyed label iteration : { np.sum( np.argmax(self.t1_label, axis =-1) != posterior ) } / {len(posterior)}" )
            self._logger.info(f"N modifyed label total : { np.sum( np.argmax(self.y_data, axis =-1) != posterior ) } / {len(posterior)}" )
            
            ## update label
            posterior = to_categorical(posterior)
            self.t_label = np.zeros((self.n_data, self.n_cat ))
            self.t_label[:,:posterior.shape[1]] = posterior
            
            # 전이확률 계산
            trans_porb = self._get_trans_porb(self.t_label, self.t1_label)
            np.save(os.path.join('prob', f"trans_porb_{iteration+1}"), trans_porb) if self._save_probs is not None else None
            np.save(os.path.join('prob', f"t_label_{iteration+1}"), self.t_label) if self._save_probs is not None else None
            
            ########## iteration end ##########
        ## end        
        return None

    
    def new_label(self):
        return self.t_label
