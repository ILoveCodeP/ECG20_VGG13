#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import os
from pyts import datasets
import matplotlib.pyplot as plt
from scipy.stats import levy_stable
from scipy.stats import levy_stable
import tensorflow as tf
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
config = tf.compat.v1.ConfigProto()
config.allow_soft_placement=True
config.gpu_options.per_process_gpu_memory_fraction=0.7
config.gpu_options.allow_growth = True


# In[3]:


from pyts import datasets
(data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset("ECG200",return_X_y=True)


# In[5]:


noise_scale = np.sqrt(0.5)*0.03 #根据github代码决定

def stable_noise(img, alpha, beta, scale):
    '''
    此函数用将产生的stable噪声加到图片上
    传入:
        img   :  原图
        alpha  :  shape parameter
        beta :  symmetric parameter
        scale : scale parameter
        random_state : 随机数种子
    返回:
        stable_out : 噪声处理后的图片
        noise        : 对应的噪声
    '''
    # 产生stable noise
    noise = levy_stable.rvs(alpha=alpha, beta=beta, scale=scale, size=img.shape)
    # 将噪声和图片叠加
    stable_out = img + noise
    # 将超过 1 的置 1，低于 0 的置 0
    # stable_out = np.clip(stable_out, 0, 255)
    # 取整
    # stable_out = np.uint(stable_out)
    return stable_out, noise  # 这里也会返回噪声，注意返回值

def stable_noise_row(row, alpha, beta=0, scale=noise_scale):  # scale修改为0.03
    #         random_state = row.name #第0张图的随机数种子就是0，第1张图的随机数种子就是1，以此类推。。。
    return stable_noise(np.asarray(row), alpha, beta, scale)[0]


# In[6]:


plt.plot(data_train[1])


# In[7]:


# In[11]:




# In[14]:


from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Flatten, Dropout, Input
from tensorflow.keras.layers import MaxPooling1D, Conv1D
from tensorflow.keras.layers import LSTM, Bidirectional
from tensorflow.keras.layers import BatchNormalization, GlobalAveragePooling1D, Permute, concatenate, Activation, add
import numpy as np
import math

def get_model(model_name, input_shape, nb_class):
    if model_name == "vgg":
        model = cnn_vgg(input_shape, nb_class)
    elif model_name == "lstm1":
        model = lstm1(input_shape, nb_class)
    elif model_name == "lstm":
        model = lstm1v0(input_shape, nb_class)
    elif model_name == "lstm2":
        model = lstm2(input_shape, nb_class)
    elif model_name == "blstm1":
        model = blstm1(input_shape, nb_class)
    elif model_name == "blstm2":
        model = blstm2(input_shape, nb_class)
    elif model_name == "lstmfcn":
        model = lstm_fcn(input_shape, nb_class)
    elif model_name == "resnet":
        model = cnn_resnet(input_shape, nb_class)
    elif model_name == "mlp":
        model = mlp4(input_shape, nb_class)
    elif model_name == "lenet":
        model = cnn_lenet(input_shape, nb_class)
    else:
        print("model name missing")
    return model


def mlp4(input_shape, nb_class):
    # Z. Wang, W. Yan, T. Oates, "Time Series Classification from Scratch with Deep Neural Networks: A Strong Baseline," Int. Joint Conf. Neural Networks, 2017, pp. 1578-1585
    
    ip = Input(shape=input_shape)
    fc = Flatten()(ip)
    
    # fc = Dropout(0.1)(fc)
            
    fc = Dense(500, activation='relu')(fc)
    # fc = Dropout(0.2)(fc)
    
    fc = Dense(500, activation='relu')(fc)
    # fc = Dropout(0.2)(fc)
    
    fc = Dense(500, activation='relu')(fc)
    # fc = Dropout(0.3)(fc)
    
    out = Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model

def cnn_lenet(input_shape, nb_class):
    # Y. Lecun, L. Bottou, Y. Bengio, and P. Haffner, “Gradient-based learning applied to document recognition,” Proceedings of the IEEE, vol. 86, no. 11, pp. 2278–2324, 1998.
    
    ip = Input(shape=input_shape)
    
    conv = ip
    
    nb_cnn = int(round(math.log(input_shape[0], 2))-3)
    print("pooling layers: %d"%nb_cnn)
    
    for i in range(nb_cnn):
        conv = Conv1D(6+10*i, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        
    flat = Flatten()(conv)
    
    fc = Dense(120, activation='relu')(flat)
    # fc = Dropout(0.5)(fc)
    
    fc = Dense(84, activation='relu')(fc)
    # fc = Dropout(0.5)(fc)
    
    out = Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model


def cnn_vgg(input_shape, nb_class ,num_filter):
    # K. Simonyan and A. Zisserman, "Very deep convolutional networks for large-scale image recognition," arXiv preprint arXiv:1409.1556, 2014.
    
    ip = Input(shape=input_shape)
    
    conv = ip
    
    # nb_cnn = int(round(math.log(input_shape[0], 2))-3)
    nb_cnn = 5
    print("pooling layers: %d"%nb_cnn)
    
    for i in range(nb_cnn):
        print('i is {}'.format(i))
        num_filters = min(64*2**i, 512)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        if i > 1:
            conv = Conv1D(num_filters, 3, padding='same', activation="relu", kernel_initializer='he_uniform')(conv)
        conv = MaxPooling1D(pool_size=2)(conv)
        
    flat = Flatten()(conv)
    
    fc = Dense(4096, activation='relu')(flat)
    # fc = Dropout(0.5)(fc)
    
    fc = Dense(4096, activation='relu')(fc)
    # fc = Dropout(0.5)(fc)
    
    out = Dense(nb_class, activation='softmax')(fc)
    
    model = Model([ip], [out])
    model.summary()
    return model

def lstm1v0(input_shape, nb_class):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
        
    ip = Input(shape=input_shape)

    l2 = LSTM(512)(ip)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def lstm1(input_shape, nb_class):
    # Original proposal:
    # S. Hochreiter and J. Schmidhuber, “Long Short-Term Memory,” Neural Computation, vol. 9, no. 8, pp. 1735–1780, Nov. 1997.
    
    # Hyperparameter choices: 
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    
    ip = Input(shape=input_shape)

    l2 = LSTM(100)(ip)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def lstm2(input_shape, nb_class):
    ip = Input(shape=input_shape)

    l1 = LSTM(100, return_sequences=True)(ip)
    l2 = LSTM(100)(l1)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model


def blstm1(input_shape, nb_class):
    # Original proposal:
    # M. Schuster and K. K. Paliwal, “Bidirectional recurrent neural networks,” IEEE Transactions on Signal Processing, vol. 45, no. 11, pp. 2673–2681, 1997.
    
    # Hyperparameter choices: 
    # N. Reimers and I. Gurevych, "Optimal hyperparameters for deep lstm-networks for sequence labeling tasks," arXiv, preprint arXiv:1707.06799, 2017
    ip = Input(shape=input_shape)

    l2 = Bidirectional(LSTM(100))(ip)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def blstm2(input_shape, nb_class):
    ip = Input(shape=input_shape)

    l1 = Bidirectional(LSTM(100, return_sequences=True))(ip)
    l2 = Bidirectional(LSTM(100))(l1)
    out = Dense(nb_class, activation='softmax')(l2)

    model = Model([ip], [out])

    model.summary()

    return model

def lstm_fcn(input_shape, nb_class):
    # F. Karim, S. Majumdar, H. Darabi, and S. Chen, “LSTM Fully Convolutional Networks for Time Series Classification,” IEEE Access, vol. 6, pp. 1662–1669, 2018.

    ip = Input(shape=input_shape)
    
    # lstm part is a 1 time step multivariate as described in Karim et al. Seems strange, but works I guess.
    lstm = Permute((2, 1))(ip)

    lstm = LSTM(128)(lstm)
    # lstm = Dropout(0.8)(lstm)

    conv = Conv1D(128, 8, padding='same', kernel_initializer='he_uniform')(ip)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(256, 5, padding='same', kernel_initializer='he_uniform')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    conv = Conv1D(128, 3, padding='same', kernel_initializer='he_uniform')(conv)
    conv = BatchNormalization()(conv)
    conv = Activation('relu')(conv)

    flat = GlobalAveragePooling1D()(conv)

    flat = concatenate([lstm, flat])

    out = Dense(nb_class, activation='softmax')(flat)

    model = Model([ip], [out])

    model.summary()

    return model


def cnn_resnet(input_shape, nb_class):
    # I. Fawaz, G. Forestier, J. Weber, L. Idoumghar, P-A Muller, "Data augmentation using synthetic data for time series classification with deep residual networks," International Workshop on Advanced Analytics and Learning on Temporal Data ECML/PKDD, 2018

    ip = Input(shape=input_shape)
    residual = ip
    conv = ip
    
    for i, nb_nodes in enumerate([64, 128, 128]):
        conv = Conv1D(nb_nodes, 8, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv1D(nb_nodes, 5, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        conv = Conv1D(nb_nodes, 3, padding='same', kernel_initializer="glorot_uniform")(conv)
        conv = BatchNormalization()(conv)
        conv = Activation('relu')(conv)

        if i < 2:
            # expands dimensions according to Fawaz et al.
            residual = Conv1D(nb_nodes, 1, padding='same', kernel_initializer="glorot_uniform")(residual)
        residual = BatchNormalization()(residual)
        conv = add([residual, conv])
        conv = Activation('relu')(conv)
        
        residual = conv
    
    flat = GlobalAveragePooling1D()(conv)

    out = Dense(nb_class, activation='softmax')(flat)

    model = Model([ip], [out])

    model.summary()

    return model


# In[18]:


from sklearn.preprocessing import LabelEncoder
from keras.utils import np_utils
from sklearn import metrics
from tensorflow.keras.optimizers import Adam
from keras.callbacks import EarlyStopping
model_type = 'vgg'
train_list = [0,2,1.9,1.5,1.3,1,0.9,6]
alpha_trains = [2,1.9,1.5,1.3,1,0.9]
times_tmp = 2 #每种噪声加几倍
nb_iterations = 10000
batch_size = 256
num_cnn = 5
for alpha_train in train_list:

    acc_temp = []
    auc_temp = []
    alpha_test_temp = []
    for repeat_time in range(5):

        (data_train, data_test, target_train, target_test)=datasets.fetch_ucr_dataset("ECG200",return_X_y=True)

        times = 10

        X_train = data_train.copy()
        epoch_num = np.ceil(nb_iterations * (batch_size / X_train.shape[0])).astype(int)

        X_train_max = np.max(X_train)
        X_train_min = np.min(X_train)
        X_train = 2. * (X_train - X_train_min) / (X_train_max - X_train_min) - 1.

        X_test_f = data_test
        X_test_max = np.max(X_test_f)
        X_test_min = np.min(X_test_f)
        X_test_f = 2. * (X_test_f - X_test_min) / (X_test_max - X_test_min) - 1.
        
        samples_train,features = np.shape(X_train)
        samples_test = np.shape(X_test_f)[0]
        noisy = np.tile(X_train, (times,1))
        y_tr = np.tile(target_train,(1,times+1))
        y_tr = y_tr.reshape(samples_train*(times+1),1)
        num_classes = 2
        encoder = LabelEncoder()

        from sklearn.metrics import mutual_info_score
        if alpha_train == 0:

            X_train = X_train
            y_tr = target_train.copy()
            y_train = encoder.fit_transform(y_tr)
            y_train = np_utils.to_categorical(y_train, num_classes)
            y_train = np.array(y_train)
        else:
            if alpha_train == 6:
                num = len(alpha_trains)
                noisy_ = np.tile(X_train, (times_tmp,1))#200*96
                for alphas in alpha_trains:
                    temp = noisy_.copy() #100*times_tmp, 96
                    noise = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alphas, scale=noise_scale), axis=1, arr=temp)

                    X_train = np.r_[X_train,noise]
                    print(np.shape(X_train))
                y_tra = np.tile(target_train,(1,times_tmp*num+1))
                y_train = y_tra.reshape(-1)
                y_train = encoder.fit_transform(y_train)
                y_train = np_utils.to_categorical(y_train, num_classes)
                y_train = np.array(y_train)
                
            else:
                X_noise = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alpha_train, scale=noise_scale),1,arr=noisy)
                # X_noise = stable_noise(noisy, alpha=alpha_train, beta=0, scale=noise_scale)[0]
                print(np.shape(X_train))
                print(np.shape(X_noise))
                X_train = np.r_[X_train,X_noise]

                y_train = encoder.fit_transform(y_tr)
                y_train = np_utils.to_categorical(y_train, num_classes)
                y_train = np.array(y_train)
                print("the third one {}".format(np.shape(X_train)))
                print("the forth one {}".format(np.shape(y_train)))

        input_shape = np.shape(X_train[0].reshape(features,1))
        # model = get_model(model_type, input_shape, num_classes)
        model = cnn_vgg(input_shape, num_classes ,num_cnn)
        es = EarlyStopping(monitor='loss', mode='min', verbose=1, patience=40, min_delta=0.01)
        model.compile(loss='categorical_crossentropy',
                      optimizer=Adam(),
                      metrics=['accuracy'])

        model.summary()
        model.fit(X_train, y_train, batch_size=batch_size, epochs=epoch_num, verbose=1,callbacks=[es])
        import gc
        del X_train
        gc.collect()
## 测试集构造
        noisy_test = np.tile(X_test_f, (times,1))

## 进行测试
        alpha_tests = [6,0,2,1.9,1.5,1.3,1,0.9]
        for alpha_test in alpha_tests:
            X_test = X_test_f.copy()
            y_test = target_test.copy()
            if alpha_test == 0:
                ########ONLY REPEAT ONCE FOR CLEAN DATA TEST###########
                y_te = y_test.copy()
                y_test = encoder.fit_transform(y_te)
                y_test = np_utils.to_categorical(y_test, num_classes)
                y_test = np.array(y_test)
                y_test = y_test.reshape(samples_test,num_classes)
                ########################################################
                X_test = X_test
            else:
                if alpha_test == 6:
                    noisy_t = np.tile(X_test, (times_tmp,1))#200*96
                    for alphas in alpha_trains:
                        temp = noisy_t.copy() #100*times_tmp, 96
                        noise_t = np.apply_along_axis(lambda x: stable_noise_row(x, alpha = alphas, scale=noise_scale), axis=1, arr=temp)

                        X_test = np.r_[X_test,noise_t]
                        print(np.shape(X_test))


                    num = len(alpha_trains)
                    y_tes = np.tile(target_test,(1,times_tmp*num+1))
                    y_test = y_tes.reshape(-1)
                    y_test = encoder.fit_transform(y_test)
                    y_test = np_utils.to_categorical(y_test, num_classes)
                    y_test = np.array(y_test)
                else:
                    y_te = np.tile(target_test,(1,times+1))
                    y_te = y_te.reshape(samples_test*(times+1),1)
                    y_test = encoder.fit_transform(y_te)
                    y_test = np_utils.to_categorical(y_test, num_classes)
                    y_test = np.array(y_test)
                    y_test = y_test.reshape(samples_test*(times+1),num_classes)

                    X_noise_test = np.apply_along_axis(lambda x: stable_noise_row(x, alpha=alpha_test, scale=noise_scale),
                                                axis=1, arr=noisy_test)
                    X_test = np.r_[X_test,X_noise_test]
        
            
            y_pred = model.predict(X_test)
            tt1=np.argmax(y_test, axis=1)
            auc_value = metrics.roc_auc_score(y_test,y_pred,multi_class='ovo',average='macro')


            tt2=np.argmax(y_pred, axis=1)
            acc = metrics.accuracy_score(tt1, tt2)

            acc_temp.append(acc)
            auc_temp.append(auc_value)
            alpha_test_temp.append(alpha_test)
        
        del model
        gc.collect()

    np.savetxt('accuracy_alpha{}_model{}_num{}.txt'.format(alpha_train, model_type,num_cnn), acc_temp)
    np.savetxt('auc_alpha{}_model{}_num{}.txt'.format(alpha_train, model_type,num_cnn), auc_temp)
    np.savetxt('alpha_test_alpha{}_model{}_num{}.txt'.format(alpha_train, model_type,num_cnn), alpha_test_temp)

