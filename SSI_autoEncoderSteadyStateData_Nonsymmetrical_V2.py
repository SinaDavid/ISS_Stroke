# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:05:18 2021

Some info about the data set, see also e-mail Sina David 30-08-2021

‘LASI' (0-2)
'RASI'(3-5)
'LPSI'(6-8)
'RPSI'(9-11)
'LANK'(12-14)
'RANK' (15-17)
'LHEE'(18-20)
'RHEE'(21-23

Version 0.1 
- creating a encoder / decoder neural network to analyse steady state data. 





- Questions i come across, do we time normalize the gait cycles. (must be)


@author: michi
"""



# 1: Loading required libararies.
import tensorflow as tf
from numpy import mean
from numpy import std
from numpy import dstack
from pandas import read_csv
from matplotlib import pyplot
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Dropout
from keras.layers.convolutional import Conv1D
from keras.layers.convolutional import MaxPooling1D
from keras.utils import to_categorical
from os.path import dirname, join as pjoin
import scipy.io as sio
import os
import scipy.io as spio
import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import MaxPool2D, AvgPool2D, BatchNormalization, Dropout, Input,Activation
from keras import layers
from keras.layers import Activation
from sklearn import preprocessing
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import itertools
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
from keras import regularizers
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,LeaveOneGroupOut,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import tensorflow





def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tensorflow.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tensorflow.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch*28*28
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tensorflow.square(distribution_mean) - tensorflow.exp(distribution_variance)
        kl_loss_batch = tensorflow.reduce_mean(kl_loss)
        return kl_loss_batch*(-0.5)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        return reconstruction_loss_batch + kl_loss_batch
    
    return total_loss

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tensorflow.shape(distribution_variance)[0]
    random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
    return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random


################# Settings over here! :-) #####################
plt.close('all')
inputTimeSerie = 'markerpos'
path = 'F:\\SSI_data\\steadystate_prefSpeed\\steadystate_prefSpeed\\5GC\\markerpos\\'
inputColumns = [18,19,20,21,22,23]



# 2 Loading the data:  

filename = []
data = []
group = []
numberPer = 0
perturbationType = []

if 'path' in locals():
    for file1 in os.listdir(path):
        data.append(spio.loadmat(path + file1))
        filename.append(file1)
        group.append(int(file1[1:4]))
    #     perturbationType.append(numberPer)
    # numberPer +=1

# get the dict out of the list
for numtrials in range(len(filename)):
    data[numtrials] = (data[numtrials]['markervel'])


# if inputTimeSerie == 'markervel':
    X_data = np.zeros(len(inputColumns))#[0,0,0]
    for indx in range(len(data)):
        x = data[indx][0:1000,inputColumns]   # 
        X_data = np.vstack((X_data,x))          
       

# ### Always remove first row (due to initialize issues)       
X_data = np.delete(X_data, (0), axis=0) 
# To reduce data we resample with a factor 5. --> TBD
X_data = X_data[::5]
minimum = np.min(X_data)
maximum = np.max(X_data)
range1 = minimum - maximum
X_data /= range1
# ### Normalize / scale input  && add perturbationSeries to the input ###

# X_data1 = scale(X_data)
# X_data /=2
# X_scaled = scale(X_data, -1, 1)
# # if numberPer > 1:
# #     X_data = np.hstack((X_data, perturbationsSeries))
# # if numberPer == 1:
# #     numberPer = 0    







##### Get the dependent y data from filenames. #######
y = [0]
for indx in range(len(filename)):
    x = filename[indx].split('FR')
    y = np.vstack((y,int(x[1][0])))    
y = np.delete(y,(0),axis=0)


########    Reshape  the X data ############
X_data = X_data.reshape(len(y),200,len(inputColumns))



########### Split in train and test set ################
train_data = X_data[np.arange(start=2,stop=np.shape(X_data)[0],step=2)]
test_data = X_data[np.arange(start=1,stop=np.shape(X_data)[0],step=2)]




##############################################################################
############################### Encoder / decoder part #######################
##############################################################################


tensorflow.compat.v1.disable_eager_execution()

input_data = tensorflow.keras.layers.Input(shape=(200, 6))
# input_data = tensorflow.keras.layers.Input(epochLength, 6)

encoder = tensorflow.keras.layers.Conv1D(64, 5,activation='relu')(input_data)
# encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
encoder = tensorflow.keras.layers.MaxPooling1D(2)(encoder)

encoder = tensorflow.keras.layers.Conv1D(64, 3,activation='relu')(encoder)
# encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
encoder = tensorflow.keras.layers.MaxPooling1D(2)(encoder)

encoder = tensorflow.keras.layers.Conv1D(32, 3, activation='relu')(encoder)
# encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
encoder = tensorflow.keras.layers.MaxPooling1D(2)(encoder)

encoder = tensorflow.keras.layers.Flatten()(encoder)
encoder = tensorflow.keras.layers.Dense(16)(encoder)

encoder = tensorflow.keras.layers.Dense(8)(encoder)

encoder = tensorflow.keras.layers.Dense(6)(encoder)
encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
distribution_mean = tensorflow.keras.layers.Dense(6, name='mean')(encoder)
distribution_variance = tensorflow.keras.layers.Dense(6, name='log_variance')(encoder)
latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])



encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
encoder_model.summary()

################### DECODER PART ############
decoder_input = tensorflow.keras.layers.Input(shape=(6)) # probably change the (6) to (2)!
decoder = tensorflow.keras.layers.Dense(64)(decoder_input)
decoder = tensorflow.keras.layers.Reshape((1, 64))(decoder)
decoder = tensorflow.keras.layers.Conv1DTranspose(16, 3, activation='relu')(decoder)

decoder = tensorflow.keras.layers.Conv1DTranspose(32, 5, activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling1D(5)(decoder)

decoder = tensorflow.keras.layers.Conv1DTranspose(64, 5, activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling1D(5)(decoder)


decoder_output = tensorflow.keras.layers.Conv1DTranspose(6, 6)(decoder)
decoder_output = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(decoder_output)



decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
print("\ndecoder summary")
decoder_model.summary()

encoded = encoder_model(input_data)
decoded = decoder_model(encoded)


autoencoder = tensorflow.keras.models.Model(input_data, decoded)


autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
print("\nautoenoder summary")
autoencoder.summary()


# history = autoencoder.fit(train_data, train_data, epochs=200, batch_size=64, validation_data=(test_data, test_data))
history = autoencoder.fit(train_data,
                          train_data,
                          epochs=200,
                          batch_size=64,
                          callbacks=[EarlyStopping(monitor='loss', patience=5)],
                          validation_data=(test_data, test_data))

plt.plot(history.history['val_loss'])


# Next visualize the decoded data against original data.
fig2, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(test_data[1])
# test = np.expand_dims(test_data[1],axis=(2,1))
test = np.expand_dims(test_data[1], axis=0)

testPredicted = autoencoder.predict(test)
testPredicted = testPredicted[0,:,:]
ax2.plot(testPredicted)





