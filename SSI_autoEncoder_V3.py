# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:55 2021



To Do list:
    - push to github
    - create scatter plot --> done
    - create multiple subplots.
    - optimize model by implementing leakyrelu on large scale. --> not sure if the model further improves.  
    - add type of perturbations. --> increase bottleneck towards 3 vars. 
        - create 3d visual and model 
    - color cotting fallers and non-fallers
    - plot the validation loss function to explore which architecture describes the data the best. 







Questions to ask:



@author: michi
"""

# cnn model

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
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import itertools
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
from keras import regularizers
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,LeaveOneGroupOut,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)



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



## Settings


seconds = 2
expectedSampleRate = 100
epochLength = seconds*expectedSampleRate
inputTimeSerie = 'markervel'
inputColumns = [3,4,5,42,43,44]
n_features = len(inputColumns)
#inputTimeSerie = 'markerpos'
############# Loading the data ###############################

filename = []
data = []
#dataDir = 'D:\\Stroke_data\\V2_Data\\10sec\\ML_markervel\\ParYpsi\\'
# dataDir1 = 'F:\\SSI_data\\10sec\\ML_markervel\\NonParCont\\' #--> 0.55
# dataDir2 = 'F:\\SSI_data\\10sec\\ML_markervel\\ParCont\\'  #--> 0.39
# dataDir3 = 'F:\\SSI_data\\10sec\\ML_markervel\\NonParYpsi\\' #--> 0.53
dataDir4 = 'F:\\SSI_data\\10sec\\ML_markervel\\ParYpsi\\' #--> 0.52
## Adding first data perturbation. ###
#dataDir3 =  'F:\\SSI_data\\test\\'

#dataDir4 = 'F:\\SSI_data\\V3_Data\\V3_Data\\10sec\\ML_markervel\\ParYpsi\\'
#dataDir1 = 'F:\\SSI_data\\Vicon\\'


group = []
numberPer = 0
perturbationType = []
if 'dataDir1' in locals():
    for file1 in os.listdir(dataDir1):
        data.append(spio.loadmat(dataDir1 + file1))
        filename.append(file1)
        group.append(int(file1[1:4]))
        perturbationType.append(numberPer)
    numberPer +=1
if 'dataDir2' in locals():        
    for file2 in os.listdir(dataDir2):
        data.append(spio.loadmat(dataDir2 + file2))
        filename.append(file2)
        group.append(int(file2[1:4]))
        perturbationType.append(numberPer)
    numberPer +=1
if 'dataDir3' in locals():         
    for file3 in os.listdir(dataDir3):
        data.append(spio.loadmat(dataDir3 + file3))
        filename.append(file3)
        group.append(int(file3[1:4]))
        perturbationType.append(numberPer)
    numberPer +=1
if 'dataDir4' in locals():     
    for file4 in os.listdir(dataDir4):
        data.append(spio.loadmat(dataDir4 + file4))
        filename.append(file4)
        group.append(int(file4[1:4]))
        perturbationType.append(numberPer)
    numberPer +=1
    
## convert perturbationtype to length of perturbation signal
perturbationsSeries = [] #np.zeros(epochLength)
for indx in range(len(perturbationType)):
    pert = np.ones(epochLength)*perturbationType[indx]
    perturbationsSeries = np.hstack((perturbationsSeries,pert))

perturbationsSeries = to_categorical(perturbationsSeries)

group = np.array(group)



# get the dict out of the list
for numtrials in range(len(filename)):
    data[numtrials] = (data[numtrials][inputTimeSerie])

##### For different inputs, different handling of the data. 
#### Stack the data of multiple trials on top of each other and perform normalisation.

if inputTimeSerie == 'markerpos': 
    tempData = np.zeros(epochLength)
    tempData1 = []
    X_data = [0,0,0] #np.zeros((1001*177,3))
    Norx = [0,0,0]
    for indx in range(len(data)):
        for j in range(0,3):
            x = data[indx][0:epochLength,j+42]
            test = (x-min(x))/(max(x)-min(x))
            tempData = np.vstack((tempData,test))
        tempData = np.delete(tempData, (0), axis=0)  
        tempData = np.transpose(tempData)
        X_data =  np.vstack((X_data,tempData))
        tempData = np.zeros(epochLength)
    
       


if inputTimeSerie == 'markervel':
   X_data = np.zeros(len(inputColumns))#[0,0,0]
   for indx in range(len(data)):
       x = data[indx][0:epochLength,inputColumns]
       X_data = np.vstack((X_data,x))
       

### Always remove first row (due to initialize issues)       
X_data = np.delete(X_data, (0), axis=0) 
### Normalize / scale input  && add perturbationSeries to the input ###
X_data /=2
# if numberPer > 1:
#     X_data = np.hstack((X_data, perturbationsSeries))
# if numberPer == 1:
#     numberPer = 0    



##### Get the dependent y data from filenames. #######
y = [0]
for indx in range(len(filename)):
    x = filename[indx].split('FR')
    y = np.vstack((y,int(x[1][0])))    
y = np.delete(y,(0),axis=0)

###### Make y a categorical value #####    
y_cat = to_categorical(y)




########    Reshape  the X data ############
X_data = X_data.reshape(len(y_cat),epochLength,n_features)# we are not going to add the perturbations into into for auteencodrrs+numberPer 



train_data = X_data[np.arange(start=2,stop=np.shape(X_data)[0],step=2)]
test_data = X_data[np.arange(start=1,stop=np.shape(X_data)[0],step=2)]


####### Split data by group (subject level) #################
#group_kfold = GroupKFold(n_splits=2)
#group_kfold.get_n_splits(X_data, y_cat, group)
#gss = GroupShuffleSplit(n_splits=2, train_size=.7)#, random_state=42
#
#for train_index, test_index in gss.split(X_data, y_cat, group):
##    print("TRAIN:", train_index, "TEST:", test_index)
#    X_train, X_test = X_data[train_index], X_data[test_index]
#    y_train, y_test = y_cat[train_index], y_cat[test_index]
#    print(X_train, X_test, y_train, y_test)

####### Split data by trial (within subject level) #################
#X_train, X_test, y_train, y_test = train_test_split(X_data, y_cat, test_size=0.3, random_state=42)
###################################################################################
epochLength = 200

import tensorflow
tensorflow.compat.v1.disable_eager_execution()



input_data = tensorflow.keras.layers.Input(shape=(epochLength, 6))
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

encoder = tensorflow.keras.layers.Dense(2)(encoder)

distribution_mean = tensorflow.keras.layers.Dense(2, name='mean')(encoder)
distribution_variance = tensorflow.keras.layers.Dense(2, name='log_variance')(encoder)
latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])



encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
encoder_model.summary()

################### DECODER PART ############

decoder_input = tensorflow.keras.layers.Input(shape=(2))
decoder = tensorflow.keras.layers.Dense(64)(decoder_input)
decoder = tensorflow.keras.layers.Reshape((1, 64))(decoder)
decoder = tensorflow.keras.layers.Conv1DTranspose(64, 3, activation='relu')(decoder)

decoder = tensorflow.keras.layers.Conv1DTranspose(32, 5, activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling1D(5)(decoder)

decoder = tensorflow.keras.layers.Conv1DTranspose(16, 5, activation='relu')(decoder)
decoder = tensorflow.keras.layers.UpSampling1D(5)(decoder)

# decoder_output = tensorflow.keras.layers.Conv1DTranspose(6, 6, activation='relu')(decoder)

decoder_output = tensorflow.keras.layers.Conv1DTranspose(6, 6)(decoder)
decoder_output = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(decoder_output)




decoder_model = tensorflow.keras.Model(decoder_input, decoder_output)
decoder_model.summary()

encoded = encoder_model(input_data)
decoded = decoder_model(encoded)


autoencoder = tensorflow.keras.models.Model(input_data, decoded)


autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
autoencoder.summary()


autoencoder.fit(train_data, train_data, epochs=200, batch_size=64, validation_data=(test_data, test_data))



# Next visualize the decoded data against original data.
fig1, (ax1,ax2) = plt.subplots(nrows=2, ncols=1)
ax1.plot(test_data[1])
# test = np.expand_dims(test_data[1],axis=(2,1))
test = np.expand_dims(test_data[1], axis=0)

testPredicted = autoencoder.predict(test)
testPredicted = testPredicted[0,:,:]
ax2.plot(testPredicted)

# plot Latent feature clusters in scatterplot. 



x = []
y = []
# z = []
for i in range(0,len(test_data)):
    # z.append(testy[i])
    test = test_data[i]
    test = np.expand_dims(test, axis=0)
    op = encoder_model.predict(test)
    x.append(op[0][0])
    y.append(op[0][1])
    # z.append(op[1][1])

import pandas as pd
# import seaborn as sns
import seaborn as sns

df = pd.DataFrame()
df['x'] = x
df['y'] = y
# df['z'] = ["digit-"+str(k) for k in z]

plt.figure(figsize=(8, 6))
sns.scatterplot(x='x', y='y', data=df)
plt.show()



























# x = np.arange(5)
# y = np.exp(x)
# fig1, ax1 = plt.subplots()
# ax1.plot(x, y)
# ax1.set_title("Axis 1 title")
# ax1.set_xlabel("X-label for axis 1")

# z = np.sin(x)
# fig2, (ax2, ax3) = plt.subplots(nrows=2, ncols=1) # two axes on figure
# ax2.plot(x, z)
# ax3.plot(x, -z)

# w = np.cos(x)
# ax1.plot(x, w) # can continue plotting on the first axis
