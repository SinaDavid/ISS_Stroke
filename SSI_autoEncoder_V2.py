# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:55 2021



To Do list:
    - make github issues.
    - add the other 3 perturbation types to the list.







Questions to ask:
    - split on group level for train / test. This more realistic to the real world. However, when data is split accuracy of the
    model varies enormous indicating that we have learned some typical responses but not all. 
    - 64 filters? how does that work?
    - split by group during training?



@author: michi
"""

# cnn model
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
from keras.layers import MaxPool2D, AvgPool2D, BatchNormalization, Dropout, Input
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import itertools
from sklearn.metrics import confusion_matrix
from keras.regularizers import l2
from keras import regularizers
import tensorflow
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
#dataDir1 = 'F:\\SSI_data\\10sec\\ML_markervel\\NonParCont\\' #--> 0.55
#dataDir2 = 'F:\\SSI_data\\10sec\\ML_markervel\\ParCont\\'  #--> 0.39
#dataDir3 = 'F:\\SSI_data\\10sec\\ML_markervel\\NonParYpsi\\' #--> 0.53
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
### Normalize / scale input ###
X_data /=2
if numberPer > 1:
    X_data = np.hstack((X_data, perturbationsSeries))
if numberPer == 1:
    numberPer = 0    



##### Get the dependent y data from filenames. #######
y = [0]
for indx in range(len(filename)):
    x = filename[indx].split('FR')
    y = np.vstack((y,int(x[1][0])))    
y = np.delete(y,(0),axis=0)

###### Make y a categorical value #####    
y_cat = to_categorical(y)




########    Reshape  the X data ############
X_data = X_data.reshape(len(y_cat),epochLength,n_features+numberPer) 



train_data = X_data[np.arange(start=1,stop=np.shape(X_data)[0],step=2)]
test_data = X_data[np.arange(start=2,stop=np.shape(X_data)[0],step=2)]


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

#np.round((epochLength/3),0)

#import tensorflow
#input_data = tensorflow.keras.layers.Input(shape=(epochLength, n_features+numberPer, 1))
#encoder = tensorflow.keras.layers.Conv1D(64, (5), activation='relu')(input_data)
#encoder = tensorflow.keras.layers.MaxPooling1D((2,2))(encoder)
#
#encoder = tensorflow.keras.layers.Conv1D(64, (3,3), activation='relu')(encoder)
#encoder = tensorflow.keras.layers.MaxPooling1D((2,2))(encoder)
#
#encoder = tensorflow.keras.layers.Conv1D(32, (3,3), activation='relu')(encoder)
#encoder = tensorflow.keras.layers.MaxPooling1D((2,2))(encoder)
#
#encoder = tensorflow.keras.layers.Flatten()(encoder)
#encoder = tensorflow.keras.layers.Dense(16)(encoder)

#def sample_latent_features(distribution):
#    distribution_mean, distribution_variance = distribution
#    batch_size = tensorflow.shape(distribution_variance)[0]
#    random = tensorflow.keras.backend.random_normal(shape=(batch_size, tensorflow.shape(distribution_variance)[1]))
#    return distribution_mean + tensorflow.exp(0.5 * distribution_variance) * random




#encoder = Sequential()
#input_data = Input(shape=(epochLength, n_features+numberPer))
#encoder = Conv1D(filters=32, kernel_size=3, activation='relu')(input_data)#, input_shape=(epochLength,n_features+numberPer))
#encoder = Conv1D(filters=48, kernel_size=3, activation='relu')(encoder)
#encoder = MaxPooling1D(pool_size=2)(encoder)
#encoder = Flatten()(encoder)
#encoder = Dense(4,activation='softmax')(encoder)
#encoder.compile()
#encoder.summary()
#
#
#distribution_mean = tensorflow.keras.layers.Dense(2, name='mean')(encoder)
#distribution_variance = tensorflow.keras.layers.Dense(2, name='log_variance')(encoder)
#latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])


#encoder_model = tensorflow.keras.Model(input_data, latent_encoding)
#encoder_model.summary()

encoder = Sequential()
encoder.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(epochLength,n_features+numberPer)))#,kernel_regularizer=regularizers.l2(0.01), bias_regularizer=l2(0.01)
#model.add(Conv1D(filters=48, kernel_size=3, activation='relu')) #,kernel_regularizer=regularizers.l2(0.01), bias_regularizer=l2(0.01)# Yang --> this one is usually larger.
encoder.add(Conv1D(filters=32,kernel_size=3, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
#encoder.add(Dropout(0.5))
encoder.add(MaxPooling1D(pool_size=2))
encoder.add(Flatten())
#model.add(Dense(100, activation='relu'),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
encoder.add(Dense(32))#, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)
encoder.add(Dense(2, activation='relu'))
#encoder.compile(loss='categorical_crossentropy', optimizer='adam')#, metrics=['accuracy']


#distribution_mean = tensorflow.keras.layers.Dense(2, name='mean')(encoder)
#distribution_variance = tensorflow.keras.layers.Dense(2, name='log_variance')(encoder)
#latent_encoding = tensorflow.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])






encoder.compile(loss='mse', optimizer='adam')
encoder.summary()




encoder.fit(train_data, train_data, epochs=20, batch_size=88)#, validation_data=(test_data, test_data))



#batch_size=64,




















################# Define the model. CNN_1D  ###########################
#n_timesteps, n_outputs = epochLength, 2
#model = Sequential()
#model.add(Conv1D(filters=32, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features+numberPer)))#,kernel_regularizer=regularizers.l2(0.01), bias_regularizer=l2(0.01)
##model.add(Conv1D(filters=48, kernel_size=3, activation='relu')) #,kernel_regularizer=regularizers.l2(0.01), bias_regularizer=l2(0.01)# Yang --> this one is usually larger.
#model.add(Conv1D(filters=32,kernel_size=3, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
#model.add(Dropout(0.5))
#model.add(MaxPooling1D(pool_size=2))
#model.add(Flatten())
##model.add(Dense(100, activation='relu'),kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01))
#model.add(Dense(64, kernel_regularizer=l2(0.01), bias_regularizer=l2(0.01)))
#model.add(Dense(n_outputs, activation='softmax'))
#model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
#
#print(model.summary())
#
#
#
#initial_weights = model.get_weights()
#
################## Run the model and evaluate output #################
#logo = LeaveOneGroupOut()
#logo.get_n_splits(X_data, y_cat, group)
#test_scores = []
#train_scores = []
#for train_index, test_index in logo.split(X_data, y_cat, group):
##     print("TRAIN:", train_index, "TEST:", test_index)
#     X_train, X_test = X_data[train_index], X_data[test_index]
#     y_train, y_test = y_cat[train_index], y_cat[test_index]
#     #print(X_train, X_test, y_train, y_test)
#     model.set_weights(initial_weights)
#     h = model.fit(X_train, y_train,
#                  verbose=1,
#                  epochs=100,
#                  callbacks=[EarlyStopping(monitor='loss', patience=2)],class_weight=class_weight)
#     r = model.evaluate(X_train, y_train, verbose = 0)
#     train_scores.append(r[-1])
#     e = model.evaluate(X_test, y_test, verbose=0)
#     test_scores.append(e[-1])
#
#
#
#
#FinalModelEvaluation = np.mean(test_scores)
#FinalModelEvaluationSD = np.std(test_scores)








































##### Some stuff from previous versions.


#train_sizes = (len(X_train) * np.linspace(0.1,0.999,4)).astype(int)




#for train_size in train_sizes:
#    X_train_frac, _, y_train_frac, _ = train_test_split(X_train,y_train, train_size=train_size)
#    # at each iteration, set original weights
#    # to the initial random weights
#    model.set_weights(initial_weights)
#    h = model.fit(X_train_frac, y_train_frac,
#                  verbose=1,
#                  epochs=100,
#                  callbacks=[EarlyStopping(monitor='loss', patience=2)],class_weight=class_weight)#,
#                  #class_weight=class_weights
#    r = model.evaluate(X_train_frac, y_train_frac, verbose = 0)
#    train_scores.append(r[-1])
#    e = model.evaluate(X_test, y_test, verbose=0)
#    test_scores.append(e[-1])
#    print ("Done size: ", train_size )
#plt.figure()
#plt.plot(train_sizes, train_scores, 'o-', label="training score")
#plt.plot(train_sizes, test_scores, 'o-',label="Cross validation scores")
#plt.legend(loc="best")
#
#print(model.evaluate(X_test,y_test))
#predicted_labels = model.predict(X_test)
#predicted_labels = np.argmax(predicted_labels, axis=1)
#precision_recall_fscore_support
#
#y_train_count = np.argmax(y_train, axis = 1) 
#
#unique, counts = np.unique(y_train_count, return_counts=True)
#dict(zip(unique, counts)) # MAKE A BAR PLOT HERE TO PUT INTO THE PRESENTATION.
#y_pos = np.arange(len(counts))
#plt.figure(0, figsize=(15, 8))
##fig = plt.figure(0)
#plt.bar(y_pos, counts, align='center', alpha=0.5)
#plt.xticks(y_pos, unique)
#plt.ylabel('Number of Observations')
#plt.title('Class observations')
# 
#plt.show()
#
#
#y_test = np.argmax(y_test, axis=1)
#cm = confusion_matrix(y_test, predicted_labels)
#print('Confusion matrix')
#fig2, ax = plt.subplots()
#plot_confusion_matrix(cm)
##fig2.savefig(Name2)
#cm1 = cm / cm.astype(np.float).sum(axis=1)
#print('Confusion matrix, with normalization')
#fig, ax = plt.subplots()
#plot_confusion_matrix(cm1)
#plt.show()
#performance = precision_recall_fscore_support(y_test, predicted_labels, average='macro')

















