# -*- coding: utf-8 -*-
"""
Created on Tue Jun  8 14:32:55 2021



To Do list:
    - make github issues.
    - 











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
from keras.layers import MaxPool2D, AvgPool2D, BatchNormalization, Dropout
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_recall_fscore_support
import itertools
from sklearn.metrics import confusion_matrix

from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
## Settings


seconds = 10
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
dataDir = 'F:\\SSI_data\\10sec\\ML_markervel\\ParYpsi\\'
#file = 'S001RQ1_MLPert1_Trial1_ParYpsi_FR0_V1'
group = []
for file in os.listdir(dataDir):
    data.append(spio.loadmat(dataDir + file))
    filename.append(file)
    group.append(int(file[1:4]))
    
group = np.array(group)
    




#### first attempt. 
    # we are going for the RANK. --? columns 43/45

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
#       x = data[indx]['markervel'][0:epochLength,42:45]
       X_data = np.vstack((X_data,x))
       
### Always remove first row (due to initialize issues)       
X_data = np.delete(X_data, (0), axis=0) 



##### Get the dependent y data from filenames. #######
y = [0]
for indx in range(len(filename)):
    x = filename[indx].split('FR')
    y = np.vstack((y,int(x[1][0])))    
y = np.delete(y,(0),axis=0)

###### Make y a categorical value #####    
y_cat = to_categorical(y)




########    Reshape  the X data ############
X_data = X_data.reshape(len(y_cat),epochLength,n_features) 
####### Split data by group (subject level) #################
group_kfold = GroupKFold(n_splits=2)
group_kfold.get_n_splits(X_data, y_cat, group)
gss = GroupShuffleSplit(n_splits=2, train_size=.7, random_state=42)

for train_index, test_index in gss.split(X_data, y_cat, group):
#    print("TRAIN:", train_index, "TEST:", test_index)
    X_train, X_test = X_data[train_index], X_data[test_index]
    y_train, y_test = y_cat[train_index], y_cat[test_index]
#    print(X_train, X_test, y_train, y_test)

####### Split data by trial (within subject level) #################
#X_train, X_test, y_train, y_test = train_test_split(X_data, y_cat, test_size=0.3, random_state=42)
###################################################################################






########### Determine imbalance and calculate weights  #######################
n_samples = len(y)
n_classes = 2
n_samplesj = np.sum(y == 0)
n_samplesi = np.sum(y == 1)
#n_samplesk = np.sum(finalDatay == 2)
#n_samplesg = np.sum(finalDatay == 3)
wj=n_samples / (n_classes * n_samplesj)
wi=n_samples / (n_classes * n_samplesi)
#wk=n_samples / (n_classes * n_samplesk)
#wg=n_samples / (n_classes * n_samplesg)
class_weight = {0: wj,
                1: wi
                }



################# Define the model. CNN_1D  ###########################
n_timesteps, n_features, n_outputs = X_train.shape[1], X_train.shape[2], y_train.shape[1]
model = Sequential()
model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(n_timesteps,n_features)))
model.add(Conv1D(filters=64, kernel_size=3, activation='relu'))
model.add(Dropout(0.5))
model.add(MaxPooling1D(pool_size=2))
model.add(Flatten())
model.add(Dense(100, activation='relu'))
model.add(Dense(n_outputs, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print(model.summary())


################# Run the model and evaluate output #################
initial_weights = model.get_weights()

train_sizes = (len(X_train) * np.linspace(0.1,0.999,4)).astype(int)

train_scores = []
test_scores = []

for train_size in train_sizes:
    X_train_frac, _, y_train_frac, _ = train_test_split(X_train,y_train, train_size=train_size)
    # at each iteration, set original weights
    # to the initial random weights
    print('check1')
    model.set_weights(initial_weights)
    print('check2')
    h = model.fit(X_train_frac, y_train_frac,
                  verbose=1,
                  epochs=100,
                  callbacks=[EarlyStopping(monitor='loss', patience=2)],class_weight=class_weight)#,
                  #class_weight=class_weights
    print('check3')
    r = model.evaluate(X_train_frac, y_train_frac, verbose = 0)
    print('check4')
    train_scores.append(r[-1])
    print('check5')
    e = model.evaluate(X_test, y_test, verbose=0)
    test_scores.append(e[-1])
    print ("Done size: ", train_size )
plt.figure()
plt.plot(train_sizes, train_scores, 'o-', label="training score")
plt.plot(train_sizes, test_scores, 'o-',label="Cross validation scores")
plt.legend(loc="best")

print(model.evaluate(X_test,y_test))
predicted_labels = model.predict(X_test)
predicted_labels = np.argmax(predicted_labels, axis=1)
precision_recall_fscore_support

y_train_count = np.argmax(y_train, axis = 1) 

unique, counts = np.unique(y_train_count, return_counts=True)
dict(zip(unique, counts)) # MAKE A BAR PLOT HERE TO PUT INTO THE PRESENTATION.
y_pos = np.arange(len(counts))
plt.figure(0, figsize=(15, 8))
#fig = plt.figure(0)
plt.bar(y_pos, counts, align='center', alpha=0.5)
plt.xticks(y_pos, unique)
plt.ylabel('Number of Observations')
plt.title('Class observations')
 
plt.show()


y_test = np.argmax(y_test, axis=1)
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

















