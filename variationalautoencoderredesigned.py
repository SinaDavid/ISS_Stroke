# -*- coding: utf-8 -*-
"""
Created on Mon Feb  7 08:15:20 2022

@author: michi

Step1 load data

step 2 organize data
    tune hz
    tune window length (5 sec and 10 sec) /dubbling frequency.
    augment data by overlapping window of maximum 50%.
    
step 3 split dataset
    
step 4 initialize model (2 / 3 / 4 latentfeatures)

step 5 train model / load pretrained model. 


step 6 apply model to explore further validation steps. 

"""

## PRE STEP STUFF ##    
windowLength = 200 # do not change this.
############################
## Some required settings ##
############################
inputColumns = [0, 3, 6, 9, 12, 15]
latentFeatures = 4#2  3 / 
trainModel =  True #False #False#
frequency = 50 # 20 / 50  
############################
##### end of settings! #####
############################


if frequency == 20:
    timeLength = 10
    refactorValue = 5 # taking each 5th value for resampling
else:
    timeLength = 4
    refactorValue = 2 


## paths to models / raw data / initialised weights / loaded weights 
pathToDataRelativeAngles = "F:\\SSI_data\\OneFilePerPerson1\\OneFilePerPerson\\prox_relativeangles\\"
pathToDataEvents = "F:\\SSI_data\\OneFilePerPerson1\\OneFilePerPerson\\Events\\"
pathToInitializedWeights = "C:\\Users\\michi\\Desktop\\SSI_Stroke\\initializedWeights\\"
pathToTrainedWeights = "C:\\Users\\michi\\Desktop\\SSI_Stroke\\trainedWeights\\12022022_withoutS017_S022\\"

pathToCleanData = "F:\\SSI_data\\OneFilePerPerson1\\OneFilePerPerson\\"
########################
### Define functions ###
########################
seed_value= 1
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)
# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
tf.compat.v1.set_random_seed(seed_value)
# 5. Configure a new global `tensorflow` session
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# 1: Loading required libararies.
# import tensorflow as tf
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
# import os
import scipy.io as spio
from matplotlib import cm as CM
#from matplotlib import mlab as ML
# import numpy as np
from keras.callbacks import EarlyStopping
#from keras.layers import MaxPool2D, AvgPool2D, BatchNormalization, Dropout, Input,Activation
#from keras import layers
#rom keras.layers import Activation
#from sklearn import preprocessing
#from sklearn.metrics import confusion_matrix
#from sklearn.metrics import precision_recall_fscore_support
#import itertools
#from sklearn.metrics import confusion_matrix
#from keras.regularizers import l2
#from keras.models import load_model
#from keras import regularizers
from matplotlib.lines import Line2D
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,LeaveOneGroupOut,
                                     StratifiedKFold, GroupShuffleSplit,
                                     GroupKFold, StratifiedShuffleSplit)
import pickle
import pandas as pd
import seaborn as sns
plt.close('all')

def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())

def nrmse(predictions, targets):
    return np.sqrt((((predictions - targets)/targets) ** 2).mean())

def get_loss(distribution_mean, distribution_variance):
    
    def get_reconstruction_loss(y_true, y_pred):
        reconstruction_loss = tf.keras.losses.mse(y_true, y_pred)
        reconstruction_loss_batch = tf.reduce_mean(reconstruction_loss)
        return reconstruction_loss_batch*28*28
    
    def get_kl_loss(distribution_mean, distribution_variance):
        kl_loss = 1 + distribution_variance - tf.square(distribution_mean) - tf.exp(distribution_variance)
        kl_loss_batch = tf.reduce_mean(kl_loss)
        return kl_loss_batch*(-0.5)
    
    def total_loss(y_true, y_pred):
        reconstruction_loss_batch = get_reconstruction_loss(y_true, y_pred)
        kl_loss_batch = get_kl_loss(distribution_mean, distribution_variance)
        # print('print recon loss: ' + str(reconstruction_loss_batch) + 'print kl loss:' + str(kl_loss_batch))
        # pickle.dump(test, locals())
        # f = open('testing123.pckl', 'wb')
        # pickle.dump(reconstruction_loss_batch, f)
        # f.close()
        return reconstruction_loss_batch + kl_loss_batch
        
    return total_loss

def sample_latent_features(distribution):
    distribution_mean, distribution_variance = distribution
    batch_size = tf.shape(distribution_variance)[0]
    random = tf.keras.backend.random_normal(shape=(batch_size, tf.shape(distribution_variance)[1]))
    return distribution_mean + tf.exp(0.5 * distribution_variance) * random


############### STEP 1 ###############################
filename = []
data = []
group = []
#### THE TIMESERIES #####
if 'pathToDataRelativeAngles' in locals():
    for file1 in os.listdir(pathToDataRelativeAngles):
        if file1[0:4] != 'S017':
            if file1[0:4] != 'S022':
                data.append(spio.loadmat(pathToDataRelativeAngles + file1))
                filename.append(file1)
                if file1[0]=='W':
                    group.append(int(file1[1:3]))
                else:
                    group.append(int(file1[1:4]))
            else:
                print(file1[0:4])
        else:
            print(file1[0:4])
group = np.array(group)
# get the dict out of the list
for numtrials in range(len(filename)):
    data[numtrials] = (data[numtrials]['prox_relativeangles'])#'markervel'  
# Finding the first file with 'W'.
start = 0
for indx in range(0,len(filename)): # find the first file with healthy "w"file name
    if filename[indx][0]=='W':
        start = indx
        break    
# Next resample / remove 3th element of w files to resample towards 100hz
for indx in range(start,len(data)): # LET OP DIT IS EEN HARDCODED 108
    delArray = np.arange(0,len(data[indx]),3)
    data[indx]= np.delete(data[indx], delArray, axis=0)
## Nu is alles 100Hz!
###### THE GAIT EVENTS ############
groupE = []
dataE = []
filenameE = []
if 'pathToDataEvents' in locals():
    for file1E in os.listdir(pathToDataEvents):
        if file1E[0:4] != 'S017':
            if file1E[0:4] != 'S022':
                dataE.append(spio.loadmat(pathToDataEvents + file1E))
                filenameE.append(file1E)
                if file1E[0]=='W':
                    groupE.append(int(file1E[1:3]))
                else:
                    groupE.append(int(file1E[1:4]))
            else:
                print(file1E[0:4])
        else:
            print(file1E[0:4])
    #     perturbationType.append(numberPer)
    # numberPer +=1
groupE = np.array(groupE)
# get the dict out of the list
for numtrials in range(len(filenameE)):
    dataE[numtrials] = (dataE[numtrials]['Events'])#'markervel'


### Nu de events naar 100Hz ###
for indx in range(start,len(dataE)):
    dataE[indx] = np.round(dataE[indx]*0.6666,0)

#### Nu alles naar 100hz qua events. #####


############### STEP 2 ###################################

# resample
for indx in range(len(dataE)):
    dataE[indx] = np.round(dataE[indx]*(frequency/100),0)
for indx in range(len(data)):
    data[indx] = data[indx][::refactorValue] 

# overlapping windows
count = 0
trackGroup = []
trackGroup1 = []
dataAugmented = np.zeros(len(inputColumns))
for indx in range(len(data)):
    if len(data[indx])>windowLength:
        for indx2 in range(0,len(dataE[indx]),2): # in steps of to so every second step can be a new data for training or testing.

            index = dataE[indx][indx2]
            tempArray = data[indx][int(index[0]):]
            if len(tempArray)>windowLength:
                count +=1
                dataAugmented = np.vstack((dataAugmented,tempArray[0:windowLength,inputColumns]))
                trackGroup.append(indx)
                if indx < start:
                    trackGroup1.append('SS'+str(group[indx]))
                else:
                    trackGroup1.append('W'+str(group[indx]))
#  Always remove first row (due to initialize issues)       
dataAugmented = np.delete(dataAugmented, (0), axis=0) 


######## STEP 3 SPLIT THE DATASET ##################################


y = [0]
# group = []
for indx in range(len(trackGroup)):
    x = filename[trackGroup[indx]].split('FR')
    print(x)
    if x[0][0]=='W':
        if x[0][1:3].isdigit():
            if int(x[0][1:3])<25:
                y = np.vstack((y,3))
            else:
                y = np.vstack((y,2))
    else:
    # if xx = x[0].split('W')
        y = np.vstack((y,int(x[1][0])))   
    # group.append(int(filename[1:4]))
    print(y)
y = np.delete(y,(0),axis=0)
# group = np.array(group)
########    Reshape  the X data ############
dataAugmented = dataAugmented.reshape(len(y),200,len(inputColumns))

externSubjects = ['','','','']#'SS1','SS38','W1','W38'
indexenexternSubjects = []
indexeninternSubjects = []
train_data = []
test_data = []
extern_data = []

## External validation stuff ###
for indx in range(0,len(trackGroup1)):
    if  trackGroup1[indx]!=externSubjects[0] and trackGroup1[indx]!=externSubjects[1] and trackGroup1[indx]!=externSubjects[2] and trackGroup1[indx]!=externSubjects[3]:
        indexeninternSubjects.append(indx)     
    else:
        indexenexternSubjects.append(indx)
        # print(trackGroup1[indx])
extern_data = dataAugmented[indexenexternSubjects,:,:]
other_data = dataAugmented[indexeninternSubjects,:,:] 
y_adapted = y[indexeninternSubjects] # which group (stroke F / NF / healthy etc)

# Make a int list of subjects.
subject = 0
groupsplit=[subject]
for indx in range(1,len(trackGroup1)):
    if trackGroup1[indx]==trackGroup1[indx-1]:
        groupsplit.append(subject)
    else:
        subject+=1
        groupsplit.append(subject)
groupsplit = np.array(groupsplit)
groupsplit =  groupsplit[indexeninternSubjects]

## Group split is a variable which contains a number for each subject. the length is equal to other_data (which needs to be split)

### saving other_data, y_adapted and groupsplit. so next the manuscrit can start over here!

# np.save(pathToCleanData + "stored_other_data_" + str(latentFeatures) + "_" + str(frequency),other_data)
# np.save(pathToCleanData + "stored_y_adapted_" + str(latentFeatures) + "_" + str(frequency),y_adapted)
# np.save(pathToCleanData + "stored_groupsplit_" + str(latentFeatures) + "_" + str(frequency),groupsplit)



# ##### During Escience session we can start from here. #############

# other_data = np.load(pathToCleanData + "stored_other_data_" + str(latentFeatures) + "_" + str(frequency)+".npy")
# y_adapted = np.load(pathToCleanData + "stored_other_data_" + str(latentFeatures) + "_" + str(frequency)+".npy")
# groupsplit = np.load(pathToCleanData + "stored_other_data_" + str(latentFeatures) + "_" + str(frequency)+".npy")




# Now divide other_data into train and test data by groupsplit.
# ####### Split data by group (subject level) #################
gss = GroupShuffleSplit(n_splits=2, train_size=.65, random_state=1)
gss.get_n_splits()

for train_idx, test_idx in gss.split(other_data, y_adapted, groupsplit):
     print("TRAIN:", train_idx, "TEST:", test_idx)
    

train_data = other_data[train_idx,:,:]
test_data = other_data[test_idx,:,:]
train_y = y_adapted[train_idx]
test_y = y_adapted[test_idx]


############# STEP 4 INITIALIZE THE MODEL #########################
tf.compat.v1.disable_eager_execution()
input_data = tf.keras.layers.Input(shape=(windowLength, 6))
# input_data = tensorflow.keras.layers.Input(epochLength, 6)
encoder = tf.keras.layers.Conv1D(64, 5,activation='relu',name='sina1')(input_data)
# encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
encoder = tf.keras.layers.MaxPooling1D(2)(encoder)
encoder = tf.keras.layers.Conv1D(64, 3,activation='relu',name='sina2')(encoder)
# encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
encoder = tf.keras.layers.MaxPooling1D(2)(encoder)
encoder = tf.keras.layers.Conv1D(32, 3, activation='relu',name='sina3')(encoder)
# encoder = tensorflow.keras.layers.LeakyReLU(alpha=0.1)(encoder)
encoder = tf.keras.layers.MaxPooling1D(2)(encoder)
encoder = tf.keras.layers.Flatten()(encoder)
encoder = tf.keras.layers.Dense(16,name='sina4')(encoder)
encoder = tf.keras.layers.Dense(8,name='sina5')(encoder)
encoder = tf.keras.layers.Dense(8,name='sina6')(encoder)
encoder = tf.keras.layers.LeakyReLU(alpha=0.1)(encoder)
distribution_mean = tf.keras.layers.Dense(latentFeatures, name='mean')(encoder)
distribution_variance = tf.keras.layers.Dense(latentFeatures, name='log_variance')(encoder)
latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])
encoder_model = tf.keras.Model(input_data, latent_encoding)
encoder_model.summary()
################### DECODER PART ############
decoder_input = tf.keras.layers.Input(shape=(latentFeatures)) 
decoder = tf.keras.layers.Dense(64)(decoder_input)
decoder = tf.keras.layers.Reshape((1, 64))(decoder)
decoder = tf.keras.layers.Conv1DTranspose(16, 3, activation='relu')(decoder)
decoder = tf.keras.layers.Conv1DTranspose(32, 5, activation='relu')(decoder)
decoder = tf.keras.layers.UpSampling1D(5)(decoder)
decoder = tf.keras.layers.Conv1DTranspose(64, 5, activation='relu')(decoder)
decoder = tf.keras.layers.UpSampling1D(5)(decoder)
decoder_output = tf.keras.layers.Conv1DTranspose(6, 6)(decoder)
decoder_output = tf.keras.layers.LeakyReLU(alpha=0.1)(decoder_output)
decoder_model = tf.keras.Model(decoder_input, decoder_output)
print("\ndecoder summary")
decoder_model.summary()

################ STEP 5 ##########################
decoderWeightstesting = []
encoderWeightstesting = []
if trainModel:
    print('train het model')
    f = open(pathToInitializedWeights + 'decoderInitialWeightsBottleneck'+ str(latentFeatures) + '.pckl', 'rb')
    decoderWeightstesting = pickle.load(f)
    f.close()
    
    f = open(pathToInitializedWeights + 'encoderInitialWeightsBottleneck'+ str(latentFeatures) + '.pckl', 'rb')
    encoderWeightstesting = pickle.load(f)
    f.close()
#  INSERTING WEIGHTS AND BIASES IN ENCODER & DECODER MODEL # for reproducibility purposes.
    for indx in range(0,len(decoder_model.layers)):
        a = decoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            decoder_model.layers[indx].set_weights([decoderWeightstesting[indx][0],decoderWeightstesting[indx][1]])
    
    for indx in range(0,len(encoder_model.layers)):
        a = encoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            encoder_model.layers[indx].set_weights([encoderWeightstesting[indx][0],encoderWeightstesting[indx][1]])  
    encoded = encoder_model(input_data)
    decoded = decoder_model(encoded)
    autoencoder = tf.keras.models.Model(input_data, decoded)
    autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
    print("\nautoenoder summary")
    autoencoder.summary()
    history = autoencoder.fit(train_data,
                              train_data,
                              epochs=300,
                              batch_size=64,
                              callbacks=[EarlyStopping(monitor='loss', patience=25)],
                              validation_data=(test_data, test_data))
    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    #### storing train weights and biases ####
    decoderWeightstrained = []
    encoderWeightstrained = []

### Now storing the trained weights ###

    for indx in range(0,len(decoder_model.layers)):
        decoderWeightstrained.append(decoder_model.layers[indx].get_weights())
    
    for indx in range(0,len(encoder_model.layers)):
        encoderWeightstrained.append(encoder_model.layers[indx].get_weights())
        
    
    f = open(pathToTrainedWeights + 'decoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_'+ str(frequency) + '_.pckl', 'wb')
    pickle.dump(decoderWeightstrained, f)
    f.close()
    
    f = open(pathToTrainedWeights + 'encoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_'+ str(frequency) + '_.pckl', 'wb')
    pickle.dump(encoderWeightstrained, f)
    f.close()

else: 
    print('more to come')
    print('here i will load trained weights and continu')
    ################### LOADING STORED WEIGHTS AND BIASES ########################################
    f = open(pathToTrainedWeights + 'decoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_'+ str(frequency) + '_.pckl', 'rb')
    decoderWeightstesting = pickle.load(f)
    f.close() 
    f = open(pathToTrainedWeights + 'encoderTRAINEDWeightsBottleneck'+ str(latentFeatures) + '_'+ str(frequency) + '_.pckl', 'rb')
    encoderWeightstesting = pickle.load(f)
    f.close()

######################  INSERTING WEIGHTS AND BIASES IN ENCODER & DECODER MODEL ################################

    for indx in range(0,len(decoder_model.layers)):
        a = decoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            decoder_model.layers[indx].set_weights([decoderWeightstesting[indx][0],decoderWeightstesting[indx][1]])
    
    for indx in range(0,len(encoder_model.layers)):
        a = encoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            encoder_model.layers[indx].set_weights([encoderWeightstesting[indx][0],encoderWeightstesting[indx][1]])  


    encoded = encoder_model(input_data)
    decoded = decoder_model(encoded)
    autoencoder = tf.keras.models.Model(input_data, decoded)
    autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
    print("\nautoenoder summary")
    autoencoder.summary()


##################################### EIND OF THIS PART #########################
    


############# STEP 6 EXPLORE MODEL PERFORMANCES ############################

sprong = 100
originalData = []
Reconstructed = []
fig2, axes = plt.subplots(6, 2, figsize=(15, 5))#, sharey=True
for indx in range(0,6): 
    originalData.append(test_data[indx*sprong])
    axes[indx,0].plot(test_data[indx*sprong])
    axes[0, 0].set_title('original timeseries')
    test = np.expand_dims(test_data[indx*sprong], axis=0)
    testPredicted = autoencoder.predict(test)
    testPredicted = testPredicted[0,:,:]
    Reconstructed.append(testPredicted)
    axes[indx,1].plot(testPredicted)
    axes[0, 1].set_title('Reconstructed timeseries')
    
SD_org = np.transpose(np.zeros(6))
SD_dec = np.zeros(6) 
signalRange = np.zeros(6)

for i in range(0,len(test_data)):
    test_new = test_data[i]
    test_new_dec = np.expand_dims(test_new, axis=0)
    test_dec = autoencoder.predict(test_new_dec)[0]
    test_new = np.expand_dims(test_new, axis=0)[0] 
    # calculate orginal SD_org
    sd_temp = []#0,0,0,0,0,0
    sd_temp_Dec = []
    range_temp = []
    for ii in range(0,6):
        sd_temp = np.hstack((sd_temp, np.std(test_new[0:200,ii])))
        sd_temp_Dec = np.hstack((sd_temp_Dec, np.std(test_dec[0:200,ii])))
        range_temp = np.hstack((range_temp,np.ptp(test_dec[0:200,ii])))
    SD_org = np.vstack((SD_org,sd_temp))
    SD_dec = np.vstack((SD_dec,sd_temp_Dec))
    signalRange = np.vstack((signalRange,range_temp))
    
encoded = []
for i in range(0,len(test_data)):
    # z.append(testy[i])
    test_new = test_data[i]
    test_new = np.expand_dims(test_new, axis=0)
    op = encoder_model.predict(test_new)
    encoded.append(op[0])


if latentFeatures ==2:
    xx = []
    yy = []
    z = []
    groupcolor = []
    for i in range(0,len(np.array(encoded))):
        xx.append(np.array(encoded)[i][0])
        yy.append(np.array(encoded)[i][1])
        z.append(test_y[i]) # Fall risk / group
    xx = np.array(xx)
    yy = np.array(yy)
    df1 = pd.DataFrame()
    df1['xx'] = xx
    df1['yy'] = yy
    df1['z'] = ["fall risk-"+str(k) for k in z]
    df1['groupcolor'] = ["subject-"+str(k) for k in test_data]
    df1.loc[df1['z'] == 'fall risk-[1]', 'z'] = 'orange'
    df1.loc[df1['z'] == 'fall risk-[0]', 'z'] = 'blue'
    df1.loc[df1['z'] == 'fall risk-[2]', 'z'] = 'green'
    df1.loc[df1['z'] == 'fall risk-[3]', 'z'] = 'lightgreen' 

    fig = plt.figure(10)
    plt.scatter(df1['xx'], df1['yy'], c=df1['z'],label=df1['z'],edgecolors='white')
    plt.title('2D Variational autoencoder / decoder')
    plt.xlabel('Latent feature 1')
    plt.ylabel('Latent feature 2')
    colors = ['orange', 'blue', 'green','lightgreen','red','firebrick']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['stroke survivors with falls', 'stroke survivors without falls', 'older adults','younger adults','selected data point for space exploration','created data point by adding/subtracting two for space exploration']
    plt.legend(lines, labels,loc='lower left')

    encoded1 = [0,0]
    for indx in range(len(encoded)):
        encoded_temp = encoded[indx]
        encoded1 = np.vstack((encoded1,np.transpose(encoded_temp)))
    encoded1 = np.delete(encoded1,(0),axis=0)

    fig = plt.figure(13)
    ax1 = fig.add_subplot(321)
    ax1.title.set_text('Range 1')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=signalRange[:,0], cmap=CM.jet,vmin=0, vmax=100, bins=None)
    plt.colorbar()

    ax2 = fig.add_subplot(322)
    ax2.title.set_text('Range2')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=signalRange[:,1], cmap=CM.jet,vmin=0, vmax=100, bins=None)
    plt.colorbar()
    
    ax3 = fig.add_subplot(323)
    ax3.title.set_text('Range 3 ')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=signalRange[:,2], cmap=CM.jet,vmin=0, vmax=100, bins=None)
    plt.colorbar()
    
    ax4 = fig.add_subplot(324)
    ax4.title.set_text('Range 4')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=signalRange[:,3], cmap=CM.jet,vmin=0, vmax=100, bins=None)
    plt.colorbar()
    
    ax5 = fig.add_subplot(325)
    ax5.title.set_text('Range 5')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=signalRange[:,4], cmap=CM.jet,vmin=0, vmax=100, bins=None)
    plt.colorbar()
    
    ax6 = fig.add_subplot(326)
    ax6.title.set_text('Range 6')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=signalRange[:,5], cmap=CM.jet,vmin=0, vmax=100, bins=None)
    plt.colorbar()
    plt.suptitle('Range ',fontsize=20)




    symmetry = []
    symmetry = signalRange[:,0]/signalRange[:,3]
    fig = plt.figure(14)
    ax1 = fig.add_subplot(321)
    ax1.title.set_text('symmetry 1')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=symmetry, cmap=CM.jet,vmin=0.5, vmax=1.5, bins=None)
    plt.colorbar()
    symmetry = signalRange[:,1]/signalRange[:,4]
    ax2 = fig.add_subplot(323)
    ax2.title.set_text('symmetry 2')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=symmetry, cmap=CM.jet,vmin=0.5, vmax=1.5, bins=None)
    plt.colorbar()
    symmetry = signalRange[:,2]/signalRange[:,5]
    ax3 = fig.add_subplot(325)
    ax3.title.set_text('symmetry 3 ')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=symmetry, cmap=CM.jet,vmin=0.5, vmax=1.5, bins=None)
    plt.colorbar()



#### Additional validation ######


#############################################################################
### training & test data  correlation / rmse and nrmse plus visualisation ###
#############################################################################

error_train = [0,0,0,0,0,0]
errortemp = []
correlationtemp = []
correlation_train = [0,0,0,0,0,0]
Norerror_train = [0,0,0,0,0,0]
Norerrortemp = []
# predictedtrainAll = [0,0,0,0,0,0]
for indx in range(0,len(train_data)):
    tempDimensiontrain = np.expand_dims(train_data[indx,:,:], axis=0) 
    predictedtrain = autoencoder.predict(tempDimensiontrain)
    predictedtrain = predictedtrain.reshape(200,6)
    # predictedtrainAll = np.vstack((predictedtrainAll,predictedtrain))
    # RMSE = rmse(predictedExternal[:,:],extern_data[indx,:,:])
    for indx2 in range(0,6):
        errortemp.append(rmse(predictedtrain[:,indx2],train_data[indx,:,indx2]))
        Norerrortemp.append(nrmse(predictedtrain[:,indx2],train_data[indx,:,indx2]))
        correlationtemp.append(np.corrcoef((predictedtrain[:,indx2],train_data[indx,:,indx2]))[0,1])
    error_train = np.vstack((error_train,errortemp))
    Norerror_train = np.vstack((Norerror_train,Norerrortemp))
    correlation_train = np.vstack((correlation_train,correlationtemp))
    errortemp = []
    correlationtemp = []
    Norerrortemp = []

error_train = np.delete(error_train, (0), axis=0)
Norerror_train = np.delete(Norerror_train, (0), axis=0)
correlation_train = np.delete(correlation_train, (0), axis=0)
trainMeanRMSE = np.mean(error_train,axis=0)

boxplottraindata_RMSE = [error_train[:,0],error_train[:,1],error_train[:,2],error_train[:,3],error_train[:,4],error_train[:,5]]
boxplottraindata_nRMSE = [Norerror_train[:,0],Norerror_train[:,1],Norerror_train[:,2],Norerror_train[:,3],Norerror_train[:,4],Norerror_train[:,5]]


################################
### Repeat for test data ###
################################

error_test = [0,0,0,0,0,0]
errortemp = []
correlationtemp = []
correlation_test = [0,0,0,0,0,0]
Norerror_test = [0,0,0,0,0,0]
Norerrortemp = []
for indx in range(0,len(test_data)):
    tempDimensiontest = np.expand_dims(test_data[indx,:,:], axis=0) 
    predictedtest = autoencoder.predict(tempDimensiontest)
    predictedtest = predictedtest.reshape(200,6)
    # RMSE = rmse(predictedExternal[:,:],extern_data[indx,:,:])
    for indx2 in range(0,6):
        errortemp.append(rmse(predictedtest[:,indx2],test_data[indx,:,indx2]))
        Norerrortemp.append(nrmse(predictedtest[:,indx2],test_data[indx,:,indx2]))
        correlationtemp.append(np.corrcoef((predictedtest[:,indx2],test_data[indx,:,indx2]))[0,1])
    error_test = np.vstack((error_test,errortemp))
    correlation_test = np.vstack((correlation_test,correlationtemp))
    Norerror_test = np.vstack((Norerror_test,Norerrortemp))
    errortemp = []
    correlationtemp = []
    errortemp = []
    Norerrortemp = []

error_test = np.delete(error_test, (0), axis=0)
testMeanRMSE = np.mean(error_test,axis=0)
Norerror_test = np.delete(Norerror_test, (0), axis=0)


boxplottestdata_RMSE = [error_test[:,0],error_test[:,1],error_test[:,2],error_test[:,3],error_test[:,4],error_test[:,5]]
boxplottestdata_nRMSE = [Norerror_test[:,0],Norerror_test[:,1],Norerror_test[:,2],Norerror_test[:,3],Norerror_test[:,4],Norerror_test[:,5]]


boxplottraindata = [correlation_train[:,0],correlation_train[:,1],correlation_train[:,2],correlation_train[:,3],correlation_train[:,4],correlation_train[:,5]]
boxplottestdata = [correlation_test[:,0],correlation_test[:,1],correlation_test[:,2],correlation_test[:,3],correlation_test[:,4],correlation_test[:,5]]

##### START THE PLOTTING ###############################

# Correlation #

fig = plt.figure(24)
ax1 = fig.add_subplot(211)
ax1.title.set_text('Correlation boxplot: training dataset')

bplot1 = ax1.boxplot(boxplottraindata,patch_artist=True,showfliers=False)

# ax7.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
ax2 = fig.add_subplot(212)
ax2.title.set_text('Correlation boxplot: testing dataset')
bplot2 = ax2.boxplot(boxplottestdata,patch_artist=True,showfliers=False)
# ax8.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
colors = ['pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)
plt.show()




# RMSE + NRMSE #

ylim = 60
ymin =0

fig = plt.figure(18)
ax1 = fig.add_subplot(221)
ax1.title.set_text('RMSE boxplot: training dataset')

bplot1 = ax1.boxplot(boxplottraindata_RMSE,patch_artist=True,showfliers=False)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.ylabel('degrees')
ax2 = fig.add_subplot(222)
ax2.title.set_text('Normalized RMSE boxplot: training dataset')
bplot2 = ax2.boxplot(boxplottraindata_nRMSE,patch_artist=True,showfliers=False)
# ax8.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
colors = ['pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('percentage')

ax3 = fig.add_subplot(223)
ax3.title.set_text('RMSE boxplot: test dataset')
bplot3 = ax3.boxplot(boxplottestdata_RMSE,patch_artist=True,showfliers=False)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.ylabel('degrees')

ax4 = fig.add_subplot(224)
ax4.title.set_text('Normalized RMSE boxplot: training dataset')
bplot4 = ax4.boxplot(boxplottestdata_nRMSE,patch_artist=True,showfliers=False)
# ax8.set_xlabel(['Left knee','left hip','left ankle','right knee','right hip','right ankle'])
colors = ['pink', 'lightblue', 'lightgreen','pink', 'lightblue', 'lightgreen']
for bplot in (bplot1, bplot2,bplot3,bplot4):
    for patch, color in zip(bplot['boxes'], colors):
        patch.set_facecolor(color)

plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('percentage')

plt.show()




### plot 1 subplt


fig = plt.figure(20)
ax1 = fig.add_subplot(121)
ax1.title.set_text('RMSE external validation')
# ax1.bar([0,1,2,3,4,5],externMeanRMSE)
# plt.ylim(top=ylim) #ymax is your value
# plt.ylim(bottom=ymin) #ymin is your value
# plt.xlabel('Column number')
# plt.ylabel('degrees')

ax2 = fig.add_subplot(121)
ax2.title.set_text('RMSE training validation')
ax2.bar([0,1,2,3,4,5],trainMeanRMSE)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('degrees')

ax3 = fig.add_subplot(122)
ax3.title.set_text('RMSE test validation')
ax3.bar([0,1,2,3,4,5],testMeanRMSE)
plt.ylim(top=ylim) #ymax is your value
plt.ylim(bottom=ymin) #ymin is your value
plt.xlabel('Column number')
plt.ylabel('degrees')
## put on discord    

   
