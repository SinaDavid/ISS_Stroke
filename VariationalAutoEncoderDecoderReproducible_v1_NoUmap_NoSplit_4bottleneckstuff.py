# -*- coding: utf-8 -*-
"""
Created on Tue Aug 31 09:05:18 2021

Authors: Michiel Punt and Sina Davids

Some info about the data set, see also e-mail Sina David 30-08-2021

‘LASI'(0-2)
'RASI'(3-5)
'LPSI'(6-8)
'RPSI'(9-11)
'LANK'(12-14)
'RANK' (15-17)
'LHEE'(18-20)
'RHEE'(21-23)

Version 0.1 
- creating a encoder / decoder neural network to analyse steady state data. 
# normalized was 20hz. 
# Stroke 100hz   
# healthy = 150 hz --> need to make consistent. 


# important findings. The new dataset for 10seconds data can be learned.

# The normalisation is of paramount importance. So far it seems that no normalisation is the best option....(zo weinig mogelijk massage aan de data)

### Decisions made ###

- only two latent variables in bottleneck.
- types of validation (group discrimination, smoothness, loss function)


### Done ###
 - leave 4 patients per group out for external validation. 
 - work with angles instead.! 
 - will fix the random seed. 
 - Kmean / density based 
 - can be raw latent features as well
 - Create an other version. and new data to split into windows of 50% overlap(squeeze the most out of the data).
 - Test whether the 5 gait cycles work or the absolute length i.e. 3 / 4 / 5 seconds or maybe 10 seconds length.
 - make an attempt only using encoder / decoder (no umap) umap tries to find clusters.
 - make different rescriptive statistics (range and std) i think are the most important one (compare the heatmap of these between umap and without)
- Make it workable for Sina again.DONE
- create weight/bias files for 2,4,5,6 bottleneck features! DONE
- create final models.Done
- create validation procedure using decoder and actual done
- create additional subgroup see whataps Sina. done
  - in the loop by 288 done
- train final model and store weights for later use. DONe  
- make plots look good. Done  
-  loss function plot
-  save as a svg
-  push to github Doe  
- exclude two stroke and two healthy. Done
- perform a validation RMSE compare with train rmse Done  
  
to do:


- get RMS loss and kl loss separately. 
- create 4 data sets around the center.
-  


Tomorrow:

    
- validation steps:
    1) loss and testing loss fucntion. 
    2) group validation. 
    3) Characteristics of the signal.
    4) RMSE training and testing.
    5) exploring the latent space. 

  5a original signal through encoder / decoder. 
  5b take latent features and add 2 points. 
  5c compare the decoded signal and created signal. 
  5d visualize the difference. 


@author: michi
"""
# from keras import backend as K
# K.clear_session()

# # Apparently you may use different seed values at each stage
seed_value= 1
# # 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
# 
# # 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)
# # 3. Set the `numpy` pseudo-random generator at a fixed value
import numpy as np
np.random.seed(seed_value)

# # 4. Set the `tensorflow` pseudo-random generator at a fixed value
import tensorflow as tf
# tf.random.set_seed(seed_value)
# # for later versions: 
tf.compat.v1.set_random_seed(seed_value)

# # 5. Configure a new global `tensorflow` session
# # from keras import backend as K
# # session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# # sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# # K.set_session(sess)
# # for later versions:
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
from matplotlib import mlab as ML
# import numpy as np
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
from keras.models import load_model
from keras import regularizers
from matplotlib.lines import Line2D
from sklearn.model_selection import (TimeSeriesSplit, KFold, ShuffleSplit,LeaveOneGroupOut,
                                      StratifiedKFold, GroupShuffleSplit,
                                      GroupKFold, StratifiedShuffleSplit)
import pickle
import pandas as pd
import seaborn as sns
plt.close('all')

# os.environ['PYTHONHASHSEED']=str(seed_value)


def rmse(predictions, targets):
    return np.sqrt(((predictions - targets) ** 2).mean())



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


def fourierTransforming(signal, lineartaper = None, sampleFreq=20):
        signal = (signal - signal.mean()) / signal.std()
        signalLength = len(signal)                

        # Apply linear taper of %5 at each end
        if lineartaper :
            int(signalLength * 0.05 )
            signal[0:int(signalLength * 0.05 )] = (signal[0:int(signalLength * 0.05 )] / 20)
            signal[signalLength - int(signalLength * 0.05): signalLength ] = signal[signalLength - int(signalLength * 0.05): signalLength ] / 20
            
        # add x amount of zeros so that the signal is 2^x
        check = True
        zeroscounter = 512
        while (check == True):
            if (signalLength > zeroscounter):
                 zeroscounter = 2 * zeroscounter
            else:
                tot = zeroscounter - signalLength
                check = False
        signal = np.concatenate([signal, np.zeros(tot)]) 
        ncor = len(signal)
        
        # Creates the necessairy frequenciees
        fftsignal           = np.fft.fft(signal)
        frequencies           = np.fft.fftfreq(ncor, sampleFreq)
        power               = (abs(fftsignal) / ncor)
        # mask                = frequencies > 0
        return power



################# Settings over here! :-) #####################
plt.close('all')
inputTimeSerie = 'markervel'

path = "F:\\SSI_data\\finalData\\Mixed\\Mixed\\no_Split\\No_Split\\prox_relativeangles\\"
pathEvents = "F:\\SSI_data\\finalData\\Mixed\\Mixed\\no_Split\\No_Split\\Events\\"
pathToWeightsavingANDextraction = "C:\\Users\\michi\\Desktop\\SSI_Stroke\\"




# inputColumns = [18,19,20,21,22,23]
inputColumns = [0, 3, 6, 9, 12, 15]
windowLength = 200 # 5 gaitcyclus, normalized to 1000 and downsampled to 200
N_bottleneckFeatures  = 4
windowLength
fs = 1/0.05 ### 20Hz downsampled
fmax = fs/2
freqResolutie = fmax/windowLength 
usingExistingModel = True
savingModel = False
loadingInitializedWeights = False
storingTrainedWeights = False  # Ook al train je indien deze op false worden nieuwe weights niet opgeslagen

pathEncoderModel = pathToWeightsavingANDextraction + "EncoderModel" + str(N_bottleneckFeatures)
pathDecoderModel = pathToWeightsavingANDextraction + "DecoderModel" + str(N_bottleneckFeatures)
pathAutoEncoderModel = pathToWeightsavingANDextraction + "autoEncoderModel" + str(N_bottleneckFeatures)

################### 2 Loading the data ###########################  

filename = []
data = []
group = []
perturbationType = []

#### THE TIMESERIES #####
if 'path' in locals():
    for file1 in os.listdir(path):
        data.append(spio.loadmat(path + file1))
        filename.append(file1)
        if file1[0]=='W':
            group.append(int(file1[1:3]))
        else:
            group.append(int(file1[1:4]))
    #     perturbationType.append(numberPer)
    # numberPer +=1
group = np.array(group)
# get the dict out of the list
for numtrials in range(len(filename)):
    data[numtrials] = (data[numtrials]['prox_relativeangles'])#'markervel'

# Next resample / remove 3th element of w files to resample towards 100hz
for indx in range(108,len(data)): # LET OP DIT IS EEN HARDCODED 108
    delArray = np.arange(0,len(data[indx]),3)
    data[indx]= np.delete(data[indx], delArray, axis=0)

## Nu is alles 100Hz!




###### THE GAIT EVENTS ############
groupE = []
dataE = []
filenameE = []
if 'pathEvents' in locals():
    for file1E in os.listdir(pathEvents):
        dataE.append(spio.loadmat(pathEvents + file1E))
        filenameE.append(file1E)
        if file1E[0]=='W':
            groupE.append(int(file1E[1:3]))
        else:
            groupE.append(int(file1E[1:4]))
    #     perturbationType.append(numberPer)
    # numberPer +=1
groupE = np.array(groupE)
# get the dict out of the list
for numtrials in range(len(filenameE)):
    dataE[numtrials] = (dataE[numtrials]['Events'])#'markervel'


### Nu de events naar 100Hz ###

for indx in range(108,len(dataE)):
    dataE[indx] = np.round(dataE[indx]*0.6666,0)

#### Nu alles naar 100hz qua events. Next alles naar 20Hz #####

for indx in range(len(dataE)):
    dataE[indx] = np.round(dataE[indx]*0.2,0)

### Gedaan voor de events!####
for indx in range(len(data)):
    data[indx] = data[indx][::5]
### Gedaan voor de data (in stapjes van 5 ::5)#####
  
#######################################################################

'''
álles is hier nu gereed om 

data en dataE te combineren data augmentation. per group! per trial!

'''
count = 0
trackGroup = []
trackGroup1 = []
dataAugmented = np.zeros(len(inputColumns))
for indx in range(len(data)):
    if len(data[indx])>200:
        print(indx)
        for indx2 in range(0,len(dataE[indx]),2):
            index = dataE[indx][indx2]
            tempArray = data[indx][int(index[0]):]
            if len(tempArray)>200:
                count +=1
                dataAugmented = np.vstack((dataAugmented,tempArray[0:200,inputColumns]))
                trackGroup.append(indx)
                if indx < 104:
                    trackGroup1.append('SS'+str(group[indx]))
                else:
                    trackGroup1.append('W'+str(group[indx]))
            

    

########################################################################
# ### Always remove first row (due to initialize issues)       
dataAugmented = np.delete(dataAugmented, (0), axis=0) 

##### Get the dependent y data from filenames. #######
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
y = np.delete(y,(0),axis=0)
# group = np.array(group)
########    Reshape  the X data ############
dataAugmented = dataAugmented.reshape(len(y),200,len(inputColumns))

'''
# Assigning two subjects from stroke + 2 from healthy for external validation.
SS1 & SS38  & W1 & W38
# split test and training on subject level. 

Also remove y indexen from excluded data
'''
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

# Make a int list of subjects (zucht wat een werk...)
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





########### Split in train and test set ################
# train_data = dataAugmented[np.arange(start=2,stop=np.shape(dataAugmented)[0],step=2)]
# test_data = dataAugmented[np.arange(start=1,stop=np.shape(dataAugmented)[0],step=2)]
# train_data_y = y[np.arange(start=2,stop=np.shape(y)[0],step=2)]
# test_data_y = y[np.arange(start=1,stop=np.shape(y)[0],step=2)]
# train_data_group = group[np.arange(start=2,stop=np.shape(group)[0],step=2)]
# test_data_group = group[np.arange(start=1,stop=np.shape(group)[0],step=2)]


##############################################################################
############################### Encoder / decoder part #######################
##############################################################################

tf.compat.v1.disable_eager_execution()
input_data = tf.keras.layers.Input(shape=(200, 6))
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
distribution_mean = tf.keras.layers.Dense(N_bottleneckFeatures, name='mean')(encoder)
distribution_variance = tf.keras.layers.Dense(N_bottleneckFeatures, name='log_variance')(encoder)
latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])
encoder_model = tf.keras.Model(input_data, latent_encoding)
encoder_model.summary()
################### DECODER PART ############
decoder_input = tf.keras.layers.Input(shape=(N_bottleneckFeatures)) 
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
############### iNITIALIZE ########################################
decoderWeightstesting = []
encoderWeightstesting = []
############# RETRIEVING WEIGHTS AND BIASES FROM INITIALZED MODEL ############
if loadingInitializedWeights == False:
    
    for indx in range(0,len(decoder_model.layers)):
        decoderWeightstesting.append(decoder_model.layers[indx].get_weights())
    
    for indx in range(0,len(encoder_model.layers)):
        encoderWeightstesting.append(encoder_model.layers[indx].get_weights())
    
    
    ############ STORING RETRIEVED WEIGHTS AND BIASES FROM INITIALIZED MODEL ###########
    
    f = open(pathToWeightsavingANDextraction + 'decoderInitialWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'wb')
    pickle.dump(decoderWeightstesting, f)
    f.close()
    
    f = open(pathToWeightsavingANDextraction + 'encoderInitialWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'wb')
    pickle.dump(encoderWeightstesting, f)
    f.close()

################### LOADING STORED WEIGHTS AND BIASES ########################################

elif loadingInitializedWeights == True:

    f = open(pathToWeightsavingANDextraction + 'decoderInitialWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'rb')
    decoderWeightstesting = pickle.load(f)
    f.close()
    
    f = open(pathToWeightsavingANDextraction + 'encoderInitialWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'rb')
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





##################################### EIND OF THIS PART #########################

############## FINAL CONFIGURATION AND COMPILE MODEL ############################
encoded = encoder_model(input_data)
decoded = decoder_model(encoded)
autoencoder = tf.keras.models.Model(input_data, decoded)
autoencoder.compile(loss=get_loss(distribution_mean, distribution_variance), optimizer='adam')
print("\nautoenoder summary")
autoencoder.summary()
 
if usingExistingModel  == False:    
    # history = autoencoder.fit(train_data, train_data, epochs=200, batch_size=64, validation_data=(test_data, test_data))
    history = autoencoder.fit(train_data,
                              train_data,
                              epochs=400,
                              batch_size=64,
                              callbacks=[EarlyStopping(monitor='loss', patience=40)],
                              validation_data=(test_data, test_data))
    
    plt.plot(history.history['val_loss'])
    plt.plot(history.history['loss'])
    #### storing train weights and biases ####
    decoderWeightstrained = []
    encoderWeightstrained = []

    for indx in range(0,len(decoder_model.layers)):
        decoderWeightstrained.append(decoder_model.layers[indx].get_weights())
    
    for indx in range(0,len(encoder_model.layers)):
        encoderWeightstrained.append(encoder_model.layers[indx].get_weights())
        
    if storingTrainedWeights ==True:
        f = open(pathToWeightsavingANDextraction + 'decoderTRAINEDWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'wb')
        pickle.dump(decoderWeightstrained, f)
        f.close()
        
        f = open(pathToWeightsavingANDextraction + 'encoderTRAINEDWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'wb')
        pickle.dump(encoderWeightstrained, f)
        f.close()
    
    
    
elif usingExistingModel==True:
    decoderWeightstrained = []
    encoderWeightstrained = []
    # autoencoder = load_model(pathAutoEncoderModel)
    # encoder_model = load_model(pathEncoderModel)
    # decoder_model = load_model(pathDecoderModel)
    # load the existingModel
    print('loaded weights and biases of the trained encoder/decodermodel ....')
    f = open(pathToWeightsavingANDextraction + 'decoderTRAINEDWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'rb')
    decoderWeightstrained = pickle.load(f)
    f.close()
    
    f = open(pathToWeightsavingANDextraction + 'encoderTRAINEDWeightsBottleneck'+ str(N_bottleneckFeatures) + '.pckl', 'rb')
    encoderWeightstrained = pickle.load(f)
    f.close()
    ######################  INSERTING WEIGHTS AND BIASES IN ENCODER & DECODER MODEL ################################ 
    for indx in range(0,len(decoder_model.layers)):
        a = decoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            decoder_model.layers[indx].set_weights([decoderWeightstrained[indx][0],decoderWeightstrained[indx][1]])
    
    for indx in range(0,len(encoder_model.layers)):
        a = encoder_model.layers[indx].get_weights()
        if not a:
            print('layer is empty')
        else:
            encoder_model.layers[indx].set_weights([encoderWeightstrained[indx][0],encoderWeightstrained[indx][1]])  
    print('the trained weights and biases are inserted in the model')

##################### Next visualize the decoded data against original data.   ####################
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
    




### Calculate for orginal signal SD(per column) + as well as for the decoder.
SD_org = np.transpose(np.zeros(6))
SD_dec = np.zeros(6) 
freq_max = np.zeros(6)
for i in range(0,len(test_data)):
    test_new = test_data[i]
    test_new_dec = np.expand_dims(test_new, axis=0)
    test_dec = autoencoder.predict(test_new_dec)[0]
    test_new = np.expand_dims(test_new, axis=0)[0] 
    # calculate orginal SD_org
    sd_temp = []#0,0,0,0,0,0
    sd_temp_Dec = []
    freq_temp_max = []
    for ii in range(0,6):
        sd_temp = np.hstack((sd_temp, np.std(test_new[0:200,ii])))
        sd_temp_Dec = np.hstack((sd_temp_Dec, np.std(test_dec[0:200,ii])))
        transform = fourierTransforming(test_dec[0:200,ii])
        freq_temp_max = np.hstack((freq_temp_max,int(np.where(transform[0:256]== np.max(transform[0:256]))[0])))
    SD_org = np.vstack((SD_org,sd_temp))
    SD_dec = np.vstack((SD_dec,sd_temp_Dec))
    freq_max = np.vstack((freq_max,freq_temp_max))

# fig = plt.figure(3)
# for indx in range(0,6):
#     plt.scatter(SD_org[:,indx],SD_dec[:,indx])





# Next get the bottleneck features visualized for test data
encoded = []
for i in range(0,len(test_data)):
    # z.append(testy[i])
    test_new = test_data[i]
    test_new = np.expand_dims(test_new, axis=0)
    op = encoder_model.predict(test_new)
    encoded.append(op[0])

plt.figure(5)
plt.plot(np.array(encoded))


###### Newly created data: #######
# createdData = []
# gemidFeatures = np.mean(np.array(encoded),axis=0)
# sdFeatures = np.mean(np.array(encoded),axis=0)
# dataTemp = []

# for indx in range(0,4):
#     inputCreated = gemidFeatures
#     inputCreated[indx]= gemidFeatures[indx]+sdFeatures[indx]*2
#     inputCreated = np.expand_dims(inputCreated, axis=0)
#     print(inputCreated)
#     dataTemp = decoder_model.predict(inputCreated).reshape((200,6))
#     createdData.append(dataTemp)






################### SAVING MODEL #######################

if savingModel == True:
    print('saving the models...')
    autoencoder.save(pathAutoEncoderModel)
    decoder_model.save(pathDecoderModel)
    encoder_model.save(pathEncoderModel)
    #save.model()autoencoder (entire model), decoder_model, encoder_model





##################################
# vanaf hier maakt N bottleneck features uit dus anders plotten ##
#################################

if N_bottleneckFeatures ==2:

    xx = []
    yy = []
    z = []
    groupcolor = []
    for i in range(0,len(np.array(encoded))):
        # z.append(testy[i])
        xx.append(np.array(encoded)[i][0])
        yy.append(np.array(encoded)[i][1])
        # xx.append(op[0][0])
        # yy.append(op[0][1])
        z.append(test_y[i]) # Fall risk / group
    
    
    xx = np.array(xx)
    yy = np.array(yy)
    df1 = pd.DataFrame()
    df1['xx'] = xx
    df1['yy'] = yy
    df1['z'] = ["fall risk-"+str(k) for k in z]
    df1['groupcolor'] = ["subject-"+str(k) for k in test_data]
    
    
    fig4, axes = plt.subplots(2, 2, figsize=(15, 5))#, sharey=True
    
    sns.scatterplot(ax=axes[0,0],x='xx', y='yy',hue='z', data=df1)
    axes[0,0].set_title('Raw latent features per perturbation colored by fall risk')
    axes[0,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)
    
    
    
    
    
    fig1,axs = plt.subplots()
    sns.scatterplot(ax=axs,x='xx', y='yy',hue='z', data=df1)
    
    
    
    # cb = plt.colorbar()
    # cb.set_label('mean value')
    # plt.show()   
    
    df1.loc[df1['z'] == 'fall risk-[1]', 'z'] = 'orange'
    df1.loc[df1['z'] == 'fall risk-[0]', 'z'] = 'blue'
    df1.loc[df1['z'] == 'fall risk-[2]', 'z'] = 'green'
    df1.loc[df1['z'] == 'fall risk-[3]', 'z'] = 'lightgreen'
    ################# Thursday 21/10/2021  ################
    


    fig = plt.figure(10)
    plt.scatter(df1['xx'], df1['yy'], c=df1['z'],label=df1['z'],edgecolors='white')
    plt.title('2D Variational autoencoder / decoder')
    plt.xlabel('Latent feature 1')
    plt.ylabel('Latent feature 2')
    colors = ['orange', 'blue', 'green','lightgreen']
    lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
    labels = ['stroke survivors with falls', 'stroke survivors without falls', 'older adults','younger adults']
    plt.legend(lines, labels)
    plt.show()
    
    encoded1 = [0,0]
    
    for indx in range(len(encoded)):
        encoded_temp = encoded[indx]
        encoded1 = np.vstack((encoded1,np.transpose(encoded_temp)))
    encoded1 = np.delete(encoded1,(0),axis=0)
    SD_dec = np.delete(SD_dec,(0),axis=0)
    freq_max = np.delete(freq_max,(0),axis=0)
    freq_max = freq_max * freqResolutie
    
    fig = plt.figure(11)
    ax1 = fig.add_subplot(321)
    ax1.title.set_text('std 1')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=SD_dec[:,0], cmap=CM.jet, bins=None)
    plt.colorbar()

    ax2 = fig.add_subplot(322)
    ax2.title.set_text('std 2')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=SD_dec[:,1], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax3 = fig.add_subplot(323)
    ax3.title.set_text('std3 ')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=SD_dec[:,2], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax4 = fig.add_subplot(324)
    ax4.title.set_text('std 4')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=SD_dec[:,3], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax5 = fig.add_subplot(325)
    ax5.title.set_text('std5 ')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=SD_dec[:,4], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax6 = fig.add_subplot(326)
    ax6.title.set_text('std 6')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=SD_dec[:,5], cmap=CM.jet, bins=None)
    plt.colorbar()
    plt.suptitle('Standard deviation per signal',fontsize=20)


    fig = plt.figure(12)
    ax1 = fig.add_subplot(321)
    ax1.title.set_text('Dominant frequency signal 1')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=freq_max[:,0], cmap=CM.jet, bins=None)
    plt.colorbar()

    ax2 = fig.add_subplot(322)
    ax2.title.set_text('Dominant frequency signal 2')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=freq_max[:,1], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax3 = fig.add_subplot(323)
    ax3.title.set_text('Dominant frequency signal 3 ')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=freq_max[:,2], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax4 = fig.add_subplot(324)
    ax4.title.set_text('Dominant frequency signal 4')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=freq_max[:,3], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax5 = fig.add_subplot(325)
    ax5.title.set_text('Dominant frequency signal 5 ')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=freq_max[:,4], cmap=CM.jet, bins=None)
    plt.colorbar()
    
    ax6 = fig.add_subplot(326)
    ax6.title.set_text('Dominant frequency signal 6')
    plt.hexbin(encoded1[:,0], encoded1[:,1], C=freq_max[:,5], cmap=CM.jet, bins=None)
    plt.colorbar()
    plt.suptitle('Dominant frequency per signal',fontsize=20)




################ STORING THE DATA ############################
if usingExistingModel == False:
    np.save('lossfunction_validation' + str(N_bottleneckFeatures) +'.npy',history.history['val_loss'])
    np.save('lossfunction_' + str(N_bottleneckFeatures) +'.npy',history.history['loss'])
# np.save('orginalData' + str(N_bottleneckFeatures) +'.npy',originalData)
# np.save('reconstructed' + str(N_bottleneckFeatures) +'.npy',Reconstructed)
# np.save('createdData' + str(N_bottleneckFeatures) +'.npy',createdData)

# np.save('LatentFeatureValues' + str(N_bottleneckFeatures) +'.npy',inputCreated)
#################################################################

# loss = np.load('lossfunction.npy')
# plt.figure()
# plt.plot(loss)
# plt.title('Validation loss',fontsize=20)
# plt.xlabel('Number of training Epochs',fontsize=16)
# plt.ylabel('Reconstructions loss & KL loss',fontsize=16)






#### Additional validation ######

################################
### Repeat for training data ###
################################

error_train = [0,0,0,0,0,0]
errortemp = []
for indx in range(0,len(train_data)):
    tempDimensiontrain = np.expand_dims(train_data[indx,:,:], axis=0) 
    predictedtrain = autoencoder.predict(tempDimensiontrain)
    predictedtrain = predictedtrain.reshape(200,6)
    # RMSE = rmse(predictedExternal[:,:],extern_data[indx,:,:])
    for indx2 in range(0,6):
        errortemp.append(rmse(predictedtrain[:,indx2],train_data[indx,:,indx2]))
    error_train = np.vstack((error_train,errortemp))
    errortemp = []

error_train = np.delete(error_train, (0), axis=0)
trainMeanRMSE = np.mean(error_train,axis=0)

################################
### Repeat for test data ###
################################

error_test = [0,0,0,0,0,0]
errortemp = []
for indx in range(0,len(test_data)):
    tempDimensiontest = np.expand_dims(test_data[indx,:,:], axis=0) 
    predictedtest = autoencoder.predict(tempDimensiontest)
    predictedtest = predictedtest.reshape(200,6)
    # RMSE = rmse(predictedExternal[:,:],extern_data[indx,:,:])
    for indx2 in range(0,6):
        errortemp.append(rmse(predictedtest[:,indx2],test_data[indx,:,indx2]))
    error_test = np.vstack((error_test,errortemp))
    errortemp = []

error_test = np.delete(error_test, (0), axis=0)
testMeanRMSE = np.mean(error_test,axis=0)


### plot 1 subplt
ylim = 30
ymin =0

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



####################### Step 5 of the validation #############################

createdData = np.zeros(N_bottleneckFeatures)
for indx in range(len(encoded)): 
    createdData = np.vstack((createdData,encoded[indx]))

createdData = np.delete(createdData, (0), axis=0)




decoderTimeSeries = decoder_model.predict(np.expand_dims(encoded[0], axis=0)).reshape(200,6)
createdTimeSeriesPos = decoder_model.predict(np.expand_dims(createdData[0]+2, axis=0)).reshape(200,6)
createdTimeSeriesNeg = decoder_model.predict(np.expand_dims(createdData[0]-2, axis=0)).reshape(200,6)
fig = plt.figure(21)
ax1 = fig.add_subplot(231)
ax1.title.set_text('Decoded timeserie')
ax1.plot(decoderTimeSeries)

ax2 = fig.add_subplot(232)
ax2.title.set_text('Created timeserie added 2')
ax2.plot(createdTimeSeriesPos)

ax3 = fig.add_subplot(233)
ax3.title.set_text('Created timeserie subtracted 2')
ax3.plot(createdTimeSeriesNeg)


decoderTimeSeries = decoder_model.predict(np.expand_dims(encoded[1000], axis=0)).reshape(200,6)
createdTimeSeriesPos = decoder_model.predict(np.expand_dims(createdData[1000]+2, axis=0)).reshape(200,6)
createdTimeSeriesNeg = decoder_model.predict(np.expand_dims(createdData[1000]-2, axis=0)).reshape(200,6)
fig = plt.figure(21)
ax1 = fig.add_subplot(234)
ax1.title.set_text('Decoded timeserie')
ax1.plot(decoderTimeSeries)

ax2 = fig.add_subplot(235)
ax2.title.set_text('Created timeserie added 2')
ax2.plot(createdTimeSeriesPos)

ax3 = fig.add_subplot(236)
ax3.title.set_text('Created timeserie subtracted 2')
ax3.plot(createdTimeSeriesNeg)



loss = np.load('C:\\Users\\michi\\Desktop\\SSI_Stroke\\lossfunction_4.npy')
lossvalidation = np.load('C:\\Users\\michi\\Desktop\\SSI_Stroke\\lossfunction_validation4.npy')


fig = plt.figure(22)
ax1 = fig.add_subplot(111)
# ax1.title.set_text()
# ax1.plot(loss)
# ax1.plot(lossvalidation)
# ax1.legend('Training loss', 'Testing loss')
# use a original point. create a new on.
# determine timeseries. 
# plot original and created. 
# determine nearest neighbour

line1, = ax1.plot(loss, label='Training loss')
line2, = ax1.plot(lossvalidation, label='Testing loss')
ax1.legend(handles=[line1, line2])
plt.title('Training and testing loss',fontsize=20)
plt.xlabel('Number of training Epochs',fontsize=16)
plt.ylabel('Reconstructions loss & KL loss',fontsize=16)




latent1 = []
latent2 = []
latent3 = []
latent4 = []
z = []
groupcolor = []
for i in range(0,len(np.array(encoded))):
    # z.append(testy[i])
    latent1.append(np.array(encoded)[i][0])
    latent2.append(np.array(encoded)[i][1])
    latent3.append(np.array(encoded)[i][2])
    latent4.append(np.array(encoded)[i][3])
    z.append(test_y[i]) # Fall risk / group


latent1 = np.array(latent1)
latent2 = np.array(latent2)
latent3 = np.array(latent3)
latent4 = np.array(latent4)
df1 = pd.DataFrame()
df1['Latent feature 1'] = latent1
df1['Latent feature 2'] = latent2
df1['Latent feature 3'] = latent3
df1['Latent feature 4'] = latent4
df1['z'] = ["fall risk-"+str(k) for k in z]
df1['groupcolor'] = ["subject-"+str(k) for k in test_data]


# fig4, axes = plt.subplots(2, 2, figsize=(15, 5))#, sharey=True

# sns.scatterplot(ax=axes[0,0],x='xx', y='yy',hue='z', data=df1)
# axes[0,0].set_title('Raw latent features per perturbation colored by fall risk')
# axes[0,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)






# cb = plt.colorbar()
# cb.set_label('mean value')
# plt.show()   

df1.loc[df1['z'] == 'fall risk-[1]', 'z'] = 'orange'
df1.loc[df1['z'] == 'fall risk-[0]', 'z'] = 'blue'
df1.loc[df1['z'] == 'fall risk-[2]', 'z'] = 'green'
df1.loc[df1['z'] == 'fall risk-[3]', 'z'] = 'lightgreen'



sns.pairplot(df1, hue="z")



fig = plt.figure(10)
plt.scatter(df1['xx'], df1['yy'], c=df1['z'],label=df1['z'],edgecolors='white')
plt.title('2D Variational autoencoder / decoder')
plt.xlabel('Latent feature 1')
plt.ylabel('Latent feature 2')
colors = ['orange', 'blue', 'green','lightgreen']
lines = [Line2D([0], [0], color=c, linewidth=3, linestyle='-') for c in colors]
labels = ['stroke survivors with falls', 'stroke survivors without falls', 'older adults','younger adults']
plt.legend(lines, labels)
plt.show()





















































# elif N_bottleneckFeatures ==3:
#     print('hoi 3d')
#     xx = []
#     yy = []
#     zz = []
#     z = []
#     groupcolor = []
#     for i in range(0,len(np.array(encoded))):
#         # z.append(testy[i])
#         xx.append(np.array(encoded)[i][0])
#         yy.append(np.array(encoded)[i][1])
#         zz.append(np.array(encoded)[i][2])
#         # xx.append(op[0][0])
#         # yy.append(op[0][1])
#         z.append(test_data_y[i]) # Fall risk / group
    
    
#     xx = np.array(xx)
#     yy = np.array(yy)
#     zz = np.array(zz)
#     df1 = pd.DataFrame()
#     df1['xx'] = xx
#     df1['yy'] = yy
#     df1['zz'] = zz
#     df1['z'] = ["fall risk-"+str(k) for k in z]
#     df1['groupcolor'] = ["subject-"+str(k) for k in test_data]
    
    
#     # fig4, axes = plt.subplots(2, 2, figsize=(15, 5))#, sharey=True
#     # sns.set(style = "darkgrid")
#     # fig = plt.figure()
#     # ax = fig.add_subplot(111, projection = '3d')
    
#     # # x = df['Happiness Score']
#     # # y = df['Economy (GDP per Capita)']
#     # # z = df['Health (Life Expectancy)']
    
#     # ax.set_xlabel("xx")
#     # ax.set_ylabel("yy")
#     # ax.set_zlabel("zz")
    
#     # ax.scatter(xs='xx', ys='yy',zs = 'zz',hue='z', data=df1)
    
#     # plt.show()
#     # plt.scatter(xx,yy,zz)
#     # plt.show()
#     import re, seaborn as sns, numpy as np, pandas as pd, random
#     from pylab import *
#     from matplotlib.pyplot import plot, show, draw, figure, cm
#     import matplotlib.pyplot as plt
#     from mpl_toolkits.mplot3d import Axes3D
    
#     color_dict = dict({'0':'blue',
#                   '1':'orange',
#                   '2': 'green'})
    
#     sns.set_style("whitegrid", {'axes.grid' : False})
    
#     fig = plt.figure(figsize=(6,6))
    
#     ax = Axes3D(fig) # Method 1
#     # ax = fig.add_subplot(111, projection='3d') # Method 2
    
#     # x = np.random.uniform(1,20,size=20)
#     # y = np.random.uniform(1,100,size=20)
#     # z = np.random.uniform(1,100,size=20)
    
    
#     ax.scatter(xx, yy, zz, c=z, marker='o')#,palette=color_dict
#     ax.set_xlabel('X Label')
#     ax.set_ylabel('Y Label')
#     ax.set_zlabel('Z Label')
    
#     plt.show()
    
    

#     # sns.scatterplot(ax=axes[0,0],)
#     # axes[0,0].set_title('Raw latent features per perturbation colored by fall risk')
#     # axes[0,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)
    
    
#     dummySignals = np.zeros((200,6))
#     dummyX = np.arange(int(np.min(xx)),int(np.max(xx)))
#     dummyY = np.arange(int(np.min(yy)),int(np.max(yy)))
#     test_pts0 = []
#     test_pts1 = []
#     for indx in range(len(dummyX)): # y values
#         for indx2 in range(0,len(dummyY)):
#             test_pts0 = np.append(test_pts0,dummyX[indx]) 
#             test_pts1 = np.append(test_pts1, dummyY[indx2])
#             dummyData = np.expand_dims([dummyX[indx],dummyY[indx2]], axis=0)
#             prediction = decoder_model.predict(dummyData)
#             predictiondim = prediction.reshape(200,6) #np.squeeze(prediction, axis=(2,))
#             dummySignals = np.vstack((dummySignals,predictiondim))
    
    
    
    
#     minimum1 = []
#     minimum4 = []
#     maximum1 = []
#     maximum4 = []
#     std1 = []
#     std4 = []
    
    
#     for indx in np.arange(200, len(dummySignals),200, dtype=None):
#         # print (indx)
#         minimum1 = np.append(minimum1, np.min(dummySignals[indx:indx+200,1]))
#         maximum1 = np.append(maximum1, np.max(dummySignals[indx:indx+200,1]))
#         minimum4 = np.append(minimum4, np.min(dummySignals[indx:indx+200,4]))
#         maximum4 = np.append(maximum4, np.max(dummySignals[indx:indx+200,4]))
#         std1 = np.append(std1, np.std(dummySignals[indx:indx+200,1]))
#         std4 = np.append(std4, np.std(dummySignals[indx:indx+200,4]))
    
    
#     ### Final features ####
#     range1 = maximum1 - minimum1
#     range4 = maximum4 - minimum4
    
    
#     fig1,axs = plt.subplots()
#     sns.scatterplot(ax=axs,x='xx', y='yy',hue='z', data=df1)
    
    
    
#     # cb = plt.colorbar()
#     # cb.set_label('mean value')
#     # plt.show()   
    
#     df1.loc[df1['z'] == 'fall risk-[1]', 'z'] = 'orange'
#     df1.loc[df1['z'] == 'fall risk-[0]', 'z'] = 'blue'
#     df1.loc[df1['z'] == 'fall risk-[2]', 'z'] = 'green'
#     ################# Thursday 21/10/2021  ################
#     fig = plt.figure(10)
#     # fig, axes = plt.subplots(2, 2)
    
#     fig = plt.figure(11)
#     ax1 = fig.add_subplot(221)
#     ax1.title.set_text('2d encoder /decoder')
#     plt.scatter(df1['xx'], df1['yy'], c=df1['z'])
#     # axes[1,0]
#     ax2 = fig.add_subplot(222)
#     ax2.title.set_text('Range 1')
#     plt.hexbin(test_pts0, test_pts1, C=range1, cmap=CM.jet, bins=None)
    
#     ax3 = fig.add_subplot(223)
#     ax3.title.set_text('std1 ')
#     plt.hexbin(test_pts0, test_pts1, C=std1, cmap=CM.jet, bins=None)
    
#     ax4 = fig.add_subplot(224)
#     ax4.title.set_text('std 4')
#     plt.hexbin(test_pts0, test_pts1, C=std4, cmap=CM.jet, bins=None)


