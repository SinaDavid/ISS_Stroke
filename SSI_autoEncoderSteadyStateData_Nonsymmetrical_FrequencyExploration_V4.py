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




# important findings. The new dataset for 10seconds data can be learned.

# The normalisation is of paramount importance. So far it seems that no normalisation is the best option....(zo weinig mogelijk massage aan de data)



### To Do list"
 - leave 4 patients per group out for external validation. 
 - work with angles instead.! 
 - will fix the random seed. 
 - Kmean / density based 
 - can be raw latent features as well




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
# import os
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

from keras import backend as K
K.clear_session()
# import tensorflow
# from numpy.random import seed
# seed(0)
# # from tensorflow import set_random_seed
# # set_random_seed(2)
# import tensorflow
# tensorflow.random.set_seed(2)
# import tensorflow as tf


# Seed value
# Apparently you may use different seed values at each stage
seed_value= 1

# 1. Set the `PYTHONHASHSEED` environment variable at a fixed value
import os
os.environ['PYTHONHASHSEED']=str(seed_value)

# 2. Set the `python` built-in pseudo-random generator at a fixed value
import random
random.seed(seed_value)

# 3. Set the `numpy` pseudo-random generator at a fixed value
# import numpy as np
np.random.seed(seed_value)

# 4. Set the `tensorflow` pseudo-random generator at a fixed value
# import tensorflow as tf
tf.random.set_seed(seed_value)
# for later versions: 
# tf.compat.v1.set_random_seed(seed_value)

# 5. Configure a new global `tensorflow` session
# from keras import backend as K
# session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
# sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
# K.set_session(sess)
# for later versions:
session_conf = tf.compat.v1.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
sess = tf.compat.v1.Session(graph=tf.compat.v1.get_default_graph(), config=session_conf)
# tf.compat.v1.keras.backend.set_session(sess)



# from keras import backend as K
# K.set_session(sess)







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
path = 'F:\\SSI_data\\steadystate_prefSpeed\\5GC_200Frames\\5GC_200Frames\\markervel\\'
path = "F:\\SSI_data\\steadystate_prefSpeed\\Data4Python1\\Data4Python1\\steadystate_prefSpeed\\10sec\\" + inputTimeSerie + '\\'
inputColumns = [18,19,20,21,22,23]
windowLength = 200 # 5 gaitcyclus, normalized to 1000 and downsampled to 200


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
group = np.array(group)
# get the dict out of the list
for numtrials in range(len(filename)):
    data[numtrials] = (data[numtrials]['markervel'])


# if inputTimeSerie == 'markervel':
X_data = np.zeros(len(inputColumns))#[0,0,0]
X_dataFreq = np.zeros(len(inputColumns))
for indx in range(len(data)):
    signalFreq = np.zeros(200)
    x = data[indx][0:200,inputColumns]   # 
    for i in range(0,5):
        signal = x[:,i]
        power = fourierTransforming(signal, lineartaper = None, sampleFreq=20)
        signalFreq = np.vstack((signalFreq,power[0:200]))
    X_data = np.vstack((X_data,x))     
    X_dataFreq = np.vstack((X_dataFreq,np.transpose(signalFreq)))       
       

# ### Always remove first row (due to initialize issues)       
X_data = np.delete(X_data, (0), axis=0) 
X_dataFreq = np.delete(X_dataFreq, (0), axis=0) 

##################################################################################################
###### Activate the line below to feed the network with the signal in the frequency domain. ######
#X_data = X_dataFreq
##################################################################################################



# To reduce data we resample with a factor 5. 
# X_data = X_data[::5]
# minimum = np.min(X_data)
# maximum = np.max(X_data)
# range1 = minimum - maximum
# X_data /= range1
# for indx in range(0,5):  
#     print(indx)
#     minimum = np.min(X_data[:,[indx,indx+3]])
#     maximum = np.max(X_data[:,[indx,indx+3]])
#     range1 = minimum - maximum
#     X_data[:,[indx,indx+3]] /= range1
# x_normed = X_data / X_data.max(axis=0)
# X_data = x_normed
# for indx in range(0,5):  
#     # print(indx)
#     minimum = np.min(X_data[:,indx])
#     maximum = np.max(X_data[:,indx])
#     range1 = minimum - maximum
#     X_data[:,indx] /= range1

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
# group = []
for indx in range(len(filename)):
    x = filename[indx].split('FR')
    y = np.vstack((y,int(x[1][0])))   
    # group.append(int(filename[1:4]))
y = np.delete(y,(0),axis=0)
# group = np.array(group)

########    Reshape  the X data ############
X_data = X_data.reshape(len(y),200,len(inputColumns))



########### Split in train and test set ################
train_data = X_data[np.arange(start=2,stop=np.shape(X_data)[0],step=2)]
test_data = X_data[np.arange(start=1,stop=np.shape(X_data)[0],step=2)]
train_data_y = y[np.arange(start=2,stop=np.shape(y)[0],step=2)]
test_data_y = y[np.arange(start=1,stop=np.shape(y)[0],step=2)]
train_data_group = group[np.arange(start=2,stop=np.shape(group)[0],step=2)]
test_data_group = group[np.arange(start=1,stop=np.shape(group)[0],step=2)]


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
distribution_mean = tf.keras.layers.Dense(8, name='mean')(encoder)
distribution_variance = tf.keras.layers.Dense(8, name='log_variance')(encoder)
latent_encoding = tf.keras.layers.Lambda(sample_latent_features)([distribution_mean, distribution_variance])



encoder_model = tf.keras.Model(input_data, latent_encoding)
encoder_model.summary()
weights = encoder_model.get_weights()
# encoder_model.get_layer('sina1').set_weights([np.ones(np.shape(weights[0]))*0.5,np.zeros(np.shape(weights[1]))])
# weights1 = encoder_model.get_weights()

################### DECODER PART ############
decoder_input = tf.keras.layers.Input(shape=(8)) # probably change the (6) to (2)!
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

encoded = encoder_model(input_data)
decoded = decoder_model(encoded)


autoencoder = tf.keras.models.Model(input_data, decoded)


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
ax1.plot(test_data[2])
# test = np.expand_dims(test_data[1],axis=(2,1))
test = np.expand_dims(test_data[2], axis=0)

testPredicted = autoencoder.predict(test)
testPredicted = testPredicted[0,:,:]
ax2.plot(testPredicted)




#################### t-NSE ###########################
# implement TSNE to be able to plot high dimensional data (TSNE is used for dimensionality reduction comparable to PCA)
#https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html

# from sklearn.manifold import TSNE

encoded = []

for i in range(0,len(test_data)):
    # z.append(testy[i])
    test_new = test_data[i]
    test_new = np.expand_dims(test_new, axis=0)
    op = encoder_model.predict(test_new)
  
    encoded.append(op[0])





# # perplexity is a factor to experiment with. A low perplexity is recommended for small datasets. perplexitiy ranges between 5 and 50
# X_embedded = TSNE(n_components=2,perplexity=13).fit_transform(np.array(encoded))
# X_embedded.shape


# ################# Creating figures #####################
# # plot Latent feature clusters in scatterplot. 

# xx = []
# yy = []
# z = []
# groupcolor = []
# for i in range(0,len(X_embedded)):
#     # z.append(testy[i])
#     test = X_embedded[i]
#     test = np.expand_dims(test, axis=0)
    
#     op = test
#     xx.append(op[0][0])
#     yy.append(op[0][1])
#     z.append(test_data_y[i]) # Fall risk

import pandas as pd
import seaborn as sns


# df = pd.DataFrame()
# df['xx'] = xx
# df['yy'] = yy
# df['z'] = ["fall risk-"+str(k) for k in z]
# df['groupcolor'] = ["subject-"+str(k) for k in test_data_group]


# # plt.figure(figsize=(8, 6))

# # fig3, (ax1,ax2,ax3,ax4) = plt.subplot2(nrows=2, ncols=2)
# fig3, axes = plt.subplots(2, 2, figsize=(15, 5))#, sharey=True

# sns.scatterplot(ax=axes[0,0],x='xx', y='yy',hue='z', data=df)
# axes[0,0].set_title('Raw latent features per perturbation colored by fall risk')
# axes[0,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)

# sns.scatterplot(ax=axes[0,1],x='xx', y='yy',hue='groupcolor', data=df )
# axes[0,1].set_title('Raw latent features per perturbation colored by subject')
# axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='xx', y='yy',hue='groupcolor', data=df)
# plt.show()

# meanDf = df.groupby(['groupcolor']).mean()

# sns.scatterplot(ax=axes[1,0],x='xx', y='yy',hue='groupcolor', data=meanDf )
# axes[1,0].set_title('Average latent features colored by subject')
# axes[1,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)


# # plt.figure(figsize=(8, 6))
# # sns.scatterplot(x='xx', y='yy',hue='groupcolor', data=meanDf)
# # plt.show()


# meanDfwithFallRisk = df.groupby(['groupcolor','z']).mean()

# sns.scatterplot(ax=axes[1,1],x='xx', y='yy',hue='z', data=meanDfwithFallRisk )
# axes[1,1].set_title('Average latent features colored by fall risk')
# axes[1,1].legend(bbox_to_anchor=(0.5, -0.2), loc='upper left', borderaxespad=0)
plt.figure(5)
plt.plot(np.array(encoded))

####################### UMAP ###############################################
import umap.umap_ as umap
reducer = umap.UMAP()
embedding = reducer.fit_transform(np.array(encoded))
embedding.shape

# plt.scatter(
#     embedding[:, 0],
#     embedding[:, 1])
# plt.gca().set_aspect('equal', 'datalim')
# plt.title('UMAP projection of the Penguin dataset', fontsize=24)

xx = []
yy = []
z = []
groupcolor = []
for i in range(0,len(embedding)):
    # z.append(testy[i])
    test = embedding[i]
    test = np.expand_dims(test, axis=0)
    
    op = test
    xx.append(op[0][0])
    yy.append(op[0][1])
    z.append(test_data_y[i]) # Fall risk

df1 = pd.DataFrame()
df1['xx'] = embedding[:, 0]
df1['yy'] = embedding[:, 1]
df1['z'] = ["fall risk-"+str(k) for k in z]
df1['groupcolor'] = ["subject-"+str(k) for k in test_data_group]


fig4, axes = plt.subplots(2, 2, figsize=(15, 5))#, sharey=True

sns.scatterplot(ax=axes[0,0],x='xx', y='yy',hue='z', data=df1)
axes[0,0].set_title('Raw latent features per perturbation colored by fall risk')
axes[0,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)

sns.scatterplot(ax=axes[0,1],x='xx', y='yy',hue='groupcolor', data=df1 )
axes[0,1].set_title('Raw latent features per perturbation colored by subject')
axes[0,1].legend(bbox_to_anchor=(1.05, 1), loc='upper left', borderaxespad=0)

# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='xx', y='yy',hue='groupcolor', data=df)
# plt.show()

meanDf1 = df1.groupby(['groupcolor']).mean()

sns.scatterplot(ax=axes[1,0],x='xx', y='yy',hue='groupcolor', data=meanDf1 )
axes[1,0].set_title('Average latent features colored by subject')
axes[1,0].legend(bbox_to_anchor=(-0.05, 1), loc='upper right', borderaxespad=0)


# plt.figure(figsize=(8, 6))
# sns.scatterplot(x='xx', y='yy',hue='groupcolor', data=meanDf)
# plt.show()


meanDfwithFallRisk1 = df1.groupby(['groupcolor','z']).mean()

sns.scatterplot(ax=axes[1,1],x='xx', y='yy',hue='z', data=meanDfwithFallRisk1 )
axes[1,1].set_title('Average latent features colored by fall risk')
axes[1,1].legend(bbox_to_anchor=(0.5, -0.2), loc='upper left', borderaxespad=0)




############# Encoder model saving stuff ######################

# encoder_model.save('C:\\Users\\michi\\Desktop\\SSI_Stroke\\savingModels\\' + 'model1')


# loaded = tf.saved_model.load('C:\\Users\\michi\\Desktop\\SSI_Stroke\\savingModels\\' + 'model1')


# encoder_model.get_layer('sina1').set_weights(loaded.variables.weights[0])
#loaded.variables.weights[0]
#a = tf.constant([[1, 2], [3, 4]])   
#b = tf.add(a, 1)
#out = tf.multiply(a, b)
