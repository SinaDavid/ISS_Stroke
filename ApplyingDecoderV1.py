# -*- coding: utf-8 -*-
"""
Created on Tue Oct 12 08:28:31 2021

Model pickling 
- using the decoder.
- using the Umap.

create new data.
apply models
generate conventional gait features.




@author: michi
"""
# 1: Loading required libararies.
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras
from keras import backend as K

K.clear_session()


pathToEncoderModel = 'C:\\Users\\michi\\Desktop\\SSI_Stroke\\savingModels\\'
# model = keras.models.load_model(pathToEncoderModel, compile=False)

from keras.models import load_model
model = load_model(pathToEncoderModel)

dummy = np.array([ 15.5187645 ,  -1.220814  ,  -5.016149  , -15.052956  ,
        13.0611925 ,  -0.26420033])
dummy1 = np.expand_dims(dummy, axis=0)

################## Generate dummy data throughout the spectrum ###############
from itertools import permutations
import itertools
# spectrum = np.arange(-14,14,2)
# list1 = [1,2,3,4,5,6]
# list2 = [3,4,5,3,2,1]
# list3 = [6,7,8,9,1,3]
# all_list works in the number f lists the terms the length 
all_list = [[1, 3, 4, 7, 3, 3], [6, 7, 9, 2,1,2], [8, 10, 5,8,4,7] ,[3, 7, 4, 7,3,3], [6, 8, 9, 2,1,2], [8, 1, 5,8,4,7] ]



# d = [4,5,6]
# e = [a,b,c,d]
# perm = permutations(e,6)
# count = 0
# dummyData = [0,0,0,0,0,0]
# for i in list(perm):
#      count+=1
#      print (i)
#      dummyData = np.vstack((dummyData,i))

# res = [[i, j, k] for i in list1 
#                   for j in list2
#                   for k in list3]
res1 = list(itertools.product(*all_list))


pathFeatures = 'C:\\Users\\michi\\Desktop\\SSI_Stroke\\features\\'
features = np.load(pathFeatures + 'features.npy')

featuresCombined = list(itertools.product(*features))
##### Make predictions.##############
dummyData = np.expand_dims(featuresCombined[1000], axis=0)
prediction = model.predict(dummyData)

prediction1 = np.reshape(prediction,(200,6))
plt.figure(2)
plt.plot(prediction1)


###### Next steps ######

# Generate gait features from artificial signals


# apply umap model



# generate heatmap per gait feature















########## Generate conventional measures ################










