'''Code for fine-tuning Inception V3 for a new task.

Start with Inception V3 network, not including last fully connected layers.

Train a simple fully connected layer on top of these.


'''

import numpy as np
from keras.preprocessing import image
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.models import Sequential, Model
from keras.layers import Dense, Activation, Flatten, Dropout
import ourNetwork as inception
import keras.backend as K

#KERAS
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation, Flatten
from keras.layers.convolutional import Convolution2D, MaxPooling2D
from keras.utils.layer_utils import layer_from_config
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils


import matplotlib.pyplot as plt
import matplotlib
import os
from numpy import *
# SKLEARN
from sklearn.utils import shuffle
from sklearn.cross_validation import train_test_split


N_CLASSES = 100
IMSIZE = (64, 64)

# TO DO:: Replace these with paths to the downloaded data.
# Training directory
train_dir = '/home/user/Desktop/CMPT726/new_all_200/train'
# Testing directory
test_dir = '/home/user/Desktop/CMPT726/new_all_200/validation'


# Start with an Inception V3 model, not including the final softmax layer.
base_model = inception.VGG16(weights=None)
print ('Loaded  model')

# Turn off training on base model layers
#for layer in base_model.layers:
#    layer.trainable = False

# Add on new fully connected layers for the output classes.
'''
x = Dense(1024, activation='relu')(base_model.get_layer('flatten').output)
x = Dropout(0.5)(x)
x = Dense(1024, activation='relu', name='fc3')(x)
x = Dropout(0.5)(x)
#x = Dense(6, activation='relu', name='fc4')(x)
predictions = Dense(N_CLASSES, activation='softmax', name='predictions')(x)

'''
predictions=base_model.get_layer('predictions').output
model = Model(input=base_model.input, output=predictions)

sgd=SGD(lr=0.01,decay=0,momentum=0.9,nesterov=True)
#K.set_value(sgd.lr,0.5*K.get_value(sgd.lr))

model.compile(loss='categorical_crossentropy', optimizer=sgd, metrics=['accuracy'])
model.load_weights('../weights/chinese_100_class.h5') 


img_path = '../samples/fo2.bmp'
img = image.load_img(img_path, target_size=IMSIZE).convert("L")
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)

#x = inception.preprocess_input(x)
out_lay_num=5   #output three convolution layers, i.e. layers 1,3 and 5
dimen = 16  #  for layer 1, use 8; layer 3, use 11; layer 5, use 16
preds = model.predict(x)
print('Predicted:', preds)


def plot_filters(layer,x):   # x and y are the height and width of each filter image
	filters = layer.get_weights()
	filters_w = filters[0]
	num_out_filt=size(filters_w,axis = 3)
	#figlen = np.sqrt(num_out_filt)
	#print(figlen)
	fig = plt.figure(figsize=(x,x))
	for j in range(x*x):
		ax = fig.add_subplot(x,x,j+1)
		ax.imshow(filters_w[:,:,0,j],cmap=matplotlib.cm.gray)#
		#ax.matshow(filters_w[:,:,0,j],cmap = matplotlib.cm.binary)
		plt.xticks(np.array([]))
		plt.yticks(np.array([]))
		plt.tight_layout()
	plt.savefig('filt_bw_fo'+str(out_lay_num)+'.png')
	return plt

plot_filters(model.layers[out_lay_num],dimen)
