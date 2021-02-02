# -*- coding: utf-8 -*-
"""BILSTM(40 class dataset).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1llX0WJA5o1w0LDtg4IWX5MWXwX2UNyLu
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import torch
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import Input , Model
from tensorflow.keras.layers import Conv1D , MaxPooling1D , Flatten , Dense , UpSampling1D , Reshape ,AveragePooling1D, UpSampling1D

from torchvision import transforms
import torch.utils.data as data
import numpy as np
#from Dataloader import *
#from model import *
import torch.nn
import torch.optim as optim
from torch.autograd import Variable
import torch
class eegloader(data.Dataset):
	def __init__(self, data_path, split_path, dtype='train', data_dir='./', split_no=0, dlen=160, stpt=320, nch=128):

		data = torch.load(data_dir + data_path)
		split = torch.load(data_dir + split_path)

		self.mean = data['means']
		self.stdev = data['stddevs']
		self.labels = split['splits'][split_no][dtype]

		self.data = []
		for l in self.labels:
			self.data.append(data['dataset'][l])

		assert len(self.data)==len(self.labels)
		self.dlen = dlen
		self.stpt = stpt
		self.nch = nch

	def __len__(self):
		return len(self.data)
	
	def __getitem__(self, idx):
		nch  = self.nch
		dlen = self.dlen 
		stpt = self.stpt
	
		x = np.zeros((nch,dlen))
		y = self.data[idx]['label']
		s = self.data[idx]['subject'] 
		
		x = torch.from_numpy(x)
		x[:,:min(int(self.data[idx]['eeg'].shape[1]),dlen)] = self.data[idx]['eeg'][:,stpt:stpt+dlen]
		x = x.type(torch.FloatTensor).sub(self.mean.expand(nch,dlen))/ self.stdev.expand(nch,dlen)

		return x, y, s

data_dir = '/content/drive/MyDrive/GP/'
batch_size = 128
data_path = 'eeg_signals_128_sequential_band_all_with_mean_std.pth'
split_path = 'block_splits_by_image.pth'
split_no = 3

#trainset = data.DataLoader(eegloader(data_path, split_path, dtype='train', data_dir=data_dir, split_no=split_no),batch_size=batch_size, shuffle=True)
#valset = data.DataLoader(eegloader(data_path, split_path, dtype='val', data_dir=data_dir, split_no=split_no),batch_size=batch_size)
#testset = data.DataLoader(eegloader(data_path, split_path, dtype='test', data_dir=data_dir, split_no=split_no),batch_size=batch_size)
#print('data loaded')

x = eegloader(data_path, split_path, dtype='train', data_dir=data_dir, split_no=split_no)
y = eegloader(data_path, split_path, dtype='val', data_dir=data_dir, split_no=split_no)
z = eegloader(data_path, split_path, dtype='test', data_dir=data_dir, split_no=split_no)

print(x[0][0])

training_x=[]
training_y=[]
testing_x=[]
testing_y=[]
for i in range(7972):
  training_x.append(tf.make_ndarray(tf.make_tensor_proto(x[i][0])))
  training_y.append(x[i][1])

for i in range(1997):
  testing_x.append(tf.make_ndarray(tf.make_tensor_proto(z[i][0])))
  testing_y.append(z[i][1])

training = np.array(training_x)
labeling = np.array(training_y)
test_x = np.array(testing_x)
test_y = np.array(testing_y)

labels = []
for i in range(7972):
  labels.append(tf.keras.utils.to_categorical(training_y[i], num_classes=40))

from keras.utils import to_categorical
y_train = to_categorical(labeling, num_classes=40)
y_test = to_categorical(test_y, num_classes=40)

# defining the keras callbacks to be used while training the network

from keras.callbacks import ModelCheckpoint, EarlyStopping
modelcheckpoint = ModelCheckpoint('lstm_model.h5', save_best_only=True, monitor='val_acc', verbose=1)
earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=10)

from tensorflow.keras import Input
from keras.models import Sequential
from keras.layers import LSTM
from keras.layers import Dense
from keras.layers import TimeDistributed
from keras.layers import Bidirectional
import matplotlib.pyplot as plt
from sklearn import preprocessing

def lstm_model(output_dim=300, num_classes=40):

    input_layer  = Input(shape=(128, 160))
    #negm_model
    # lstm1 = LSTM(output_dim, return_sequences=False)(input_layer)
    # dense1 = Dense(128, activation='relu')(lstm1)  
    # output = Dense(num_classes, activation='softmax')(dense1)
    # model = Model(inputs=input_layer, outputs=output)
    # model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    #78% Acc
    # model = Sequential()
    # model.add(Bidirectional(LSTM(output_dim, return_sequences=True), input_shape=(128, 160)))
    # model.add(Bidirectional(LSTM(output_dim)))
    # model.add(Dense(128, activation='relu'))
    # model.add(Dense(40, activation='softmax'))
    # model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    #forward & backward
    model = Sequential()
    forward_layer = LSTM(output_dim, return_sequences=True)
    backward_layer = LSTM(output_dim, activation='relu', return_sequences=True, go_backwards=True)
    model.add(Bidirectional(forward_layer, backward_layer=backward_layer, input_shape=(128, 160)))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(40, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
    return model

#one-hot-encoding the labels of classes


# defining the keras callbacks to be used while training the network

from keras.callbacks import ModelCheckpoint, EarlyStopping
modelcheckpoint = ModelCheckpoint('lstm_model.h5', save_best_only=True, monitor='val_acc', verbose=1)
earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=10)


# creating the model object and putting the model to training and testing  
# we use validation data as test data and vice versa because we are not performing any kind of model selection 

model = lstm_model(output_dim=300, num_classes=40)
model.summary()
print('MODEL TRAINING BEGINNING------------->')
model.fit(training, y_train, batch_size=16, epochs=1000, callbacks=[modelcheckpoint, earlystopping], validation_data=(test_x, y_test))
print('EVALUATING MODEL ON TEST SET------------>')
print(model.evaluate(x_test, y_test))

# print(x_test.shape)
x1 = model.predict(test_x)
# print(h[0][0][0],x1.shape,len(x1[0]),x1[1][10]*40-0)
print(np.argmax(x1[1]),
      x1.shape,
      test_y[0:10], 
      sep='\n')

t = 0
f = 0
for i in range(1997):
  tst = np.argmax(x1[i])
  if(tst==test_y[i]):t+=1
  else:f+=1

print(t,f,t/1997)