# -*- coding: utf-8 -*-
"""Conv1D_final.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/19p4JQktT-6duKYLrp0kLOSd99uOPKU9D
"""

from google.colab import drive
drive.mount('/content/drive')

!cp /content/drive/MyDrive/Dataset/eeg/data_import.py /content
from data_import import data_load 

strt= 160
flen = 160
x_train,y_train,x_test,y_test,x_val,y_val = data_load(strt ,flen)

import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import torch
from tensorflow import keras
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from tensorflow.keras import Input , Model
from tensorflow.keras.layers import Conv1D , MaxPooling1D , Flatten , Dense, UpSampling1D , Reshape ,AveragePooling1D, UpSampling1D, Permute, RepeatVector, Lambda, Multiply, Reshape
from keras.callbacks import ModelCheckpoint, EarlyStopping
# from keras.utils import to_categorical

from torchvision import transforms
import torch.utils.data as data

import torch.optim as optim
from torch.autograd import Variable
import torch
import torch.nn
import torch.optim as optim
from torch.autograd import Variable
from tensorflow.keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K
from keras.models import Sequential
import matplotlib.pyplot as plt
from sklearn import preprocessing

y_train = to_categorical(y_train, num_classes=40)
y_test = to_categorical(y_test, num_classes=40)

# defining the keras callbacks to be used while training the network
modelcheckpoint = ModelCheckpoint('lstm_model.h5', save_best_only=True, monitor='val_acc', verbose=1)
earlystopping = EarlyStopping(monitor='val_acc', verbose=1, patience=10)

class Attention(Layer):
    
    def __init__(self, return_sequences=True):
        self.return_sequences = return_sequences
        super(Attention,self).__init__()
        
    def build(self, input_shape):
        
        self.W=self.add_weight(name="att_weight", shape=(input_shape[-1],1),
                               initializer="normal")
        self.b=self.add_weight(name="att_bias", shape=(input_shape[1],1),
                               initializer="zeros")
        
        super(Attention,self).build(input_shape)
        
    def call(self, x):
        
        e = K.tanh(K.dot(x,self.W)+self.b)
        a = K.softmax(e, axis=1)
        output = x*a
        
        if self.return_sequences:
            return output
        
        return K.sum(output, axis=1)

#model
input_sig = Input(shape=(160,128)) 
x = Conv1D(128,3, activation='relu', padding='valid')(input_sig)
x_ = MaxPooling1D(2)(x)
x1 = Conv1D(64,3, activation='relu', padding='valid')(x_)
x1_ = MaxPooling1D(2)(x1)
x2 = Conv1D(32,3, activation='relu', padding='valid')(x1_)
x2_ = MaxPooling1D(2)(x2)
att = Attention(return_sequences=True)(x2_)
flat = Flatten()(att)
dense = Dense(128,activation = 'relu')(flat)
encoded = Dense(40,activation = 'softmax')(dense)
autoencoder = Model(input_sig, encoded)
autoencoder.compile(optimizer='adam', loss='mse', metrics=['accuracy'])

autoencoder.summary()

history = autoencoder.fit(x_train, y_train, batch_size=128, epochs=75, callbacks=[modelcheckpoint, earlystopping], validation_data=(x_test, y_test))

print(autoencoder.evaluate(x_test, y_test))

loss_train = history.history['loss']
loss_val = history.history['val_loss']
epochs = range(0,40)
plt.plot(epochs, loss_train, 'g', label='Training loss')
plt.plot(epochs, loss_val, 'b', label='validation loss')
plt.title('Training and Validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

acc_train = history.history['accuracy']
acc_val = history.history['val_accuracy']
epochs = range(0,40)
plt.plot(epochs, acc_train, 'g', label='Training accuracy')
plt.plot(epochs, acc_val, 'b', label='validation accuracy')
plt.title('Training and Validation accuracy')
plt.xlabel('Epochs')
plt.ylabel('accuracy')
plt.legend()
plt.show()

from tensorflow.keras.layers import Flatten, Dense, Bidirectional, TimeDistributed, LSTM


model = Sequential()
model.add(Bidirectional(LSTM(128, return_sequences=True), input_shape=(160, 128)))
model.add(Attention(return_sequences=True)) # receive 3D and output 3D
# model.add(Bidirectional(LSTM(300)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(40, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

print('MODEL TRAINING BEGINNING------------->')
history = model.fit(x_train, y_train, batch_size=128, epochs=40, callbacks=[modelcheckpoint, earlystopping], validation_data=(x_test, y_test))

print(model.evaluate(x_test, y_test))