# -*- coding: utf-8 -*-
"""test_gan_discrimenator.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1FZDOOOZAYMNOdqKCxnubRGT8WmMQJ2dA
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
tf.config.run_functions_eagerly(True)

from tensorflow.keras.layers import Input, Reshape, Dropout, Dense 
from tensorflow.keras.layers import Flatten, BatchNormalization,MaxPool2D
from tensorflow.keras.layers import Activation, ZeroPadding2D
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.layers import UpSampling2D, Conv2D
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.optimizers import Adam

import numpy as np
from PIL import Image
from tqdm import tqdm
import os 
import time
import matplotlib.pyplot as plt
import torch

from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from tensorflow.keras import backend as K

GENERATE_SQUARE = 64
IMAGE_CHANNELS = 3

# Size vector to generate images from
SEED_SIZE = 228

# Configuration
DATA_PATH = '/content/drive/My Drive/GAN/attention'
EPOCHS = 500
BATCH_SIZE = 16
BUFFER_SIZE = 1300

def images_features():
  labels =np.load('/content/drive/MyDrive/projects/images_labels.npy') 
  feature =[]
  image = []
  features = np.load('/content/drive/MyDrive/projects/EEG_features.npy')
  images = np.load('/content/drive/MyDrive/images.npy')
  for i in range(1996):
    if labels[i] == 9 :
        feature.append(features[i])
        image.append(images[i])
  feature = np.asarray(feature)
  image = np.asarray(image)
  X = image.astype('float32')
  X = X*2-1
  del  features ,images ,image
  return X , (feature //5)

def cats_data():
  imagenet_img = np.load('/content/drive/MyDrive/projects/imageNet_img.npy') 
  imagenet_l = np.load('/content/drive/MyDrive/projects/imageNet_l.npy') -1
  cats = []
  i = 0
  for label in imagenet_l:
    if label == 9 :
      cats.append(imagenet_img[i])

    i =i + 1
  cats = np.asarray(cats)
  X = cats.astype('float32')
  X = X/255*2-1
  del cats , imagenet_l , imagenet_img 
  return X

im_50 ,feature = images_features()

image_50 = tf.data.Dataset.from_tensor_slices((im_50 ,feature)) \
.batch(BATCH_SIZE)

training_data = cats_data()
# Batch and shuffle the data
train_dataset = tf.data.Dataset.from_tensor_slices(training_data) \
    .shuffle(BUFFER_SIZE).batch(BATCH_SIZE)

def hw_flatten(x):
    # Input shape x: [BATCH, HEIGHT, WIDTH, CHANNELS]
    # flat the feature volume across the width and height dimensions 
    x_shape = tf.shape(x)
    return tf.reshape(x, [x_shape[0], -1, x_shape[-1]]) # return [BATCH, W*H, CHANNELS]

class SelfAttention(tf.keras.Model):
  def __init__(self, number_of_filters, dtype=tf.float32):
    super(SelfAttention, self).__init__()
    self.number_of_filters = number_of_filters
    self.f = Conv2D(number_of_filters//8, 1, 
                                     strides=1, padding='SAME', name="f_x",
                                     activation=None, dtype=dtype)
    
    self.g = Conv2D(number_of_filters//8, 1,
                                     strides=1, padding='SAME', name="g_x",
                                     activation=None, dtype=dtype)
    
    self.h = Conv2D(number_of_filters//2, 1,
                                     strides=1, padding='SAME', name="h_x",
                                     activation=None, dtype=dtype)
   
    self.gamma = tf.compat.v1.get_variable("gamma", [1], initializer=tf.constant_initializer(0.0))
    self.flatten = tf.keras.layers.Flatten()
  
  def call(self, x ):

    f = self.f(x)
    g = self.g(x)
    h = self.h(x)
    f_flatten = hw_flatten(f)
    g_flatten = hw_flatten(g)
    h_flatten = hw_flatten(h)

    s = tf.matmul(g_flatten, f_flatten, transpose_b=True) # [B,N,C] * [B, N, C] = [B, N, N]
    beta = tf.nn.softmax(s, axis=-1)
    o = tf.matmul(beta, h_flatten)
    o = tf.reshape(o, tf.shape(x))
    o = Conv2D(self.number_of_filters//2,kernel_size=1,padding="same")(o) 
    

    y = self.gamma * tf.reshape(o, tf.shape(x)) + x

    return y

def build_generator(seed_size, channels):
    input_sig = Input(shape=(seed_size)) 

    x = Dense(4*4*512,activation="relu",input_dim=seed_size)(input_sig)
    x = Reshape((4,4,512))(x)

    x = UpSampling2D()(x)
    x = Conv2D(512,kernel_size=3,padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
    

    x = UpSampling2D()(x)
    x = Conv2D(256,kernel_size=3,padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)
   
   #attention layer
    x = SelfAttention(number_of_filters = 256*2)(x)
   # x = Attention(256)(x)

    # Output resolution, additional upsampling
    x = UpSampling2D()(x)
    x = Conv2D(128,kernel_size=3,padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)

    x = UpSampling2D(size=(GENERATE_RES,GENERATE_RES))(x)
    x = Conv2D(64,kernel_size=3,padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = Activation("relu")(x)

    # Final CNN layer
    x = Conv2D(channels,kernel_size=3,padding="same")(x)
    x = Activation("tanh")(x)
    model= Model(input_sig, x)

    return model

def build_discriminator(image_shape):
    input_sig = Input(shape=(image_shape)) 

    x = Conv2D(32, kernel_size=3, strides=2, input_shape=image_shape, 
                     padding="same")(input_sig)
    x = LeakyReLU(alpha=0.2)(x)
    #attention layer
    x = SelfAttention(number_of_filters = 32*2)(x)
    #x = Attention(32)(x)

    x = Dropout(0.25)(x)
    x = Conv2D(64, kernel_size=3, strides=2, padding="same")(x)
   # x = ZeroPadding2D(padding=((0,1),(0,1)))(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)
    

    x = Dropout(0.25)(x)
    x = Conv2D(128, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Conv2D(256, kernel_size=3, strides=2, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Conv2D(512, kernel_size=3, strides=1, padding="same")(x)
    x = BatchNormalization(momentum=0.8)(x)
    x = LeakyReLU(alpha=0.2)(x)

    x = Dropout(0.25)(x)
    x = Flatten()(x)
    x = Dense(1042, activation='sigmoid')(x)
    x = Dense(1, activation='sigmoid')(x)
    model= Model(input_sig, x)
    return model

# Preview image 
PREVIEW_ROWS = 4
PREVIEW_COLS = 4

GENERATE_SQUARE = 64
DATA_PATH = '/content/drive/My Drive/GAN/attention'

def save_images(cnt,generated_images):

  image_array = np.full( (64*4,64*4, 3),255,dtype=np.uint8)

  #generated_images = 0.5 * generated_images + 0.5

  image_count = 0
  for row in range(PREVIEW_ROWS):
      for col in range(PREVIEW_COLS):
        r = row * (GENERATE_SQUARE) 
        c = col * (GENERATE_SQUARE) 
        image_array[r:r+GENERATE_SQUARE,c:c+GENERATE_SQUARE] \
            = generated_images[image_count] * 255
        image_count += 1

          
  output_path = os.path.join(DATA_PATH,'output')
  if not os.path.exists(output_path):
    os.makedirs(output_path)
  
  filename = os.path.join(output_path,f"train-{cnt}.png")
  im = Image.fromarray(image_array)
  im.save(filename)

generator = build_generator(SEED_SIZE, IMAGE_CHANNELS)


generator.summary()

image_shape = (GENERATE_SQUARE,GENERATE_SQUARE,IMAGE_CHANNELS)

discriminator = build_discriminator(image_shape)
discriminator.summary()

"""# **Training**"""

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy()

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

generator_optimizer = tf.keras.optimizers.Adam(1.5e-5,0.5)
discriminator_optimizer = tf.keras.optimizers.Adam(1.5e-5,0.5)

@tf.function
def train_step(images,features):
  seed = tf.random.normal([features.shape[0], 100])
  seed = tf.concat([seed,features],1)

  with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
    generated_images = generator(seed, training=True)

    real_output = discriminator(images, training=True)
    fake_output = discriminator(generated_images, training=True)

    gen_loss = generator_loss(fake_output)
    disc_loss = discriminator_loss(real_output, fake_output)
    

    gradients_of_generator = gen_tape.gradient(\
        gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(\
        disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(
        gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(
        gradients_of_discriminator, 
        discriminator.trainable_variables))
  return gen_loss,disc_loss

def train(dataset, epochs ,image_50):

  zeros = tf.zeros([BATCH_SIZE,128])
  start = time.time()

  for epoch in range(epochs):
    epoch = epoch +500
    epoch_start = time.time()
    
    gen_loss_list = []
    disc_loss_list = []
    if epoch > 300 :
      for image_, feat in image_50:
        t=train_step(image_,feat)
        gen_loss_list.append(t[0])
        disc_loss_list.append(t[1])
    else :
      for image_batch in dataset:
        t = train_step(image_batch,zeros)
        
        gen_loss_list.append(t[0])
        disc_loss_list.append(t[1])
      

    g_loss = sum(gen_loss_list) / len(gen_loss_list)
    d_loss = sum(disc_loss_list) / len(disc_loss_list)

    

    epoch_elapsed = (time.time() - start)/60.0
    print (f'Epoch {epoch+1}, gen loss={g_loss},disc loss={d_loss},Training time: {epoch_elapsed}')
    if epoch % 10 == 0:
      if epoch >300 :
        fixed_seed = np.concatenate([np.random.randn(16, 100),feature[:16]],1).astype(np.float32)
        save_images(epoch,(generator.predict(fixed_seed)+1)/2)
      else:
        fixed_seed =  np.concatenate([np.random.randn(16, 100),np.zeros([BATCH_SIZE,128])],1).astype(np.float32)
        save_images(epoch,generator.predict(fixed_seed))

train(train_dataset, EPOCHS , image_50)

"""# **plot result**"""

f, a = plt.subplots(2, 10, figsize=(40, 10))
for i in range(10):
    a[0][i].imshow((im_50[i]+1)/2)
    a[1][i].imshow((gen[i]+1)/2)