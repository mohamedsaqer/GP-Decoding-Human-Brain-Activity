# -*- coding: utf-8 -*-
"""Temp Lstm Real image Dataset.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1xZ2LTzKWBszSXwRTyFwYTgN_ZtHfcR6x
"""

from google.colab import drive
drive.mount('/content/drive')

import tensorflow as tf
from tensorflow import keras  
from tensorflow.keras import layers
import numpy as np
import h5py
import csv

path = '/content/drive/My Drive/Dataset/'

f = h5py.File(path+'random.hdf5','r')
l = f.get('default')
fileArray = np.empty((4608,128,1230),dtype='d')
fileArray = l[:][:][:]
f.close()
del(l,f)

#print(fileArray[192])
print(fileArray.shape,fileArray.dtype)

while True: pass

#ltrain = 3456
#ltest = 1152
x_train = fileArray[:3456,:,:]
x_val = fileArray[3456:,:,:]
'''
for s in range(18):
   for o in range(192):
        x_train.append(fileArray[o,:,:,s]) 
 = []
for s in range(18,24,1):
   for o in range(192):
        x_val.append(fileArray[o,:,:,s])
x_train = np.array(x_train)
x_val = np.array(x_val)'''
del(fileArray)

classArray = []
with open(path + 'class.csv') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    for row in csv_reader:
      classArray.append(row)
#print(classArray)
print(len(classArray))
y_train= []
for s in range(18):
   for o in range(192):
        classArray[s][o]
        y_train.append(float(classArray[s][o])/10)
y_val = []
for s in range(18,24,1):
   for o in range(192):
        y_val.append(float(classArray[s][o])/10)
y_train = np.array(y_train)
y_val = np.array(y_val)
del(classArray)
print(y_train[2000])
print(x_train.shape,x_val.shape,x_train.dtype)
print(y_train.shape,y_val.shape)

inp = 3456
vocab_size = 3456  # Only consider the top 20k words
maxc = 128  # Only consider the first 200 words of each movie review
maxt = 1230
    
model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(maxc, maxt)),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])
model.fit(x_train, y_train, epochs=10)

test_loss, test_acc = model.evaluate(x_train, y_train, verbose=2)

print('\nTest accuracy:', test_acc)

if __name__ == '__main__':
    vocab_size = 20000  # Only consider the top 20k words
    maxlen = 200  # Only consider the first 200 words of each movie review
    print(len(x_train), "Training sequences")
    print(len(x_val), "Validation sequences")
    x_train = keras.preprocessing.sequence.pad_sequences(x_train, maxlen=maxlen)
    x_val = keras.preprocessing.sequence.pad_sequences(x_val, maxlen=maxlen)

    embed_dim = 32  # Embedding size for each token
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    inputs = layers.Input(shape=(maxlen,))
    embedding_layer = TokenAndPositionEmbedding(maxlen, vocab_size, embed_dim)
    x = embedding_layer(inputs)
    transformer_block = TransformerBlock(embed_dim, num_heads, ff_dim)
    x = transformer_block(x)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dropout(0.1)(x)
    x = layers.Dense(20, activation="relu")(x)
    x = layers.Dropout(0.1)(x)
    outputs = layers.Dense(2, activation="softmax")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)

    model.compile("adam", "sparse_categorical_crossentropy", metrics=["accuracy"])
    history = model.fit(
        x_train, y_train, batch_size=32, epochs=2, validation_data=(x_val, y_val)
    )

    import tensorflow as tf
    tf.test.is_gpu_available(
        cuda_only=False, min_cuda_compute_capability=None
    )



print(x_train.dtype,'\n',y_train)

class MultiHeadSelfAttention(layers.Layer):
    def __init__(self, embed_dim, num_heads=8):
        super(MultiHeadSelfAttention, self).__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        if embed_dim % num_heads != 0:
            raise ValueError(
                f"embedding dimension = {embed_dim} should be divisible by number of heads = {num_heads}"
            )
        self.projection_dim = embed_dim // num_heads
        self.query_dense = layers.Dense(embed_dim)
        self.key_dense = layers.Dense(embed_dim)
        self.value_dense = layers.Dense(embed_dim)
        self.combine_heads = layers.Dense(embed_dim)

    def attention(self, query, key, value):
        score = tf.matmul(query, key, transpose_b=True)
        dim_key = tf.cast(tf.shape(key)[-1], tf.float32)
        scaled_score = score / tf.math.sqrt(dim_key)
        weights = tf.nn.softmax(scaled_score, axis=-1)
        output = tf.matmul(weights, value)
        return output, weights

    def separate_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.projection_dim))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, inputs):
        # x.shape = [batch_size, seq_len, embedding_dim]
        batch_size = tf.shape(inputs)[0]
        query = self.query_dense(inputs)  # (batch_size, seq_len, embed_dim)
        key = self.key_dense(inputs)  # (batch_size, seq_len, embed_dim)
        value = self.value_dense(inputs)  # (batch_size, seq_len, embed_dim)
        query = self.separate_heads(
            query, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        key = self.separate_heads(
            key, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        value = self.separate_heads(
            value, batch_size
        )  # (batch_size, num_heads, seq_len, projection_dim)
        attention, weights = self.attention(query, key, value)
        attention = tf.transpose(
            attention, perm=[0, 2, 1, 3]
        )  # (batch_size, seq_len, num_heads, projection_dim)
        concat_attention = tf.reshape(
            attention, (batch_size, -1, self.embed_dim)
        )  # (batch_size, seq_len, embed_dim)
        output = self.combine_heads(
            concat_attention
        )  # (batch_size, seq_len, embed_dim)
        return output

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0.1):
        super(TransformerBlock, self).__init__()
        self.att = MultiHeadSelfAttention(embed_dim, num_heads)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        self.dropout1 = layers.Dropout(rate)
        self.dropout2 = layers.Dropout(rate)

    def call(self, inputs, training):
        attn_output = self.att(inputs)
        attn_output = self.dropout1(attn_output, training=training)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)
        ffn_output = self.dropout2(ffn_output, training=training)
        return self.layernorm2(out1 + ffn_output)

class TokenAndPositionEmbedding(layers.Layer):
    def __init__(self, maxlen, vocab_size, embed_dim):
        super(TokenAndPositionEmbedding, self).__init__()
        self.token_emb = layers.Embedding(input_dim=vocab_size, output_dim=embed_dim)
        self.pos_emb = layers.Embedding(input_dim=maxlen, output_dim=embed_dim)

    def call(self, x):
        maxlen = tf.shape(x)[-1]
        positions = tf.range(start=0, limit=maxlen, delta=1)
        positions = self.pos_emb(positions)
        x = self.token_emb(x)
        return x + positions