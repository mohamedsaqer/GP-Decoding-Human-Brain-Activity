from tensorflow.keras import layers
from torchvision import transforms
from tensorflow.keras.models import Sequential
from keras.layers import Input, Dense, LSTM,Dropout
import keras
import tensorflow as tf,keras
from keras.models import Model

class TransformerBlock(layers.Layer):
    def __init__(self, embed_dim, num_heads, ff_dim, rate=0):
        super(TransformerBlock, self).__init__()
        self.att = layers.MultiHeadAttention(num_heads=num_heads, key_dim=embed_dim)
        self.ffn = keras.Sequential(
            [layers.Dense(ff_dim, activation="relu"), layers.Dense(embed_dim),]
        )
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)
        
    def call(self, inputs, training):
        attn_output = self.att(inputs, inputs)
        out1 = self.layernorm1(inputs + attn_output)
        ffn_output = self.ffn(out1)        
        return self.layernorm2(out1 + ffn_output)


def transformer(x , y):
    num_heads = 2  # Number of attention heads
    ff_dim = 32  # Hidden layer size in feed forward network inside transformer

    sequence_input = tf.keras.layers.Input(shape=(x,y))
    transformer_block = TransformerBlock(y, num_heads, ff_dim)
    x = transformer_block(sequence_input)
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    outputs = layers.Dense(40, activation="softmax")(x)
    model = tf.keras.Model(inputs=sequence_input, outputs=outputs)
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    model.summary()
    return(model)
    