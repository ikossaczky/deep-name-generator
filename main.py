import os, sys, shutil, json, time
import numpy as np
# import pandas as pd
import tensorflow as tf

'''
Word-Onehot Encoder-Decoder
'''
class NameCoder(object):
    def __init__(self, namelist):
        self.namelist = [name.lower() for name in namelist]
        self.charset = set()
        for char in "".join(self.namelist):
            self.charset.add(char)
        self.charset.add("$")
        self.charlist = sorted(self.charset)
        self.chardict = {char: num for num, char in enumerate(self.charlist)}
        self.onehot_base = np.eye(len(self.chardict))[None, ...]

        self.onehot_size = len(self.charset)
        self.max_word_size = max([len(name) for name in self.namelist])

        self.train_data = self.__call__(self.namelist)
        self.train_labels = self.__call__([x[1:] for x in self.namelist])

    def encode(self, word):
        word_ids = [self.chardict[char] for char in word.lower().ljust(self.max_word_size, "$")]
        return self.onehot_base[:, word_ids, :]

    def decode(self, code):
        return "".join([self.charlist[np.argmax(row)] for row in code[0]]) # taking first (0) batch element

    def __call__(self, x):
        if isinstance(x, str):
            return self.encode(x)
        elif isinstance(x, np.ndarray) and len(x.shape)==3 and x.shape[0]==1:
            return self.decode(x)
        elif isinstance(x, np.ndarray) and len(x.shape)==3 and x.shape[0]>1:
            return [self.decode(w[None,...]) for w in x]
        elif isinstance(x, list):
            return np.concatenate([self.encode(w) for w in x], axis=0)
        else:
            raise ValueError("Input should be either string, list of strings or 2D or 3D numpy array.")

class NameInterface(object):
    def __init__(self, coder, model):
        self.coder = coder
        self.model = model

    def extend(self, word_start):
        word = word_start
        for k in range(coder.max_word_size):
            new_letter = coder(model.predict(coder(word)))[len(word)-1]
            if new_letter != "$":
                word = word + new_letter
            else:
                break
        return word

'''
Reading name dataset
'''
print("reading names")
with open("slovak_names.txt", "r") as f:
    namelist = f.read().split('\n')
coder = NameCoder(namelist)

'''
Neural network
'''
input_layer = tf.keras.layers.Input(shape=[coder.max_word_size, coder.onehot_size])
x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, activation="elu")(input_layer)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, activation="elu")(x)
x = tf.keras.layers.LSTM(units=32, return_sequences=True)(x)
output_layer = tf.keras.layers.Conv1D(filters=coder.onehot_size, kernel_size=1, strides=1, activation="softmax")(x)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
model.summary()

model.fit(x=coder.train_data, y=coder.train_labels, epochs=50)

ni = NameInterface(coder, model)