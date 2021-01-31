import os, sys, shutil, json, time
import numpy as np
# import pandas as pd
import tensorflow as tf
'''
Reading name dataset
'''
print("reading names")
with open("slovak_names.txt", "r") as f:
    namelist = f.read().lower().split('\n')
print("extracting characters")
charset = set()
for char in "".join(namelist):
    charset.add(char)
print(charset)

'''
Word-Onehot Encoder-decoder
'''

class OneHotCoder(object):
    def __init__(self, charset):
        if "$" not in charset:
            charset.add("$")
        self.charlist = sorted(charset)
        self.chardict = {char: num for num, char in enumerate(self.charlist)}
        self.identity_mat = np.eye(len(self.chardict))

    def encode(self, word):
        word_ids = [self.chardict[char] for char in word.lower()]
        return self.identity_mat[word_ids, :]

    def decode(self, code):
        return "".join([self.charlist[np.argmax(row)] for row in code])

    def __call__(self, word_or_code):
        if isinstance(word_or_code, str):
            return self.encode(word_or_code)
        else:
            return self.decode(word_or_code)

coder = OneHotCoder(charset)

'''
Neural network
'''
onehot_size = len(charset)
max_word_size = max([len(name) for name in namelist])

input_layer = tf.keras.layers.Input(shape=[max_word_size, onehot_size])
x = tf.keras.layers.Conv1D(filters=64, kernel_size=1, strides=1, activation="elu")(input_layer)
x = tf.keras.layers.Conv1D(filters=32, kernel_size=1, strides=1, activation="elu")(x)
x = tf.keras.layers.LSTM(units=32, return_sequences=True)(x)
output_layer = tf.keras.layers.Conv1D(filters=onehot_size, kernel_size=1, strides=1, activation="softmax")(x)
model = tf.keras.Model(inputs=input_layer, outputs=output_layer)
model.compile(optimizer="adam", loss=tf.keras.losses.CategoricalCrossentropy())
model.summary()

data = np.array([coder(name.ljust(max_word_size, "$")) for name in namelist])
labels = np.array([coder(name[1:].ljust(max_word_size, "$")) for name in namelist])

model.fit(x=data, y=labels, epochs=50)



