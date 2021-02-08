import os, sys, shutil, json, time
import numpy as np
import matplotlib.pyplot as plt
# import pandas as pd
import tensorflow as tf
import tensorflowjs as tfjs
from collections.abc import Iterable

'''
Word-Onehot Encoder-Decoder class
'''
class NameCoder(object):
    def __init__(self, namelist):
        self.namelist = [name.lower() for name in namelist]
        self.charset = set()
        for char in "".join(self.namelist):
            self.charset.add(char)
        self.start_token = "#"
        self.end_token = "$"
        self.charset.add(self.start_token)
        self.charset.add(self.end_token)

        self.charlist = sorted(self.charset)
        self.chardict = {char: num for num, char in enumerate(self.charlist)}
        self.onehot_base = np.eye(len(self.chardict))[None, ...]

        self.onehot_size = len(self.charset)
        self.max_word_size = max([len(name) for name in self.namelist]) + 1 # +1 for start token

        self.train_data = self.encode_or_decode([self.start_token + x for x in self.namelist])
        self.train_labels = self.encode_or_decode(self.namelist)

    def __call__(self, x, randomness_coefs=0):
        return self.encode_or_decode(x, randomness_coefs=randomness_coefs)

    def encode(self, word):
        word_ids = [self.chardict[char] for char in word.lower().ljust(self.max_word_size, self.end_token)]
        return self.onehot_base[:, word_ids, :]

    def get_character(self, probabilities, randomness_coef=0):
        if randomness_coef:
            x = probabilities**(1/randomness_coef)
            new_probs = x / x.sum()
            return self.charlist[np.random.choice(np.arange(self.onehot_size), p=new_probs)]
        else:
            return self.charlist[np.argmax(probabilities)]

    def decode(self, code, randomness_coefs=[0]):
        if not isinstance(randomness_coefs, Iterable):
            randomness_coefs = [randomness_coefs]*self.max_word_size
        elif len(randomness_coefs)<self.max_word_size:
            randomness_coefs = randomness_coefs + [randomness_coefs[-1]] * self.max_word_size
        return "".join([self.get_character(row, randomness_coef=randomness_coefs[k])
                        for k, row in enumerate(code[0])])  # taking first (0) batch element

    def encode_or_decode(self, x, randomness_coefs=0):
        if isinstance(x, str):
            return self.encode(x)
        elif isinstance(x, np.ndarray) and len(x.shape)==3 and x.shape[0]==1:
            return self.decode(x, randomness_coefs=randomness_coefs)
        elif isinstance(x, np.ndarray) and len(x.shape)==3 and x.shape[0]>1:
            return [self.decode(w[None,...], randomness_coefs=randomness_coefs) for w in x]
        elif isinstance(x, list):
            return np.concatenate([self.encode(w) for w in x], axis=0)
        else:
            raise ValueError("Input should be either string, list of strings or 2D or 3D numpy array.")

    def coder_props(self):
        coder_props = {}
        coder_props["onehot_size"] = self.onehot_size
        coder_props["max_word_size"] = self.max_word_size
        coder_props["start_token"] = self.start_token
        coder_props["end_token"] = self.end_token
        coder_props["start_token"] = self.start_token
        coder_props["chardict"] = self.chardict
        coder_props["charlist"] = self.charlist
        return coder_props
'''
Python Inference Interface class
'''
class InferenceInterface(object):
    def __init__(self, coder, model):
        self.coder = coder
        self.model = model

    def generate(self, word_start, randomness_coefs=[0]):
        word = self.coder.start_token + word_start
        for k in range(self.coder.max_word_size):
            new_letter = self.coder(model.predict(self.coder(word)), randomness_coefs=randomness_coefs)[len(word)-1]
            if new_letter != "$":
                word = word + new_letter
            else:
                break
        return word.replace(self.coder.start_token, "")

    def get_probs(self, word_start):
        word = self.coder.start_token + word_start
        return model.predict(self.coder(word))[0, len(word)-1, :]

    def plot_probs(self, word_start):
        word = self.coder.start_token + word_start
        plt.bar(self.coder.charlist, self.get_probs(word))
        plt.show()

'''
Reading name dataset
'''
dataset_key = "slovak"
datasets = {"slovak": "slovak_names.txt"}
print("reading names")
with open(datasets[dataset_key], "r") as f:
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

'''
Training, saving examining
'''
# training:
model.fit(x=coder.train_data, y=coder.train_labels, epochs=150)

# saving
output_path="tmp/{}".format(dataset_key)
if not os.path.exists(output_path):
    os.makedirs(output_path)
# save model in keras h5 format
model.save(os.path.join(output_path, "model.h5"))
# save model in tfjs format:
tfjs.converters.save_keras_model(model, output_path)
# fix a tfjs bug (https://github.com/tensorflow/tfjs/issues/3786)
with open(os.path.join(output_path, "model.json"), "r") as f:
    corrected = f.read().replace("Functional", "Model")
with open(os.path.join(output_path, "model.json"), "w") as f:
    f.write(corrected)
# save coder serialization properties
with open(os.path.join(output_path, "coder.json"), "w") as f:
    json.dump(coder.coder_props(), f)

# examining
ii = InferenceInterface(coder, model)

# tensorflowjs_converter --input_format keras ./tmp/slovak/model.h5 ./tmp/slovak/