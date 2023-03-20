
import keras
from keras import regularizers

from keras.models import Sequential, Model, model_from_json, load_model
from keras.layers import Dense, Embedding, LSTM, Reshape, Bidirectional
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras.layers import Conv1D, MaxPooling1D, AveragePooling1D, Conv2D, MaxPooling2D
from keras.utils import np_utils
from keras.callbacks import (EarlyStopping, LearningRateScheduler,
                             ModelCheckpoint, TensorBoard, ReduceLROnPlateau)
from keras import losses, models, optimizers
from keras.activations import relu, softmax
from keras.layers import (Convolution2D, GlobalAveragePooling2D, BatchNormalization, Flatten, Dropout,
                          GlobalMaxPool2D, MaxPool2D, concatenate, Activation, Input, Dense, TimeDistributed)
from keras_self_attention import SeqSelfAttention
import tensorflow as tf
from tensorflow.keras import layers
#tf.keras.layers.Attention
# sklearn
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# Other
from tqdm import tqdm, tqdm_pandas
import scipy
from scipy.stats import skew
import librosa
import librosa.display
import json
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from matplotlib.pyplot import specgram
import pandas as pd
import seaborn as sns
import glob
import os
import sys
import pickle

from Data_Augmentation import speedNpitch


def CNN_2D_prepare_data(df, n, aug, mfcc):
    X = np.empty(shape=(df.shape[0], n, 216, 1))
    input_length = sampling_rate * audio_duration

    cnt = 0

    file_path = df.path
    data, _ = librosa.load('C:/Users/DELL/PycharmProjects/FYP_proj/03-01-05-02-01-01-22.wav', sr=sampling_rate
                           , res_type="kaiser_fast"
                           , duration=2.5
                           , offset=0.5
                           )

    # Random offset / Padding
    if len(data) > input_length:
        max_offset = len(data) - input_length
        offset = np.random.randint(max_offset)
        data = data[offset:(input_length + offset)]
    else:
        if input_length > len(data):
            max_offset = input_length - len(data)
            offset = np.random.randint(max_offset)
        else:
            offset = 0
        data = np.pad(data, (offset, int(input_length) - len(data) - offset), "constant")

    # Augmentation?
    if aug == 1:
        data = speedNpitch(data)

    # which feature?
    if mfcc == 1:
        # MFCC extraction
        MFCC = librosa.feature.mfcc(data, sr=sampling_rate, n_mfcc=n_mfcc)
        MFCC = np.expand_dims(MFCC, axis=-1)
        X[cnt,] = MFCC

    else:
        # Log-melspectogram
        melspec = librosa.feature.melspectrogram(data, n_mels=n_melspec)
        logspec = librosa.amplitude_to_db(melspec)
        logspec = np.expand_dims(logspec, axis=-1)
        X[cnt,] = logspec

    cnt += 1

    return X


# Demo Audio sample
audio = ['C:/Users/DELL/PycharmProjects/FYP_proj/03-01-05-02-01-01-22.wav']  # Change audio sample directory
df = pd.DataFrame(audio, columns=['path'])

sampling_rate=44100
audio_duration=2.5
n_melspec = 80
specgram = CNN_2D_prepare_data(df, n = n_melspec, aug = 0, mfcc = 0)

X = specgram

#Load Model
# loading json and model architecture
json_file = open(r'C:\Users\DELL\PycharmProjects\FYP_proj\Model_Attention_CNNLSTM\model_json_noaug.json', 'r')
loaded_model_json = json_file.read()
json_file.close()

print('done')

loaded_model = model_from_json(loaded_model_json, custom_objects={'SeqSelfAttention': SeqSelfAttention})

# load weights into new model
loaded_model.load_weights(r'C:\Users\DELL\PycharmProjects\FYP_proj\Model_Attention_CNNLSTM\best_model_noaug.h5')#("saved_models/best_model.h5")
print("Loaded model from disk")



