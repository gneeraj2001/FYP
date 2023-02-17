import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten
import tensorflow as tf
import keras
from keras import optimizers
from keras.preprocessing import sequence
from keras.models import Sequential
from keras.layers import Dense, Embedding
from keras.layers import Input, Flatten, Dropout, Activation
from keras.layers import Conv1D, MaxPooling1D
from keras.models import Model
from keras.callbacks import ModelCheckpoint
from keras.callbacks import ReduceLROnPlateau
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import KFold


# Function to split the data into train and test set
def split(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=25)

    return X_train, X_test, y_train, y_test


# Dataset Loading
data = pd.read_csv(r'C:\Users\DELL\PycharmProjects\FYP_proj\Datasets\dataset7.csv')
emotions_filter = ['angry', 'sad', 'neutral', 'happy']

# 376+376+376+188 => Total Number of samples in new dataset (reduce size of dataset)
data = data.loc[data['labels'].isin(['angry', 'sad', 'neutral', 'happy'])]
X = data.iloc[:, 1:-1].values
y = data.iloc[:, -1].values
encoder = LabelEncoder()
y = encoder.fit_transform(y)


def create_model():
    model = Sequential()

    model.add(Conv1D(128, 3, padding='same', input_shape=(193, 1)))
    model.add(Activation('relu'))
    model.add(Dropout(0.1))
    model.add(MaxPooling1D(pool_size=(2)))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.1))

    model.add(Conv1D(128, 3, padding='same'))
    model.add(Activation('relu'))
    model.add(MaxPooling1D(pool_size=(2)))
    model.add(Dropout(0.1))

    model.add(Flatten())
    model.add(Dense(4))
    model.add(Activation('softmax'))

    opt = optimizers.RMSprop(lr=0.00005, rho=0.9, epsilon=None, decay=0.0)
    model.summary()
    model.compile(loss='sparse_categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

    return model


def plots_training(cnn,i):
    # Training Loss vs Validation Loss
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(cnn.history["loss"], label="loss")
    plt.plot(cnn.history["val_loss"], label="val_loss")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig(f'loss_vs_valloss_{i}.png')
    plt.show()

    # Training Accuracy vs Validation Accuracy
    fig, ax = plt.subplots(figsize=(12, 6))
    plt.plot(cnn.history["accuracy"], label="loss")
    plt.plot(cnn.history["val_accuracy"], label="val_accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Cost")
    plt.legend()
    plt.savefig(f'acc_vs_valacc_{i}.png')
    plt.show()



n_split = 3
i=1
for train_index, test_index in KFold(n_split).split(X):

    x_train, x_test = X[train_index], X[test_index]
    x_train = np.expand_dims(x_train, axis=2)
    x_test = np.expand_dims(x_test, axis=2)

    y_train, y_test = y[train_index], y[test_index]

    model = create_model()
    cnn = model.fit(x_train, y_train, epochs=400,validation_data=(x_test, y_test))

    print('Model evaluation ', model.evaluate(x_test, y_test))
    plots_training(cnn,i)
    i = i+1

model.save(r"C:\Users\DELL\PycharmProjects\FYP_proj\Model_KFOLDS")