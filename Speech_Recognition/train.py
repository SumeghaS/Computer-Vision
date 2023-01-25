'''
CS7180: Advanced Perception
Name: Sumegha Singhania
Date: 11.11.22
This file contains the training model.
'''
import os
from matplotlib import pyplot as plt
import tensorflow as tf
import tensorflow_io as tfio
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Flatten
from data_loader import prepare_data

def train():
    train,test = prepare_data()

    # Training model
    model = Sequential()
    model.add(Conv2D(16, (3,3), activation='relu', input_shape=(1491, 257,1)))
    model.add(Conv2D(16, (3,3), activation='relu'))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile('Adam', loss='BinaryCrossentropy', metrics=[tf.keras.metrics.Recall(),tf.keras.metrics.Precision()])
    model.summary()

    # Fitting model on the data
    hist = model.fit(train, epochs=4, validation_data=test)

    # Plot loss, precision and recall
    plt.title('Loss')
    plt.plot(hist.history['loss'], 'r')
    plt.plot(hist.history['val_loss'], 'b')
    plt.show()

    plt.title('Precision')
    plt.plot(hist.history['precision'], 'r')
    plt.plot(hist.history['val_precision'], 'b')
    plt.show()

    plt.title('Recall')
    plt.plot(hist.history['recall'], 'r')
    plt.plot(hist.history['val_recall'], 'b')
    plt.show()
    return model