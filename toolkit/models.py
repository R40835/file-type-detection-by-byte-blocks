import keras

import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.models import Sequential # type: ignore
from keras.layers import LSTM, Conv1D, MaxPooling1D, GRU, Dense, Bidirectional, Flatten, Input, Dropout # type: ignore

from .models_interface import BaseModel


class Ffnn(BaseModel):

    name = "Feed Forward Neural Network"

    def __init__(self, timesteps: int=4096, features: int=1):
        self.model = Sequential([
            Input(shape=(timesteps, features)),
            Dense(256, activation='relu'),
            Dense(128, activation='relu'),
            Flatten(),
            Dense(self.NUM_CLASSES, activation="softmax")
        ])

        self.model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        self.model.summary()


class Ffnn2(BaseModel):

    name = "Feed Forward Neural Network"

    def __init__(self, timesteps: int=4096, features: int=1):
        self.model = Sequential([
            Input(shape=(timesteps, features)),
            Dense(352, activation='relu'),
            Dense(32, activation='relu'),
            Flatten(),
            Dense(self.NUM_CLASSES, activation="softmax")
        ])
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        self.model.summary()


class Cnn(BaseModel):

    name = "Convolutional Neural Network"

    def __init__(self, timesteps: int=4096, features: int=1):
        self.model = Sequential([
            Input(shape=(timesteps, features)),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(128, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        self.model.summary()


class Cnn2(BaseModel):

    name = "Convolutional Neural Network"

    def __init__(self, timesteps: int=256, features: int=1):
        self.model = Sequential([
            Input(shape=(timesteps, features)),
            Conv1D(64, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Conv1D(192, kernel_size=3, activation='relu'),
            MaxPooling1D(pool_size=2),
            Flatten(),
            Dropout(0.3),
            Dense(256, activation='relu'),
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001), 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        self.model.summary()


class Gru(BaseModel):

    name = "Gated Reccurent Unit"

    def __init__(self, timesteps: int=4096, features: int=1):
        self.model = Sequential([
            Input(shape=(timesteps, features)),
            Bidirectional(GRU(68, return_sequences=False)),
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        self.model.summary()


class Lstm(BaseModel):

    name = "Long Short-Term Memory"

    def __init__(self, timesteps: int=4096, features: int=1):
        self.model = Sequential([
            Input(shape=(timesteps, features)),
            Bidirectional(LSTM(68, return_sequences=False)),
            Dense(self.NUM_CLASSES, activation='softmax')
        ])

        self.model.compile(
            optimizer='adam', 
            loss='sparse_categorical_crossentropy', 
            metrics=['accuracy']
        )

        self.model.summary()