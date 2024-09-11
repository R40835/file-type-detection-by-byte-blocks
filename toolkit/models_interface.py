
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from keras.src.callbacks import History


class BaseModel:
    name = None
    model = None
    history = None
    NUM_CLASSES = 13
    
    def __init__(self, timesteps: int, features: int):
        """
        Abstract Class: extends the model classes to avoid code repetition
        """
        self.features = features
        self.timesteps = timesteps

    def fit(self, x: np.ndarray, y: np.ndarray, validation_data: tuple, epochs: int, batch_size: int) -> History:
        """
        This method trains the models.
        """
        if not hasattr(self, 'model'):
            raise AttributeError("Model has not been initialised.")

        self.history = self.model.fit(x=x, y=y, validation_data=validation_data, epochs=epochs, batch_size=batch_size)
        return self.history
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        This method is for making predictions on unseen data.
        """
        if not hasattr(self, 'model'):
            raise AttributeError("Model has not been initialised.")

        self.predictions = self.model.predict(x)
        return np.argmax(self.predictions, axis=-1)

    def plot_learning_curves(self) -> None:
        """
        This method plots the model learning curves.
        """
        if self.history is None:
            raise ValueError("Model has not been trained yet. Call `fit` method first.")

        plt.plot(self.history.history['loss'])
        plt.plot(self.history.history['val_loss'])
        plt.title(f'{self.name} Learning Curve')
        plt.ylabel('Loss')
        plt.xlabel('Epoch')
        plt.legend(['training', 'validation'], loc='upper right')
