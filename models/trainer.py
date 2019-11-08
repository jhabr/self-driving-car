import os
import pandas as pd
from sklearn.model_selection import train_test_split
from keras import backend
from keras_preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt


class Trainer:
    TRAIN_DIR = os.path.join(os.getcwd(), '..', 'images', 'train')

    def __init__(self):
        self.data_frame = None
        self.X_train = None
        self.X_valid = None
        self.y_train = None
        self.y_valid = None
        self.history = None

    def clear_environment(self):
        backend.clear_session()

    def load_data(self):
        self.data_frame = pd.read_csv(
            os.path.join(Trainer.TRAIN_DIR, 'driving_log.csv'),
            names=['center-image', 'left-image', 'right-image', 'steering', 'throttle', 'reverse', 'speed']
        )

        X = self.data_frame[['center-image', 'left-image', 'right-image']].values
        y = self.data_frame['steering'].values

        self.X_train, self.X_valid, self.y_train, self.y_valid = train_test_split(X, y, test_size=0.2, random_state=0)

    def train_brain(self, brain):
        train_generator = self.generator('training')
        validation_generator = self.generator('validation')
        self.history = brain.train(
            train_generator,
            len(self.X_train),
            validation_generator,
            len(self.X_valid)
        )

    def generator(self, subset):
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            validation_split=0.2,
            rotation_range=40,
            width_shift_range=0.2,
            height_shift_range=0.2,
            shear_range=0.2,
            zoom_range=0.2,
            horizontal_flip=True,
            vertical_flip=True,
            fill_mode='constant')

        return train_datagen.flow_from_dataframe(
            dataframe=self.data_frame,
            directory=Trainer.TRAIN_DIR,
            x_col='center-image',
            y_col='steering',
            batch_size=20,
            target_size=(310, 160),
            class_mode='binary',
            subset=subset,
            shuffle=True)

    def plot_history(self):
        acc = self.history.history['accuracy']
        val_acc = self.history.history['val_accuracy']
        loss = self.history.history['loss']
        val_loss = self.history.history['val_loss']

        epochs = range(len(acc))

        plt.plot(epochs, acc, color='blue', label='Training accuracy')
        plt.plot(epochs, val_acc, color='red', label='Validation accuracy')
        plt.title('Training and validation accuracy')
        plt.legend()

        plt.figure()

        plt.plot(epochs, loss, color='blue', label='Training Loss')
        plt.plot(epochs, val_loss, color='red', label='Validation Loss')
        plt.title('Training and validation loss')
        plt.legend()

        plt.show()
