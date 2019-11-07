from keras.callbacks import ModelCheckpoint
from keras.optimizers import Adam
from keras.models import Sequential, load_model
from keras.layers import Lambda, Conv2D, Dropout, Flatten, Dense


class Model:

    def __init__(self):
        self.model = None

    def build(self):
        model = Sequential()
        model.add(Lambda(lambda x: x / 127.5 - 1.0, input_shape=(310, 160, 3)))
        model.add(Conv2D(24, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(36, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(48, (5, 5), activation='elu', strides=(2, 2)))
        model.add(Conv2D(64, (5, 5), activation='elu'))
        model.add(Dropout(0.5))
        model.add(Flatten())
        model.add(Dense(100, activation='elu'))
        model.add(Dense(50, activation='elu'))
        model.add(Dense(10, activation='elu'))
        model.add(Dense(1))

        model.summary()

        self.model = model

    def load(self, path):
        self.model = load_model(path)

    def load_weights(self, path):
        self.build()
        self.model.load_weights(path)

    def train(self, train_generator, steps, validation_generator, validation_steps):
        checkpoint = ModelCheckpoint(
            'models/model.h5',
            monitor='val_loss',
            verbose=1,
            save_best_only=True,
            save_weights_only=True,
            mode='auto'
        )

        self.model.compile(
            loss='mean_squared_error',
            optimizer=Adam(lr=0.0001),
            metrics=['accuracy']
        )

        self.model.fit_generator(
            train_generator,
            steps_per_epoch=steps,
            epochs=40,
            validation_data=validation_generator,
            validation_steps=validation_steps,
            callbacks=[checkpoint],
            verbose=2
        )

    def predict_steering_angle(self, image):
        return float(self.model.predict(image, batch_size=1))
