import numpy as np
from keras import layers
from tensorflow import keras
from configuration import ApplicationConfig


class KerasNetwork:
    def __init__(self, app_config: ApplicationConfig):
        self.__batch_size = app_config.batch_size()
        self.__number_epochs = app_config.number_epochs()

        self.__model = keras.Sequential([
            layers.Dense(app_config.hidden_layer_neurons(),
                         activation=app_config.activation_hidden(),
                         input_shape=(app_config.number_of_features(),)),
            layers.Dense(app_config.hidden_layer_neurons(), activation=app_config.activation_hidden()),
            layers.Dense(app_config.number_of_clusters(), activation=app_config.activation_classification())
        ])
        self.__model.compile(optimizer=app_config.optimizer(), loss=app_config.loss(), metrics=['accuracy'])

    def train(self, train_set: tuple, evaluate_set: tuple):
        train_data, train_classes = train_set

        return self.__model.fit(train_data, train_classes,
                                batch_size=self.__batch_size,
                                epochs=self.__number_epochs,
                                validation_data=evaluate_set)

    def evaluate(self, evaluate_set: tuple):
        evaluate_data, evaluate_classes = evaluate_set

        return self.__model.evaluate(evaluate_data, evaluate_classes)

    def predict(self, set_for_prediction: list) -> tuple:
        predicted_data = self.__model.predict(set_for_prediction)
        predicted_classes = np.argmax(predicted_data, axis=1)

        return predicted_data, predicted_classes
