from random import shuffle, seed
from sklearn.datasets import make_blobs
from configuration import ApplicationConfig
import numpy as np


class DatasetGenerator:
    colors = ['r', 'g', 'b', 'y', 'c', 'm', 'k']

    def __init__(self, app_config: ApplicationConfig):
        self.__seed_number = app_config.seed_number()
        self.__samples = app_config.samples()
        self.__number_of_clusters = app_config.number_of_clusters()
        self.__number_of_features = app_config.number_of_features()

    def get_dataset(self) -> tuple:
        # Generate dataset
        seed(self.__seed_number)
        np.random.seed(self.__seed_number)
        x, y = make_blobs(n_samples=self.__samples,
                          centers=self.__number_of_clusters,
                          n_features=self.__number_of_features,
                          random_state=0)

        # Shuffle the dataset
        idx = list(range(len(x)))
        shuffle(idx)
        x = x[idx]
        y = y[idx]

        # Pick colors for dots
        colors_copy = self.colors
        shuffle(colors_copy)

        return x, y, np.array(colors_copy[:self.__number_of_clusters])

    @staticmethod
    def split_dataset(data: list[int], classes: list[int], train_allocation: int) -> tuple:
        train_data, train_classes = data[:train_allocation], classes[:train_allocation]
        validation_data, validation_classes = data[train_allocation:], classes[train_allocation:]

        return (train_data, train_classes), (validation_data, validation_classes)
