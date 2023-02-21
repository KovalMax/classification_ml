import matplotlib.pyplot as plt
from configuration import ApplicationConfig
from dataset_generator import DatasetGenerator
from network import KerasNetwork


def run_ml():
    config = ApplicationConfig()
    dataset_generator = DatasetGenerator(config)
    network = KerasNetwork(config)

    dataset, dataset_classes, colors = dataset_generator.get_dataset()
    plt.scatter(dataset[:, 0], dataset[:, 1], c=colors[dataset_classes], alpha=0.5)
    plt.title('Original dataset', fontsize=16)
    plt.show()

    train, validation = dataset_generator.split_dataset(dataset, dataset_classes, int(0.8 * dataset.shape[0]))
    history = network.train(train, validation)

    loss, accuracy = network.evaluate(validation)
    print(f'Validation accuracy: {accuracy}\nValidation loss: {loss}')

    plt.plot(history.history['accuracy'], label='Training accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation accuracy')
    plt.legend()
    plt.title('Accuracy')
    plt.show()

    plt.plot(history.history['loss'], label='Training loss')
    plt.plot(history.history['val_loss'], label='Validation loss')
    plt.legend()
    plt.title('Loss')
    plt.show()

    predicted_data, predicted_classes = network.predict(dataset)

    plt.scatter(predicted_data[:, 0], predicted_data[:, 1], c=colors[predicted_classes], alpha=0.5)
    plt.title('Predicted(classified) dataset')
    plt.show()


run_ml()
