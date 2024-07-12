import argparse
import numpy as np
from matplotlib import pyplot
from tf_keras import backend as K, callbacks, optimizers
from tf_keras.saving import custom_object_scope
from model import Cifar10Model
from utils import data_generator


def lr_decay_callback(model, initial_lr, epoch_count):
    def _lr_beta_decay(epoch, logs):
        K.set_value(model.optimizer.lr, initial_lr * np.minimum(2. - epoch * 2 / epoch_count, 1.))
        K.set_value(model.optimizer.beta_1, 0.5 if epoch > epoch_count // 2 else 0.9)

    return callbacks.LambdaCallback(_lr_beta_decay)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Weight Normalization - CIFAR10')
    parser.add_argument('--batch_size', type=int, default=16, help='Size of batch')
    parser.add_argument('--epoch_count', type=int, default=256, help='Number of epochs')
    parser.add_argument('--initial_lr', type=float, default=0.001, help='Initial learning rate')
    parser.add_argument('--random_seed', type=int, default=1, help='Random seed')
    argv, _ = parser.parse_known_args()

    np.random.seed(argv.random_seed)

    train_dataset, test_dataset = data_generator(argv.batch_size)

    with custom_object_scope({'leaky_relu': K.nn.leaky_relu}):
        # Train weighted
        weighted_model = Cifar10Model(weight_norm=True)
        weighted_model.compile(
            optimizer=optimizers.Adam(argv.initial_lr, beta_1=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False,
        )
        weighted_metrics = weighted_model.fit(
            train_dataset,
            epochs=argv.epoch_count,
            validation_data=test_dataset,
            callbacks=[lr_decay_callback(weighted_model, argv.initial_lr, argv.epoch_count)],
        ).history
        weighted_model.summary()

        # Train regular
        regular_model = Cifar10Model(weight_norm=False)
        regular_model.compile(
            optimizer=optimizers.Adam(argv.initial_lr, beta_1=0.9),
            loss='sparse_categorical_crossentropy',
            metrics=['accuracy'],
            run_eagerly=False,
        )
        regular_metrics = regular_model.fit(
            train_dataset,
            epochs=argv.epoch_count,
            validation_data=test_dataset,
            callbacks=[lr_decay_callback(weighted_model, argv.initial_lr, argv.epoch_count)],
        ).history
        regular_model.summary()

    # Draw metrics
    interval = np.linspace(0, argv.epoch_count, argv.epoch_count)

    pyplot.plot(interval, regular_metrics['accuracy'], color='blue', dashes=[6, 2], label='Regular train')
    pyplot.plot(interval, weighted_metrics['accuracy'], color='green', dashes=[6, 2], label='Weighted train')
    pyplot.plot(interval, regular_metrics['val_accuracy'], color='blue', label='Regular valid')
    pyplot.plot(interval, weighted_metrics['val_accuracy'], color='green', label='Weighted valid')
    pyplot.title('Weight normalization on CIFAR10. Batch size: {}'.format(argv.batch_size))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Accuracy')
    pyplot.legend()
    pyplot.savefig('accuracy_{}.png'.format(argv.batch_size))
    pyplot.close()

    pyplot.plot(interval, regular_metrics['loss'], color='blue', dashes=[6, 2], label='Regular train')
    pyplot.plot(interval, weighted_metrics['loss'], color='green', dashes=[6, 2], label='Weighted train')
    pyplot.title('Weight normalization on CIFAR10. Batch size: {}'.format(argv.batch_size))
    pyplot.xlabel('Epoch')
    pyplot.ylabel('Loss')
    pyplot.legend()
    pyplot.savefig('loss_{}.png'.format(argv.batch_size))
    pyplot.close()
