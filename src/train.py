import tensorflow as tf
import json
import argparse
from model import GoModel
from dataloader import load_data
import golois
import gc
import os
from datetime import datetime

"""def train_model(epochs, batch_size, N, planes, moves, filters):
    go = GoModel(planes, filters)
    model = go.build()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                  loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                  loss_weights={'policy': 1.0, 'value': 1.0},
                  metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

    input_data, policy, value, end, groups = load_data(N, planes, moves)
    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    for epoch in range(1, epochs + 1):
        print('Epoch', epoch)
        golois.getBatch(input_data, policy, value, end, groups, epoch * N)
        model.fit(input_data, {'policy': policy, 'value': value}, epochs=1, batch_size=batch_size)
        if epoch % 5 == 0:
            gc.collect()
        if epoch % 20 == 0:
            golois.getValidation(input_data, policy, value, end)
            val = model.evaluate(input_data, [policy, value], verbose=0, batch_size=batch_size)
            print("Validation metrics:", val)
            model.save('test.h5')

    return model"""


class CustomCallback(tf.keras.callbacks.Callback):
    def __init__(self, input_data, policy, value, end, groups, N, batch_size):
        super(CustomCallback, self).__init__()
        self.input_data = input_data
        self.policy = policy
        self.value = value
        self.end = end
        self.groups = groups
        self.N = N
        self.batch_size = batch_size

    def on_epoch_end(self, epoch, logs=None):
        golois.getBatch(self.input_data, self.policy, self.value, self.end, self.groups, (epoch + 1) * self.N)

        if (epoch + 1) % 5 == 0:
            gc.collect()

        if (epoch + 1) % 20 == 0:
            golois.getValidation(self.input_data, self.policy, self.value, self.end)
            val = self.model.evaluate(self.input_data, [self.policy, self.value], verbose=0, batch_size=self.batch_size)
            print("Validation metrics:", val)
            self.model.save('test.h5')


def train_model(epochs, batch_size, N, planes, moves, filters):
    go = GoModel(planes, filters)
    model = go.build()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.005, momentum=0.9),
                  loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                  loss_weights={'policy': 1.0, 'value': 1.0},
                  metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

    input_data, policy, value, end, groups = load_data(N, planes, moves)
    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    # TensorBoard callback
    log_dir = os.path.join("logs", "fit", datetime.now().strftime("%Y%m%d-%H%M%S"))
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

    # Custom callback
    custom_callback = CustomCallback(input_data, policy, value, end, groups, N, batch_size)

    model.fit(input_data, {'policy': policy, 'value': value}, epochs=epochs, batch_size=batch_size,
              callbacks=[tensorboard_callback, custom_callback])

    return model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, required=True, help='Path to config.json file')
    args = parser.parse_args()

    # Load configuration from JSON file
    with open(args.config, 'r') as f:
        config = json.load(f)

    epochs = config.get('epochs', 100)
    batch = config.get('batch', 256)
    N = config.get('N', 10000)
    planes = config.get('planes', 31)
    moves = config.get('moves', 361)
    filters = config.get('filters', 16)

    model = train_model(epochs, batch, N, planes, moves, filters)
    model.save('test.h5')
