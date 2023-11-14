import tensorflow as tf
import json
import argparse
from model import ClassicGo, ParisGo, LyonGo
from dataloader import load_data
import golois
import gc
import os
from datetime import datetime
from prettytable import PrettyTable


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
            self.model.save(f'models/ParisGo_MixNet_Cosin_Swish_256_0.005_{val[3]:.2f}.h5')
            # self.model.save(f'models/LyonGo_128_5_0.00001_{val[3]:.2f}.h5')


def train_model(epochs, batch_size, N, planes, moves, filters):
    classic = ClassicGo(planes, filters)
    lyon = LyonGo(planes, filters, 128, 5)
    paris = ParisGo(planes, filters, 1000, 0.005)
    model = paris.build()
    model.summary()

    """lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
        initial_learning_rate=0.005,
        decay_steps=4000,
        decay_rate=0.9)"""

    """model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
                  loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                  loss_weights={'policy': 1.0, 'value': 1.0},
                  metrics={'policy': 'categorical_accuracy', 'value': 'mse'})"""

    """model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005),
                  loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                  loss_weights={'policy': 1.0, 'value': 1.0},
                  metrics={'policy': 'categorical_accuracy', 'value': 'mse'})"""

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=paris.lr_schedule),
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
    batch = config.get('batch', 128)
    N = config.get('N', 10000)
    planes = config.get('planes', 31)
    moves = config.get('moves', 361)
    filters = config.get('filters', 32)

    table = PrettyTable()
    table.field_names = ["Parameter/Hyperparameter", "Value"]
    table.add_row(["Epochs", epochs])
    table.add_row(["Batch Size", batch])
    table.add_row(["N", N])
    table.add_row(["Planes", planes])
    table.add_row(["Moves", moves])
    table.add_row(["Filters", filters])
    print(table)

    model = train_model(epochs, batch, N, planes, moves, filters)
