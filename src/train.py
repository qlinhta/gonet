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
            # self.model.save(f'models/ParisGo_MixNet_Cosin_Swish_128_0.005_{val[3]:.2f}.h5')
            self.model.save(f'models/LyonGo_10K_128_5_0.0001_64_{val[3]:.2f}.h5')


def train_model(model_name, epochs, batch_size, N, planes, moves, filters):
    if model_name == "LyonGo":
        model = LyonGo(planes, filters, 128, 5).build()
    elif model_name == "ClassicGo":
        model = ClassicGo(planes, filters).build()
    elif model_name == "ParisGo":
        model = ParisGo(planes, filters, 1000, 0.005).build()
    else:
        raise ValueError(f"No model found with the name '{model_name}'.")

    model.summary()

    if model_name == "LyonGo":
        boundaries = [100, 150]
        values = [0.005, 0.0005, 0.00005]
        lr_schedule = tf.keras.optimizers.schedules.PiecewiseConstantDecay(boundaries, values)
        optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
    elif model_name == "ClassicGo":
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            initial_learning_rate=0.005, decay_steps=4000, decay_rate=0.9)
        optimizer = tf.keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9)
    elif model_name == "ParisGo":
        optimizer = tf.keras.optimizers.Adam(learning_rate=model.lr_schedule)

    model.compile(optimizer=optimizer,
                  loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                  loss_weights={'policy': 1.0, 'value': 1.0},
                  metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

    input_data, policy, value, end, groups = load_data(N, planes, moves)
    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    log_dir = os.path.join("logs", "fit", f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1, update_freq='epoch')

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
    model_name = config.get('model_name', "ClassicGo")

    table = PrettyTable()
    table.field_names = ["Parameter/Hyperparameter", "Value"]
    table.add_row(["Model", model_name])
    table.add_row(["Epochs", epochs])
    table.add_row(["Batch Size", batch])
    table.add_row(["N", N])
    table.add_row(["Planes", planes])
    table.add_row(["Moves", moves])
    table.add_row(["Filters", filters])
    print(table)

    model = train_model(model_name, epochs, batch, N, planes, moves, filters)