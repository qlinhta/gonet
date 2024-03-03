import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import gc
from prettytable import PrettyTable
import golois
from tensorflow.keras.layers import Input, Conv2D, DepthwiseConv2D, BatchNormalization, Add, ReLU, \
    GlobalAveragePooling2D, Dense, Activation, Multiply
import matplotlib.pyplot as plt

planes = 31
moves = 361
N = 10000
epochs = 500
batch = 256
filters = 40
learning_rate = 0.0005
dropout_rate = 0.0
decay_steps = N / batch * epochs
blocks = 5

table = PrettyTable()
table.field_names = ["Epoch", "Batch", "N", "Planes", "Moves", "Filters", "Learning Rate"]
table.add_row([epochs, batch, N, planes, moves, filters, learning_rate])
print(table)

train_losses, val_losses, train_acc, val_acc, train_mse, val_mse = [], [], [], [], [], []

input_data = np.random.randint(2, size=(N, 19, 19, planes))
input_data = input_data.astype('float32')

policy = np.random.randint(moves, size=(N,))
policy = keras.utils.to_categorical(policy)

value = np.random.randint(2, size=(N,))
value = value.astype('float32')

end = np.random.randint(2, size=(N, 19, 19, 2))
end = end.astype('float32')

groups = np.zeros((N, 19, 19, 1))
groups = groups.astype('float32')

print("getValidation", flush=True)
golois.getValidation(input_data, policy, value, end)


def residual_block(x, filters):
    shortcut = x
    y = Conv2D(filters, (3, 3), padding='same', use_bias=False)(x)
    y = BatchNormalization()(y)
    y = ReLU()(y)
    y = Conv2D(filters, (3, 3), padding='same', use_bias=False)(y)
    y = BatchNormalization()(y)
    y = Add()([shortcut, y])
    y = ReLU()(y)
    return y


input = keras.Input(shape=(19, 19, planes), name='board')
x = Conv2D(filters, (3, 3), padding='same')(input)
x = BatchNormalization()(x)
x = ReLU()(x)
for _ in range(blocks):
    x = residual_block(x, filters)

policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                           kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])
model.summary()

with open('ResNet.csv', 'w') as f:
    f.write("Epoch, Loss, Policy Loss, Value Loss, Policy Accuracy, Value MSE\n")

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=learning_rate,
    decay_steps=decay_steps,
    decay_rate=0.96,
    staircase=True)

model.compile(optimizer=keras.optimizers.SGD(learning_rate=lr_schedule, momentum=0.9),
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy': 1.0, 'value': 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

for i in range(1, epochs + 1):
    print('epoch ' + str(i))
    golois.getBatch(input_data, policy, value, end, groups, i * N)
    history = model.fit(input_data,
                        {'policy': policy, 'value': value},
                        epochs=1, batch_size=batch)
    if (i % 5 == 0):
        gc.collect()
    if (i % 10 == 0):
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data,
                             [policy, value], verbose=0, batch_size=batch)
        print("val =", val)
        train_losses.append(history.history['policy_loss'][0])
        val_losses.append(val[1])
        train_acc.append(history.history['policy_categorical_accuracy'][0])
        val_acc.append(val[3])
        train_mse.append(history.history['value_mse'][0])
        val_mse.append(val[4])

        with open('ResNet.csv', 'a') as f:
            f.write(
                f"{i},{history.history['loss'][0]},{history.history['policy_loss'][0]},{history.history['value_loss'][0]},{history.history['policy_categorical_accuracy'][0]},{history.history['value_mse'][0]}\n")

        model.save(f'models/ResNet_{i}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_val_{val[3]:.2f}.h5')

        fig, axs = plt.subplots(1, 3, figsize=(15, 5))
        axs[0].plot(train_losses, label='Train loss', color='lightcoral', linestyle='-', linewidth=2)
        axs[0].plot(val_losses, label='Validation loss', color='lightseagreen', linestyle='-.', linewidth=2)
        axs[0].set_title(f"Loss: {val[1]:.2f}")
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(train_acc, label='Train accuracy', color='lightcoral', linestyle='-', linewidth=2)
        axs[1].plot(val_acc, label='Validation accuracy', color='lightseagreen', linestyle='-.', linewidth=2)
        axs[1].set_title(f"Accuracy: {val[3]:.2f}")
        axs[1].legend()
        axs[1].grid()
        axs[2].plot(train_mse, label='Train MSE', color='lightcoral', linestyle='-', linewidth=2)
        axs[2].plot(val_mse, label='Validation MSE', color='lightseagreen', linestyle='-.', linewidth=2)
        axs[2].set_title(f"Mean Squared Error: {val[4]:.2f}")
        axs[2].legend()
        axs[2].grid()
        axs[0].set(xlabel='Every #10 Epoch')
        axs[1].set(xlabel='Every #10 Epoch')
        axs[2].set(xlabel='Every #10 Epoch')
        plt.tight_layout()
        plt.savefig(
            f"figures/ResNet_{i}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_val_{val[3]:.2f}.pdf")
        plt.close()