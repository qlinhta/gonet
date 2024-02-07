import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import gc
from tensorflow.keras.optimizers.schedules import CosineDecay
from prettytable import PrettyTable
import golois
import matplotlib.pyplot as plt

plt.style.use('default')
plt.rc('text', usetex=False)
plt.rc('font', family='sans-serif')
plt.rc('font', size=14)
plt.rc('axes', titlesize=14)
plt.rc('axes', labelsize=14)
plt.rc('xtick', labelsize=14)
plt.rc('ytick', labelsize=14)
plt.rc('legend', fontsize=14)
plt.rc('lines', markersize=10)

planes = 31
moves = 361
N = 10000
epochs = 200
batch = 128
filters = 60
dropout_rate = 0
learning_rate = 0.005
decay_steps = N / batch * epochs

table = PrettyTable()
table.field_names = ["Epoch", "Batch", "N", "Planes", "Moves", "Filters", "Learning Rate",
                     "Dropout Rate", "Decay Steps"]
table.add_row([epochs, batch, N, planes, moves, filters, learning_rate, dropout_rate, decay_steps])
print(table)

train_losses = []
val_losses = []
train_acc = []
val_acc = []

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


def MBConvBlock(input_tensor, kernel_size, filters, mix_kernels, expansion_factor=6, stride=1, alpha=1.0):
    channel_axis = -1
    input_filters = input_tensor.shape[channel_axis]
    pointwise_filters = int(filters * alpha)

    x = input_tensor

    # Expand
    if expansion_factor != 1:
        expanded_filters = input_filters * expansion_factor
        x = layers.Conv2D(expanded_filters, 1, padding='same', use_bias=False)(x)
        x = layers.BatchNormalization(axis=channel_axis)(x)
        x = layers.ReLU()(x)

    # Depthwise
    if len(mix_kernels) > 1:
        mixed = []
        for kernel in mix_kernels:
            dw = layers.DepthwiseConv2D(kernel, strides=stride, padding='same', use_bias=False)(x)
            mixed.append(dw)
        x = layers.Concatenate(axis=channel_axis)(mixed)
    else:
        x = layers.DepthwiseConv2D(mix_kernels[0], strides=stride, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)
    x = layers.ReLU()(x)

    # Project
    x = layers.Conv2D(pointwise_filters, 1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization(axis=channel_axis)(x)

    # Skip connection and stride
    if stride == 1 and input_filters == pointwise_filters:
        x = layers.Add()([input_tensor, x])

    return x


alpha = 1.0
input = keras.Input(shape=(19, 19, planes), name='board')

x = MBConvBlock(input, kernel_size=[3], filters=16, mix_kernels=[3, 5], expansion_factor=1, alpha=alpha)
x = MBConvBlock(x, kernel_size=[3], filters=24, mix_kernels=[3, 5], stride=2, alpha=alpha)
x = MBConvBlock(x, kernel_size=[3], filters=24, mix_kernels=[3, 5], alpha=alpha)
x = MBConvBlock(x, kernel_size=[3], filters=40, mix_kernels=[3, 5, 7], stride=2, alpha=alpha)
x = MBConvBlock(x, kernel_size=[3], filters=40, mix_kernels=[3, 5, 7], alpha=alpha)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation='relu')(x)
x = layers.Dropout(0.3)(x)

policy_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Dropout(dropout_rate)(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)

value_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias=False,
                           kernel_regularizer=regularizers.l2(0.0001))(x)
value_head = layers.Flatten()(value_head)
value_head = layers.Dropout(dropout_rate)(value_head)
value_head = layers.Dense(50, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])
model.summary()
lr_schedule = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
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
        model.save(
            f"models/GoX_{i}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.h5")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(train_losses, label='Train loss', color='grey', linestyle='dashed', linewidth=1, marker='o',
                    markerfacecolor='grey', markersize=5)
        axs[0].plot(val_losses, label='Validation loss', color='black', linestyle='dashed', linewidth=1, marker='v',
                    markerfacecolor='black', markersize=5)
        axs[0].set_title(f"Validation loss: {val[1]:.2f}")
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(train_acc, label='Train accuracy', color='grey', linestyle='dashed', linewidth=1, marker='o',
                    markerfacecolor='grey', markersize=5)
        axs[1].plot(val_acc, label='Validation accuracy', color='black', linestyle='dashed', linewidth=1, marker='v',
                    markerfacecolor='black', markersize=5)
        axs[1].set_title(f"Validation accuracy: {val[3]:.2f}")
        axs[1].legend()
        axs[1].grid()
        axs[0].set(xlabel='Every #10 Epoch')
        axs[1].set(xlabel='Every #10 Epoch')
        plt.tight_layout()
        plt.savefig(
            f"figures/GoX_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.pdf")
        plt.close()
