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
N = 50000
epochs = 500
batch = 256
dropout_rate = 0
learning_rate = 0.0005
decay_steps = N / batch * epochs

table = PrettyTable()
table.field_names = ["Epoch", "Batch", "N", "Planes", "Moves", "Learning Rate",
                     "Dropout Rate", "Decay Steps"]
table.add_row([epochs, batch, N, planes, moves, learning_rate, dropout_rate, decay_steps])
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


def SEBlock(input_tensor, ratio=16):
    channel_axis = -1
    filters = input_tensor.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    return layers.multiply([input_tensor, se])


def make_divisible(v, divisor, min_value=None):
    if min_value is None:
        min_value = divisor
    new_v = max(min_value, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v


def InvertedResidualBlock(x, expansion, filters, stride, alpha=1.0):
    in_channels = x.shape[-1]
    pointwise_conv_filters = int(filters * alpha)
    pointwise_filters = make_divisible(pointwise_conv_filters, 8)
    x = layers.Conv2D(expansion * in_channels, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU(max_value=6)(x)

    if stride == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, 3))(x)

    x = layers.DepthwiseConv2D(kernel_size=3, strides=stride, activation=None, use_bias=False,
                               padding='same' if stride == 1 else 'valid')(x)
    x = layers.BatchNormalization(axis=-1)(x)
    x = layers.ReLU(max_value=6)(x)

    x = layers.Conv2D(pointwise_filters, kernel_size=1, padding='same', use_bias=False, activation=None)(x)
    x = layers.BatchNormalization(axis=-1)(x)

    if in_channels == pointwise_filters and stride == 1:
        return layers.Add()([x, input_tensor])
    return x


def correct_pad(inputs, kernel_size):
    img_dim = 2
    input_size = tf.keras.backend.int_shape(inputs)[1:3]
    adjust = (1, 1) if input_size[0] is None else (input_size[0] % 2, input_size[1] % 2)
    correct = (kernel_size // 2, kernel_size // 2)
    return ((correct[0] - adjust[0], correct[0]), (correct[1] - adjust[1], correct[1]))


inputs = keras.Input(shape=(19, 19, planes), name='board')
x = InvertedResidualBlock(inputs, expansion=1, filters=16, stride=1)
x = SEBlock(x)
x = InvertedResidualBlock(x, expansion=6, filters=24, stride=2)
x = SEBlock(x)

x = layers.GlobalAveragePooling2D()(x)
x = layers.Reshape((1, 1, x.shape[1]))(x)
x = layers.Dropout(0.3)(x)

policy_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)
value_head = layers.GlobalAveragePooling2D()(x)
value_head = layers.Dense(50, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(value_head)

model = keras.Model(inputs=inputs, outputs=[policy_head, value_head])

lr_schedule = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.AdamW(learning_rate=lr_schedule)

model.compile(optimizer=optimizer,
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              loss_weights={'policy': 1.0, 'value': 1.0},
              metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

model.summary()

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
            f"models/ParisGo_{i}_{epochs}_{batch}_{learning_rate}_{N}_{dropout_rate}_val_{val[3]:.2f}.h5")

        fig, axs = plt.subplots(1, 2, figsize=(10, 5))
        axs[0].plot(train_losses, label='Train loss', color='grey', linestyle='dashed', linewidth=1, marker='o',
                    markerfacecolor='white', markersize=5)
        axs[0].plot(val_losses, label='Validation loss', color='black', linestyle='dashed', linewidth=1, marker='v',
                    markerfacecolor='white', markersize=5)
        axs[0].set_title(f"Validation loss: {val[1]:.2f}")
        axs[0].grid()
        axs[0].legend()
        axs[1].plot(train_acc, label='Train accuracy', color='grey', linestyle='dashed', linewidth=1, marker='o',
                    markerfacecolor='white', markersize=5)
        axs[1].plot(val_acc, label='Validation accuracy', color='black', linestyle='dashed', linewidth=1, marker='v',
                    markerfacecolor='white', markersize=5)
        axs[1].set_title(f"Validation accuracy: {val[3]:.2f}")
        axs[1].legend()
        axs[1].grid()
        axs[0].set(xlabel='Every #10 Epoch')
        axs[1].set(xlabel='Every #10 Epoch')
        plt.tight_layout()
        plt.savefig(
            f"figures/ParisGo_{i}_{epochs}_{batch}_{learning_rate}_{N}_val_{val[3]:.2f}.pdf")
        plt.close()
