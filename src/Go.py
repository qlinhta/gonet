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
table.field_names = ["LyonGo Blocks", "Epoch", "Batch", "N", "Planes", "Moves", "Filters", "Learning Rate",
                     "Dropout Rate", "Decay Steps"]
table.add_row([blocks, epochs, batch, N, planes, moves, filters, learning_rate, dropout_rate, decay_steps])
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


class Swish(layers.Layer):
    def call(self, inputs):
        return inputs * tf.nn.sigmoid(inputs)


class SqueezeAndExcite(layers.Layer):
    def __init__(self, input_channels, se_ratio=0.25, reduced_channels=None, **kwargs):
        super(SqueezeAndExcite, self).__init__(**kwargs)
        self.se_ratio = se_ratio
        self.reduced_channels = reduced_channels if reduced_channels else int(input_channels * self.se_ratio)

        self.se_reduce = layers.Conv2D(self.reduced_channels, kernel_size=1, activation='relu', use_bias=True)
        self.se_expand = layers.Conv2D(input_channels, kernel_size=1, activation='sigmoid', use_bias=True)

    def call(self, inputs):
        x = tf.reduce_mean(inputs, axis=[1, 2], keepdims=True)
        x = self.se_reduce(x)
        x = self.se_expand(x)
        return inputs * x


class MixNetBlock(layers.Layer):
    def __init__(self, in_channels, out_channels, dw_kernel_size, expand_ratio=1, se_ratio=0.25, stride=1, **kwargs):
        super(MixNetBlock, self).__init__(**kwargs)
        self.stride = stride
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.expand_ratio = expand_ratio
        self.se_ratio = se_ratio
        self.dw_kernel_size = dw_kernel_size

        self.use_residual = self.stride == 1 and in_channels == out_channels
        expanded_channels = in_channels * expand_ratio

        if expand_ratio != 1:
            self.expand_conv = layers.Conv2D(expanded_channels, 1, padding='same', use_bias=False)
            self.expand_bn = layers.BatchNormalization()
            self.expand_swish = Swish()

        self.dw_conv = layers.DepthwiseConv2D(dw_kernel_size, strides=stride, padding='same', use_bias=False,
                                              depth_multiplier=1)
        self.dw_bn = layers.BatchNormalization()
        self.dw_swish = Swish()

        if se_ratio > 0:
            self.se = SqueezeAndExcite(expanded_channels, se_ratio=se_ratio)

        self.project_conv = layers.Conv2D(out_channels, 1, padding='same', use_bias=False)
        self.project_bn = layers.BatchNormalization()

    def call(self, inputs):
        x = inputs
        if self.expand_ratio != 1:
            x = self.expand_conv(x)
            x = self.expand_bn(x)
            x = self.expand_swish(x)

        x = self.dw_conv(x)
        x = self.dw_bn(x)
        x = self.dw_swish(x)

        if self.se_ratio > 0:
            x = self.se(x)

        x = self.project_conv(x)
        x = self.project_bn(x)

        if self.use_residual:
            x = layers.add([x, inputs])

        return x


def create_mixnet_model(input_layer, blocks_config):
    input = input_layer
    x = layers.Conv2D(32, kernel_size=3, strides=1, padding='same', use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = Swish()(x)

    for bc in blocks_config:
        in_channels, out_channels, dw_kernel_size, expand_ratio, se_ratio, stride = bc
        x = MixNetBlock(in_channels, out_channels, dw_kernel_size, expand_ratio, se_ratio, stride)(x)

    policy_conv = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                kernel_regularizer=regularizers.l2(0.0001))(x)
    policy_flatten = layers.Flatten()(policy_conv)
    policy_output = layers.Activation('softmax', name='policy')(policy_flatten)

    value_x = layers.GlobalAveragePooling2D()(x)
    value_x = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_x)
    value_output = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(
        value_x)

    model = keras.Model(inputs=input, outputs=[policy_output, value_output])

    return model


blocks_config = [
    # (in_channels, out_channels, dw_kernel_size, expand_ratio, se_ratio, stride)
    (32, 64, 3, 1, 0.25, 1),
    (64, 128, 3, 6, 0.25, 2),
]

input_layer = layers.Input(shape=(19, 19, planes), name='board')
model = create_mixnet_model(input_layer, blocks_config)

model.compile(optimizer='adam',
              loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
              metrics={'policy': 'accuracy', 'value': 'mse'})

model.summary()

lr_schedule = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

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
            f"models/GoX_{i}_{blocks}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.h5")

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
            f"figures/GoX_{blocks}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.pdf")
        plt.close()
