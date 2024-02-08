import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import gc
from prettytable import PrettyTable
import golois

planes = 31
moves = 361
N = 10000
epochs = 250
batch = 128
filters = 24
learning_rate = 0.005
dropout_rate = 0.0
decay_steps = N / batch * epochs
blocks = 3

table = PrettyTable()
table.field_names = ["Epoch", "Batch", "N", "Planes", "Moves", "Filters", "Learning Rate"]
table.add_row([epochs, batch, N, planes, moves, filters, learning_rate])
print(table)

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


def correct_pad(inputs, kernel_size):
    """Returns a tuple for zero-padding for 2D convolution with downsampling."""
    input_size = inputs.shape[1:3]
    adjust = (1, 1) if input_size[0] % 2 == 0 else (0, 0)
    return ((0, adjust[0]), (0, adjust[1]))


def MBConv(input, filters, kernel_size, strides, expand_ratio, se_ratio, dropout_rate):
    # Expansion phase (optional)
    x = layers.Conv2D(filters * expand_ratio, kernel_size=1, padding='same', use_bias=False)(input)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Depthwise convolution phase
    if strides == 2:
        x = layers.ZeroPadding2D(padding=correct_pad(x, kernel_size))(x)
    x = layers.DepthwiseConv2D(kernel_size, strides=strides, padding='same' if strides == 1 else 'valid',
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)

    # Squeeze and Excitation phase
    if se_ratio:
        se = layers.GlobalAveragePooling2D()(x)
        se = layers.Reshape((1, 1, filters * expand_ratio))(se)
        se = layers.Conv2D(int(filters * expand_ratio * se_ratio), kernel_size=1, activation='relu')(se)
        se = layers.Conv2D(filters * expand_ratio, kernel_size=1, activation='sigmoid')(se)
        x = layers.multiply([x, se])

    # Output phase
    x = layers.Conv2D(filters, kernel_size=1, padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    if input.shape[-1] == filters:
        x = layers.add([input, x])
    return x


def squeeze_excite_block(input, ratio=16):
    init = input
    channel_axis = -1
    filters = init.shape[channel_axis]
    se_shape = (1, 1, filters)

    se = layers.GlobalAveragePooling2D()(init)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', kernel_initializer='he_normal', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', kernel_initializer='he_normal', use_bias=False)(se)

    x = layers.multiply([init, se])
    return x


input = keras.Input(shape=(19, 19, planes), name='board')
x = layers.Conv2D(filters, 3, activation='relu', padding='same')(input)

for i in range(blocks):
    x = MBConv(x, filters=filters, kernel_size=3, strides=1, expand_ratio=6, se_ratio=0.25, dropout_rate=dropout_rate)
    x = squeeze_excite_block(x, ratio=16)

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

model.compile(optimizer=keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9),
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
    if (i % 20 == 0):
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data,
                             [policy, value], verbose=0, batch_size=batch)
        print("val =", val)
        model.save(f'models/ClassicGo_{i}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_val_{val[3]:.2f}.h5')
