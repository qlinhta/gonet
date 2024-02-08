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

planes = 31
moves = 361
N = 10000
epochs = 250
batch = 128
filters = 40
learning_rate = 0.005
dropout_rate = 0.0
decay_steps = N / batch * epochs
blocks = 5

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


def swish(x):
    return x * tf.nn.sigmoid(x)


def residual_block(x, filters):
    shortcut = x
    y = Conv2D(filters, (3, 3), padding='same')(x)
    y = BatchNormalization()(y)
    y = swish(y)
    y = DepthwiseConv2D((5, 5), padding='same')(y)
    y = BatchNormalization()(y)
    y = swish(y)
    y = Conv2D(filters, 1, padding='same')(y)
    y = BatchNormalization()(y)
    y = Add()([shortcut, y])
    return y


input = keras.Input(shape=(19, 19, planes), name='board')
x = Conv2D(filters, (3, 3), padding='same')(input)
x = BatchNormalization()(x)
x = swish(x)
x = Conv2D(filters, (5, 5), padding='same')(x)
x = BatchNormalization()(x)
x = swish(x)
for _ in range(4):
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
checkpoint_path = './src/model.ckpt'
model.load_weights(checkpoint_path)
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
