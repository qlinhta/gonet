import tensorflow as tf
import tensorflow.keras as keras
import numpy as np
from tensorflow.keras import layers
from tensorflow.keras import regularizers
import gc
from tensorflow.keras.optimizers.schedules import CosineDecay

import golois

planes = 31
moves = 361
N = 10000
epochs = 100
batch = 128
filters = 32
dropout_rate = 0.1
trunk = 128
blocks = 5
learning_rate = 0.0001

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


def se_block(input_tensor, filters, ratio=16):
    se_shape = (1, 1, filters)
    se = layers.GlobalAveragePooling2D()(input_tensor)
    se = layers.Reshape(se_shape)(se)
    se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
    se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
    x = layers.multiply([input_tensor, se])
    return x


def mobilenet(x, expand, squeeze):
    x = layers.Conv2D(expand, (1, 1), kernel_regularizer=regularizers.l2(0.0001), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.DepthwiseConv2D((3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), use_bias=False)(
        x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = se_block(x, expand)
    x = layers.Conv2D(squeeze, (1, 1), kernel_regularizer=regularizers.l2(0.0001), use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    return layers.Add()([x, x])


input = keras.Input(shape=(19, 19, planes), name='board')
x = layers.Conv2D(trunk, 1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(input)
x = layers.BatchNormalization()(x)
x = layers.ReLU()(x)
for _ in range(blocks):
    x = mobilenet(x, filters, trunk)

policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                            kernel_regularizer=regularizers.l2(0.0001))(x)
policy_head = layers.Flatten()(policy_head)
policy_head = layers.Activation('softmax', name='policy')(policy_head)

value_head = layers.GlobalAveragePooling2D()(x)
value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(
    value_head)

model = keras.Model(inputs=input, outputs=[policy_head, value_head])

model.summary()

lr_schedule = CosineDecay(initial_learning_rate=0.0005, decay_steps=32000)
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
    if (i % 20 == 0):
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data,
                             [policy, value], verbose=0, batch_size=batch)
        print("val =", val)
        model.save(f"models/LyonGo_{i}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.h5")
