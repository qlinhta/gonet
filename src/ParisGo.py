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
epochs = 250
batch = 128
filters = 16
dropout_rate = 0
learning_rate = 0.005
decay_steps = N/batch * epochs

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


def mixnet_block(input_tensor, filters):
    branch_a = layers.Conv2D(filters, 1, activation='swish', padding='same')(input_tensor)
    branch_a = layers.BatchNormalization()(branch_a)
    branch_b = layers.Conv2D(filters, (3, 3), activation='swish', padding='same')(input_tensor)
    branch_b = layers.BatchNormalization()(branch_b)
    branch_c = layers.Conv2D(filters, (5, 5), activation='swish', padding='same')(input_tensor)
    branch_c = layers.BatchNormalization()(branch_c)

    mixed = layers.Concatenate()([branch_a, branch_b, branch_c])
    mixed = layers.Conv2D(filters, 1, activation='swish', padding='same')(mixed)
    return mixed


input = keras.Input(shape=(19, 19, planes), name='board')
x = mixnet_block(input, filters)
x = layers.Dropout(dropout_rate)(x)

for _ in range(5):
    x = mixnet_block(x, filters)
    x = layers.Dropout(dropout_rate)(x)

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
    if (i % 20 == 0):
        golois.getValidation(input_data, policy, value, end)
        val = model.evaluate(input_data,
                             [policy, value], verbose=0, batch_size=batch)
        print("val =", val)
        model.save(f"models/ParisGo_{i}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.h5")
