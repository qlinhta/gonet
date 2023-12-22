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
N = 30000
epochs = 200
batch = 128
filters = 60
dropout_rate = 0
trunk = 128
blocks = 5
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
            f"models/LyonGo_{i}_{blocks}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.h5")

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
            f"figures/LyonGo_{blocks}_{epochs}_{batch}_{learning_rate}_{N}_{filters}_{dropout_rate}_val_{val[3]:.2f}.pdf")
        plt.close()
