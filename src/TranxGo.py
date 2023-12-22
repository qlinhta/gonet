import tensorflow as tf
import numpy as np
from tensorflow.keras import layers, regularizers, Model
import golois
import matplotlib.pyplot as plt
from prettytable import PrettyTable
from tensorflow import keras
import gc
from tensorflow.keras.optimizers.schedules import CosineDecay

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
epochs = 100
batch = 128
learning_rate = 0.001
num_heads = 4
num_transformer_blocks = 4
d_model = 32
dropout_rate = 0
decay_steps = N / batch * epochs
num_res_blocks = 3
res_filters = 32

table = PrettyTable()
table.field_names = ["Transformer Blocks", "Head", "Epoch", "Batch", "N", "Planes", "Moves", "D-Model", "Learning Rate",
                     "Dropout Rate", "Decay Steps"]
table.add_row(
    [num_transformer_blocks, num_heads, epochs, batch, N, planes, moves, d_model, learning_rate, dropout_rate,
     decay_steps])
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


def residual_block(x, filters):
    shortcut = x
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.Conv2D(filters, kernel_size=3, padding='same')(x)
    x = layers.BatchNormalization()(x)
    if shortcut.shape[-1] != filters:
        shortcut = layers.Conv2D(filters, kernel_size=1, padding='same')(shortcut)
    x = layers.Add()([x, shortcut])
    x = layers.ReLU()(x)
    return x


def feature_extractor(input_shape, num_blocks, filters):
    inputs = layers.Input(shape=input_shape)
    x = inputs
    for _ in range(num_blocks):
        x = residual_block(x, filters)
    return keras.Model(inputs, x)


def scaled_dot_product_attention(queries, keys, values, mask):
    product = tf.matmul(queries, keys, transpose_b=True)
    keys_dim = tf.cast(tf.shape(keys)[-1], tf.float32)
    scaled_product = product / tf.math.sqrt(keys_dim)
    if mask is not None:
        scaled_product += (mask * -1e9)
    attention = tf.nn.softmax(scaled_product, axis=-1)
    return tf.matmul(attention, values)


class MultiHeadAttention(tf.keras.layers.Layer):
    def __init__(self, num_heads, d_model):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        assert d_model % self.num_heads == 0
        self.depth = d_model // self.num_heads
        self.wq = tf.keras.layers.Dense(d_model)
        self.wk = tf.keras.layers.Dense(d_model)
        self.wv = tf.keras.layers.Dense(d_model)
        self.dense = tf.keras.layers.Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q, mask):
        batch_size = tf.shape(q)[0]
        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)
        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)
        scaled_attention = scaled_dot_product_attention(q, k, v, mask)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))
        output = self.dense(concat_attention)
        return output


def point_wise_feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model)
    ])


def transformer_block(x, num_heads, d_model, dff, rate=0.1):
    shape = tf.shape(x)
    batch_size, height, width, channels = shape[0], shape[1], shape[2], shape[3]
    x_flattened = tf.reshape(x, [batch_size, height * width, channels])

    attn_layer = MultiHeadAttention(num_heads, d_model)
    attn_output = attn_layer(x_flattened, x_flattened, x_flattened, None)
    attn_output = tf.keras.layers.Dropout(rate)(attn_output)

    attn_output = tf.reshape(attn_output, [batch_size, height, width, channels])
    out1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)(x + attn_output)
    ffn_output = point_wise_feed_forward_network(d_model, dff)(out1)
    ffn_output = tf.keras.layers.Dropout(rate)(ffn_output)
    return tf.keras.layers.LayerNormalization(epsilon=1e-6)(out1 + ffn_output)


input = layers.Input(shape=(19, 19, planes), name='board')
_resnet = feature_extractor(input.shape[1:], num_res_blocks, res_filters)
x = _resnet(input)
for _ in range(num_transformer_blocks):
    x = transformer_block(x, num_heads, d_model, d_model * 2)

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

# lr_schedule = CosineDecay(initial_learning_rate=learning_rate, decay_steps=decay_steps)
optimizer = tf.keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=0.0001)

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
            f"models/TranxGo_{i}_{num_transformer_blocks}_{num_heads}_{epochs}_{batch}_{N}_{planes}_{moves}_{d_model}_{learning_rate}_{dropout_rate}_{decay_steps}_val_{val[3]:.2f}.h5")


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
            f"figures/TranxGo_{i}_{num_transformer_blocks}_{num_heads}_{epochs}_{batch}_{N}_{planes}_{moves}_{d_model}_{learning_rate}_{dropout_rate}_{decay_steps}_val_{val[3]:.2f}.pdf")
        plt.close()
