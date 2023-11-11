import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


class MyModel(Model):
    def __init__(self, planes, filters):
        super(MyModel, self).__init__()
        self.conv1 = layers.Conv2D(filters, 1, activation='relu', padding='same')
        self.conv_blocks = [layers.Conv2D(filters, 3, activation='relu', padding='same') for _ in range(5)]
        self.policy_head_conv = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                              kernel_regularizer=regularizers.l2(0.0001))
        self.value_head_conv = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                             kernel_regularizer=regularizers.l2(0.0001))
        self.flatten = layers.Flatten()
        self.policy_head_activation = layers.Activation('softmax', name='policy')
        self.value_head_dense1 = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))
        self.value_head_dense2 = layers.Dense(1, activation='sigmoid', name='value',
                                              kernel_regularizer=regularizers.l2(0.0001))

    def call(self, inputs):
        x = self.conv1(inputs)
        for conv in self.conv_blocks:
            x = conv(x)
        policy_head = self.policy_head_activation(self.flatten(self.policy_head_conv(x)))
        value_head = self.value_head_dense2(self.value_head_dense1(self.flatten(self.value_head_conv(x))))
        return {'policy': policy_head, 'value': value_head}

    def summary(self):
        x = layers.Input(shape=(19, 19, 31))
        model = Model(inputs=[x], outputs=self.call(x))
        return model.summary()