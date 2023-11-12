import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model


class GoModel:
    def __init__(self, planes, filters):
        # Define the input
        input_layer = layers.Input(shape=(19, 19, planes), name='input')

        # Convolutional layers
        x = layers.Conv2D(filters, 1, activation='relu', padding='same')(input_layer)
        for _ in range(5):
            x = layers.Conv2D(filters, 3, activation='relu', padding='same')(x)

        # Policy head
        policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                    kernel_regularizer=regularizers.l2(0.0001))(x)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Activation('softmax', name='policy')(policy_head)

        # Value head
        value_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                   kernel_regularizer=regularizers.l2(0.0001))(x)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)

        # Create the model
        self.model = Model(inputs=input_layer, outputs=[policy_head, value_head])

    def build(self):
        return self.model

    def summary(self):
        return self.model.summary()
