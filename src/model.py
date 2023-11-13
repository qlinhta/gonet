import tensorflow as tf
from tensorflow.keras import layers, regularizers, Model
from tensorflow.keras.optimizers.schedules import CosineDecay


class ClassicGo:
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


class LyonGo:
    def __init__(self, planes, filters, trunk, blocks):
        self.input_layer = layers.Input(shape=(19, 19, planes), name='input')
        self.filters = filters
        self.trunk = trunk
        self.blocks = blocks
        self.model = self.build()

    def se_block(self, input_tensor, filters, ratio=16):
        # Squeeze and Excitation block implementation
        se_shape = (1, 1, filters)
        se = layers.GlobalAveragePooling2D()(input_tensor)
        se = layers.Reshape(se_shape)(se)
        se = layers.Dense(filters // ratio, activation='relu', use_bias=False)(se)
        se = layers.Dense(filters, activation='sigmoid', use_bias=False)(se)
        x = layers.multiply([input_tensor, se])
        return x

    def mobilenet(self, x, expand, squeeze):
        x = layers.Conv2D(expand, (1, 1), kernel_regularizer=regularizers.l2(0.0001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.DepthwiseConv2D((3, 3), padding='same', kernel_regularizer=regularizers.l2(0.0001), use_bias=False)(
            x)
        x = layers.BatchNormalization()(x)
        x = layers.Activation('relu')(x)
        x = self.se_block(x, expand)
        x = layers.Conv2D(squeeze, (1, 1), kernel_regularizer=regularizers.l2(0.0001), use_bias=False)(x)
        x = layers.BatchNormalization()(x)
        return layers.Add()([x, x])

    def build(self):
        x = layers.Conv2D(self.trunk, 1, padding='same', kernel_regularizer=regularizers.l2(0.0001))(self.input_layer)
        x = layers.BatchNormalization()(x)
        x = layers.ReLU()(x)
        for _ in range(self.blocks):
            x = self.mobilenet(x, self.filters, self.trunk)

        policy_head = layers.Conv2D(1, 1, activation='relu', padding='same', use_bias=False,
                                    kernel_regularizer=regularizers.l2(0.0001))(x)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Activation('softmax', name='policy')(policy_head)

        value_head = layers.GlobalAveragePooling2D()(x)
        value_head = layers.Dense(50, activation='relu', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = layers.Dense(1, activation='sigmoid', name='value', kernel_regularizer=regularizers.l2(0.0001))(
            value_head)

        model = Model(inputs=self.input_layer, outputs=[policy_head, value_head])
        return model

    def summary(self):
        return self.model.summary()

    def save_model(self, file_path):
        self.model.save(file_path)


class ParisGo:
    def __init__(self, planes, filters, steps, initial_learning_rate):
        self.input_layer = layers.Input(shape=(19, 19, planes), name='input')
        self.filters = filters
        self.planes = planes
        self.lr_schedule = CosineDecay(initial_learning_rate, steps)

        self.model = self.build()

    def build(self):
        x = self.mixnet_block(self.input_layer, self.filters)

        for _ in range(4):
            x = self.mixnet_block(x, self.filters)

        policy_head = self.create_policy_head(x)
        value_head = self.create_value_head(x)

        model = Model(inputs=self.input_layer, outputs=[policy_head, value_head])
        return model

    def mixnet_block(self, input_tensor, filters):
        # BatchNorm
        branch_a = layers.Conv2D(filters, 1, activation='swish', padding='same')(input_tensor)
        branch_a = layers.BatchNormalization()(branch_a)
        branch_b = layers.Conv2D(filters, (3, 3), activation='swish', padding='same')(input_tensor)
        branch_b = layers.BatchNormalization()(branch_b)
        branch_c = layers.Conv2D(filters, (5, 5), activation='swish', padding='same')(input_tensor)
        branch_c = layers.BatchNormalization()(branch_c)

        mixed = layers.Concatenate()([branch_a, branch_b, branch_c])
        mixed = layers.Conv2D(filters, 1, activation='swish', padding='same')(mixed)
        return mixed

    def create_policy_head(self, x):
        policy_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias=False,
                                    kernel_regularizer=regularizers.l2(0.0001))(x)
        policy_head = layers.Flatten()(policy_head)
        policy_head = layers.Activation('softmax', name='policy')(policy_head)
        return policy_head

    def create_value_head(self, x):
        value_head = layers.Conv2D(1, 1, activation='swish', padding='same', use_bias=False,
                                   kernel_regularizer=regularizers.l2(0.0001))(x)
        value_head = layers.Flatten()(value_head)
        value_head = layers.Dense(50, activation='swish', kernel_regularizer=regularizers.l2(0.0001))(value_head)
        value_head = layers.Dense(1, activation='sigmoid', name='value')(value_head)
        return value_head

    def summary(self):
        return self.model.summary()

    def save_model(self, file_path):
        self.model.save(file_path)
