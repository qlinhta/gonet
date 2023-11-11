import numpy as np
import golois
import tensorflow.keras as keras


def load_data(N, planes, moves):
    input_data = np.random.randint(2, size=(N, 19, 19, planes)).astype('float32')
    policy = np.random.randint(moves, size=(N,))
    policy = keras.utils.to_categorical(policy)
    value = np.random.randint(2, size=(N,)).astype('float32')
    end = np.random.randint(2, size=(N, 19, 19, 2)).astype('float32')
    groups = np.zeros((N, 19, 19, 1)).astype('float32')
    return input_data, policy, value, end, groups
