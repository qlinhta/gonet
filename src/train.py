import tensorflow as tf
from model import GoModel
from dataloader import load_data
import golois
import gc


def train_model(epochs, batch_size, N, planes, moves, filters):
    go = GoModel(planes, filters)
    model = go.build()
    model.summary()
    model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=0.0005, momentum=0.9),
                  loss={'policy': 'categorical_crossentropy', 'value': 'binary_crossentropy'},
                  loss_weights={'policy': 1.0, 'value': 1.0},
                  metrics={'policy': 'categorical_accuracy', 'value': 'mse'})

    input_data, policy, value, end, groups = load_data(N, planes, moves)
    print("getValidation", flush=True)
    golois.getValidation(input_data, policy, value, end)

    for epoch in range(1, epochs + 1):
        print('Epoch', epoch)
        golois.getBatch(input_data, policy, value, end, groups, epoch * N)
        model.fit(input_data, {'policy': policy, 'value': value}, epochs=1, batch_size=batch_size)
        if epoch % 5 == 0:
            gc.collect()
        if epoch % 20 == 0:
            golois.getValidation(input_data, policy, value, end)
            val = model.evaluate(input_data, [policy, value], verbose=0, batch_size=batch_size)
            print("Validation metrics:", val)
            model.save('test.h5')

    return model


if __name__ == '__main__':
    epochs = 5
    batch = 128
    N = 10000
    planes = 31
    moves = 361
    filters = 16

    model = train_model(epochs, batch, N, planes, moves, filters)
    model.save('test.h5')