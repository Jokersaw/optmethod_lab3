import time
import tensorflow as tf
import tracemalloc
import numpy as np
import matplotlib.pyplot as plt

# инициализация реальных коэффициентов
a_real = 3
b_real = 8
c_real = 50

# инициализация тренировочных данных
cnt_features = 1000
X_train = (np.random.rand(cnt_features, 2) * 2 - 1) * 3
y_train = (np.dot(X_train, [a_real, b_real]) + c_real + np.random.rand(1, cnt_features))[0]

# инициализация тестовых данных
cnt_test = 20
X_test = (np.random.rand(cnt_test, 2) * 2 - 1) * 3
y_test = (np.dot(X_test, [a_real, b_real]) + c_real + np.random.rand(1, cnt_test))[0]

# Обучение моделей с различными оптимизаторами
def train_model(optimizer, epochs=50, batch_size=32):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Dense(1, input_shape=(X_train.shape[1],), activation='linear')
    ])
    model.compile(optimizer=optimizer, loss='mse', metrics=[tf.keras.metrics.MeanSquaredError()])
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)
    y_pred = model.predict(X_test)

    mse = np.sum((y_test - [i[0] for i in y_pred]) ** 2) / len(y_test)

    return model, mse, history


learning_rate_sgd = 0.01
learning_rate_nesterov = 0.01
learning_rate_momentum = 0.01
learning_rate_adagard = 0.01
learning_rate_rmsprop = 0.01
learning_rate_adam = 0.01

epochs_sgd = 100
epochs_nesterov = 100
epochs_momentum = 100
epochs_adagard = 200
epochs_rmsprop = 100
epochs_adam = 100

batch_size_sgd = 100
batch_size_nesterov = 100
batch_size_momentum = 100
batch_size_adagard = 100
batch_size_rmsprop = 100
batch_size_adam = 100


def dump(model, name, time, memory, mse, learning_rate, cnt_max_iterations, batch_size):
    a_out = model.layers[0].get_weights()[0][0][0]
    b_out = model.layers[0].get_weights()[0][1][0]
    c_out = model.layers[0].get_weights()[1][0]
    print(f'--- {name} ---')
    print(f'cnt_features: {cnt_features}, learning rate: {learning_rate}, epochs: {cnt_max_iterations}')
    print(f'batch size: {batch_size}')
    print(f'a_real: {a_real}, b_real: {b_real}, c_real: {c_real}')
    print(f'a_exec: {a_out}, b_exec: {b_out}, c_exec: {c_out}')
    print(f'a_diff: {abs(a_real - a_out)}, b_diff: {abs(b_real - b_out)}, c_diff: {abs(c_real - c_out)}')
    print(f'mse: {mse}')
    print(f'work time: {time} ms\nmemory: {memory[1]} bytes')
    print(f'--- {name} ---')
    print()

# SGD
start = time.time()
tracemalloc.start()
sgd_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_sgd)
model_sgd, mse_sgd, history_sgd = train_model(sgd_optimizer, epochs_sgd, batch_size_sgd)
memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()
dump(model_sgd, "SGD", (end - start) * 10 ** 3, memory, mse_sgd, learning_rate_sgd, epochs_sgd, batch_size_sgd)

# Nesterov
start = time.time()
tracemalloc.start()
sgd_nesterov_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_nesterov, momentum=0.9, nesterov=True)
model_nesterov, mse_nesterov, history_nesterov = train_model(sgd_nesterov_optimizer, epochs_nesterov, batch_size_nesterov)
memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()
dump(model_nesterov, "NESTEROV", (end - start) * 10 ** 3, memory, mse_nesterov, learning_rate_nesterov, epochs_nesterov, batch_size_nesterov)

# Momentum
start = time.time()
tracemalloc.start()
sgd_momentum_optimizer = tf.keras.optimizers.SGD(learning_rate=learning_rate_momentum, momentum=0.9)
model_momentum, mse_momentum, history_momentum = train_model(sgd_momentum_optimizer, epochs_momentum, batch_size_momentum)
memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()
dump(model_momentum, "MOMENTUM", (end - start) * 10 ** 3, memory, mse_momentum, learning_rate_momentum, epochs_momentum, batch_size_momentum)

# AdaGrad
start = time.time()
tracemalloc.start()
adagrad_optimizer = tf.keras.optimizers.Adagrad(learning_rate=learning_rate_adagard)
model_adagrad, mse_adagrad, history_adagrad = train_model(adagrad_optimizer, epochs_adagard, batch_size_adagard)
memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()
dump(model_adagrad, "ADAGARD", (end - start) * 10 ** 3, memory, mse_adagrad, learning_rate_adagard, epochs_adagard, batch_size_adagard)

# RMSProp
start = time.time()
tracemalloc.start()
rmsprop_optimizer = tf.keras.optimizers.RMSprop(learning_rate=learning_rate_rmsprop)
model_rmsprop, mse_rmsprop, history_rmsprop = train_model(rmsprop_optimizer, epochs_rmsprop, batch_size_rmsprop)
memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()
dump(model_rmsprop, "RMSPROP", (end - start) * 10 ** 3, memory, mse_rmsprop, learning_rate_rmsprop, epochs_rmsprop, batch_size_rmsprop)

# Adam
start = time.time()
tracemalloc.start()
adam_optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_adam)
model_adam, mse_adam, history_adam = train_model(adam_optimizer, epochs_adam, batch_size_adam)
memory = tracemalloc.get_traced_memory()
tracemalloc.stop()
end = time.time()
dump(model_adam, "ADAM", (end - start) * 10 ** 3, memory, mse_adam, learning_rate_adam, epochs_adam, batch_size_adam)

# Визуализация процесса обучения
plt.figure(figsize=(10, 6))
plt.plot(history_sgd.history['loss'], label='SGD')
plt.plot(history_nesterov.history['loss'], label='Nesterov Momentum')
plt.plot(history_momentum.history['loss'], label='Momentum')
plt.plot(history_adagrad.history['loss'], label='AdaGrad')
plt.plot(history_rmsprop.history['loss'], label='RMSProp')
plt.plot(history_adam.history['loss'], label='Adam')
plt.xlabel('Epoch')
plt.ylabel('MSE')
plt.legend()
plt.title(f'MSE / EPOCHS')
plt.show()
