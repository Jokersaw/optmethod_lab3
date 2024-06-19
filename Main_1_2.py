import numpy as np

def SGD(X, y, a0, b0, c0, learning_rate, cnt_max_iterations, batch_size, learning_rate_scheduling):
    coefs = np.array([a0, b0])
    c = c0

    for i in range(cnt_max_iterations):

        # создаем массивы данных размером batch_size
        # простыми словами: случайным образом сгенерировали батч точек, которые собираемся рассматривать
        indexes = np.random.randint(0, len(X) - 1, batch_size)
        # реальные данные (аргументы , реальные значения от этих аргументов)
        X_batch = X[indexes]
        y_batch = y[indexes]

        # считаем значение для текущих коэффициентов
        y_exec = np.dot(X_batch, coefs) + c

        # используем квадратичную функцию потерь
        # квадратичная функция потерь = (линейная функция потерь) ^ 2
        # линейная ошибка (то_что_должно - то_что_получилось)
        # (производные у квадратичной функции потерь)
        # расчет градиентов вектора и свободного скаляра
        coefs_grad = -2 * np.dot(X_batch.T, y_batch - y_exec) / batch_size
        c_grad = -2 * np.sum(y_batch - y_exec) / batch_size

        # идем вдоль антиградиента
        coefs -= learning_rate * coefs_grad
        c -= learning_rate * c_grad

        # меняем шаг
        if learning_rate_scheduling == "exponential":
            learning_rate *= 0.999
        elif learning_rate_scheduling == "stepwise":
            if i % 100 == 0:
                learning_rate *= 0.7

    return coefs, c

# начало работы программы (время):
start = time.time()

# инициализация реальных коэффициентов
a_real = 3
b_real = 20
c_real = 100


# инициализация тренировочных данных
cnt_features = 100
X = (np.random.rand(cnt_features, 2) * 2 - 1) * 10
y = (np.dot(X, [a_real, b_real]) + c_real + np.random.rand(1, cnt_features))[0]

learning_rate = 0.01
cnt_max_iterations = 1000

# инициализация стартовых коэффициентов
a_start = np.random.random() * 10
b_start = np.random.random() * 10
c_start = np.random.random() * 10

# пробегаемся по всем значениям:
# 1) размер батча (часть от общего
# количества данных в процентах [0 соответствует размеру батча, равному 1])
# 2) функция изменения шага
for batch_size_part in [0, 30, 50, 100]:

    # реальный размер батча (считается как часть от общего количества данных)
    # [если batch_size_part равен 0, то размер батча должен быть равен 1]
    batch_size_start = max(1, cnt_features * batch_size_part // 100)

    for learning_rate_scheduling_start in ["none", "exponential", "stepwise"]:

        # запуск функции
        coefs_out, c_out = SGD(X, y, a_start, b_start, c_start, learning_rate, cnt_max_iterations, batch_size_start, learning_rate_scheduling_start)

        # конец работы программы (время):
        end = time.time()

        
        # вывод результатов
        print(f'cnt_features: {cnt_features}, learning rate: {learning_rate}, epochs: {epochs}')
        print(f'batch size: {batch_size_start}, learning rate scheduling: {learning_rate_scheduling_start}')
        print(f'a_real: {a_real}, b_real: {b_real}, c_real: {c_real}')
        print(f'a_exec: {coefs_out[0]}, b_exec: {coefs_out[1]}, c_exec: {c_out}')
        print(f'a_diff: {a_real - coefs_out[0]}, b_diff: {b_real - coefs_out[1]}, c_diff: {c_real - c_out}')
        print(f'count_function_runs: {9 * batch_size_start * epochs}\n')
        print(f'work time: {(end - start) * 10 ** 3} ms\nmemory: {memory[1]} bytes')
        print()

        # начало работы программы (время):
        start = time.time()







