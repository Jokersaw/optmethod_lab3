import time
import tracemalloc
import numpy as np

# Генерация полиномиальных признаков
def generate_random_polynomial_features(X, degree):
    n_samples, n_features = X.shape
    random_degrees = np.random.randint(1, degree + 1, size=n_features)
    X_poly = np.ones((n_samples, n_features))

    for feature in range(n_features):
        X_poly[:, feature] = X[:, feature] ** random_degrees[feature]

    return X_poly, random_degrees

def SGD(X, y, a0, b0, c0, learning_rate, cnt_max_iterations, batch_size, learning_rate_scheduling, regularization=None, lambda_=0.01, alpha=0.5):
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

        # используем среднеквадратичную функцию потерь
        # (производные у квадратичной функции потерь) -> чтобы получить градиент
        # расчет градиентов вектора и свободного скаляра
        coefs_grad = -2 * np.dot(X_batch.T, y_batch - y_exec) / batch_size # 2* batch_size    (...) - batch_size
        #Np.sum так как по факту скалярное умножение (1....1) * error
        c_grad = -2 * np.sum(y_batch - y_exec) / batch_size


        if regularization == 'L1':
            coefs_grad += lambda_ * np.sign(coefs)
        elif regularization == 'L2':
            coefs_grad += lambda_ * coefs
        elif regularization == 'Elastic':
            coefs_grad += alpha * lambda_ * np.sign(coefs) + (1 - alpha) * lambda_ * coefs

        # идем вдоль антиградиента
        coefs -= learning_rate * coefs_grad
        c -= learning_rate * c_grad

        # меняем шаг
        if learning_rate_scheduling == "exponential":
            learning_rate *= 0.999
        elif learning_rate_scheduling == "stepwise":
            if i % 200 == 0:
                learning_rate *= 0.8

    return coefs, c


# начало работы программы (время):
start = time.time()

# инициализация реальных коэффициентов
a_real = 3
b_real = 8
c_real = 50

polinomial = True
degree = 2
polinomInfo = None

# инициализация тренировочных данных
cnt_features = 1000
X = (np.random.rand(cnt_features, 2) * 2 - 1) * 3

if polinomial:
    X, polinomInfo = generate_random_polynomial_features(X, degree)

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
    for learning_rate_scheduling_start in [None, "exponential", "stepwise"]:
        for regularization_method in [None, 'L1', 'L2', 'Elastic']:
            # начинаем считать память
            tracemalloc.start()

            # запуск функции
            coefs_out, c_out = SGD(X, y, a_start, b_start, c_start, learning_rate, cnt_max_iterations, batch_size_start, learning_rate_scheduling_start)

            # фиксируем результаты вычислений памяти
            memory = tracemalloc.get_traced_memory()

            # заканчиваем считать память
            tracemalloc.stop()
            # конец работы программы (время):
            end = time.time()
        
            a_diff = abs(a_real - coefs_out[0])
            b_diff = abs(b_real - coefs_out[1])
            c_diff = abs(c_real - c_out)
            
            # вывод результатов
            print(f'cnt_features: {cnt_features}, learning rate: {learning_rate}, cnt_max_iterations: {cnt_max_iterations}')
            print(f'batch size: {batch_size_start}, learning rate scheduling: {learning_rate_scheduling_start}, regularization method: {regularization_method}')
            print(f'a_real: {a_real}, b_real: {b_real}, c_real: {c_real}')
            print(f'a_exec: {coefs_out[0]}, b_exec: {coefs_out[1]}, c_exec: {c_out}')

            print(f'a_diff: {a_diff}, b_diff: {b_diff}, c_diff: {c_diff}')
            print(f'count_function_operations: {10 * batch_size_start * cnt_max_iterations}\n')
            print(f'work time: {(end - start) * 10 ** 3} ms\nmemory: {memory[1]} bytes')
            print()

            start = time.time()