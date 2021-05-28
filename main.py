from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as  tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

import numpy as np

# Вводим данные для обучения модели
dollar_q    = np.array([10, 20, 60, 100, 150, 220, 38], dtype=float)
euro_a = np.array([8.18, 16.36, 49.08, 81.80, 122.71, 179.97, 31.09], dtype=float)

for i,c in enumerate(dollar_q):
  print("{} долларов США = {} евро".format(c, euro_a[i]))

# Содаем модель
# Используем модель плотной сети (Dense-сеть),
# которая будет состоять из единственного слоя с еднственым нейроном

# Создаем слой l0 количесвто нейронов (units) равно 1,
# размерность входного параметра (input_shape) - единичное значение
# разменость входных данных = размерность всей модели

l0 = tf.keras.layers.Dense(units=1, input_shape=[1])

# Преобразуем слой в модель

model = tf.keras.Sequential([l0])

# Компилируем модель с функцией потерь и оптимизаций
# Функция потерь - среденквалратичная ошибка
# Для функции оптимизации параметр, коэфициент скорости ибучения, равен 0.1
# - это размер шага при корректировке внутренних значений переменных

model.compile(loss='mean_squared_error',
              optimizer=tf.keras.optimizers.Adam(0.1))

# Тренируем модель
# используем метод fit, первый аргумент - входные значения, второй арумент - желаемые выходные значения
# epochs - количество итераций цыкла обучения
# verbose - контроль уровня логирования

history = model.fit(dollar_q, euro_a, epochs=1000, verbose=False)
print("Завершили тренировку модели")

# Используем модель для предсказаний
print(model.predict([100.0]))

