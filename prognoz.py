import tensorflow as tf
import numpy as np
from tensorflow import keras

# Функция предсказания
def toy_model(Pred_Number2):
    #Количество дополнительных игрушек в игровом наборе
    Number1 = np.array([0.0, 1.0, 2.0, 3.0, 4.0, 6.0], dtype=float)
    #Цена дополнительных игрушек в игровом наборе
    Number2 = np.array([0.5, 1.0, 1.5, 2.0, 2.5, 3.5], dtype=float)
    model = tf.keras.Sequential([keras.layers.Dense(units=1, input_shape=[1])])
    model.compile(optimizer='sgd', loss='mean_squared_error')
    model.fit(Number1, Number2, epochs=600)
    return model.predict(Pred_Number2)[0]

#Предсказание цены за желаемое количество дополнительных игрушек при покупке игрового набора
print("Введите количество дополнительных игрушек для игрового набора, чтобы предсказать их цену (1 дополнительная игрушка -> + 1 в цене):")
Predict_Number=float(input())
prediction = toy_model([Predict_Number])
print("Цена: ", prediction)
