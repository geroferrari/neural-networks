# Red neuronal para obtener la ecuacion que relaciona
# los grados celsius con los grados farenheit
# Problema de Regresion, tenemos un numero como salida
#
# Red de una sola entrada y una sola salida
# 
#   (entrada)  --> (salida)
#
# Partimos de conocer los valores: 
#   Celsius (entrada)  | Farenheit (salida)
#      -40             |     -40
#      -10             |      14
#        0             |      32
#        8             |      46
#       15             |      59
#       22             |      72
#       38             |      100



import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

celsius = np.array([-40, -10, 0, 8, 15, 22, 38], dtype=float)
farenheit = np.array([-40, 14, 32, 46, 59, 72, 100], dtype=float)

capa = tf.keras.layers.Dense(units=1, input_shape=[1])
modelo = tf.keras.Sequential([capa])

modelo.compile(
    optimizer=tf.keras.optimizers.Adam(0.1),
    loss='mean_squared_error'
)

print("Entrenamos la red")
historial = modelo.fit(celsius,farenheit,epochs=1000, verbose=False)
print("Terminamos de entrenar la red")

# Se puede usar este grafico para entender cuantos ciclos se necesitaron para llegar al resultado correcto
# plt.xlabel("Numero de ejecución")
# plt.ylabel("Magnitud de perdida")
# plt.plot(historial.history["loss"])
# plt.show()

ejemplos = np.array([-18, 4, 25, 31, 40, 60, 85], dtype=float)
for i in ejemplos:
    print("Probamos convertir un valor de Celsius a Farenheit")
    Resultado = modelo.predict([i])
    print(f"{i} °C = {Resultado} °F")
    Error = abs(((i* 1.8 + 32) - Resultado)/100)
    print(f"Error de {Error}")

# Para ver como quedo compuesta la red:
print("Variables internas: (peso, sesgo)")
print(f"Peso: {capa.get_weights()[0]}, Sesgo:{capa.get_weights()[1]}")

print("Formula Teorica (°C --> °F) : F = C * 1.8 + 32")
print(f"Formula Obtenida (°C --> °F) : F = C *{capa.get_weights()[0]} + {capa.get_weights()[1]}")

 
