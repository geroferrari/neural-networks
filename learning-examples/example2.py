# Red neuronal para obtener la ecuacion que relaciona
# los grados celsius con los grados farenheit
#
# Red de una sola entrada y una sola salida
# Agregamos dos capas intermedias
#
#          --> () --> () -->
# (entrada)--> () --> () --> (salida)
#          --> () --> () -->
# (la primer neurona de la capa intermedia tiene tres salidas)
# (la segunda neurona de la capa intermedia tiene tres entradas
#  correspondientes a cada salida de las neuronas de la capa anterior)
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

# input_shape nos ahorra de escribir la entrada
capa_intermedia_1 = tf.keras.layers.Dense(units=3, input_shape=[1])
capa_intermedia_2 = tf.keras.layers.Dense(units=3)
salida = tf.keras.layers.Dense(units=1)
modelo = tf.keras.Sequential([capa_intermedia_1, capa_intermedia_2, salida])

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
print(f"Capa Intermedia 1: Peso: {capa_intermedia_1.get_weights()[0]}, Sesgo:{capa_intermedia_1.get_weights()[1]}")
print(f"Capa Intermedia 2: Peso: {capa_intermedia_2.get_weights()[0]}, Sesgo:{capa_intermedia_2.get_weights()[1]}")
print(f"Salida: Peso: {salida.get_weights()[0]}, Sesgo:{salida.get_weights()[1]}")

print("En este caso no es posible recrear la formula de conversión, hay mas capas, deja de ser lineal")

 
