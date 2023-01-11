# Clasificador de imagenes de ropa
# Problema de clasificación
#
# Al trabajar con imagenes en pixeles:
#    0  --> negro
#   255 --> blanco
#
# Vamos a usar imagenes de 28px x 28px (784px) como entrada
# 784 neuronas de entrada
#
# Supondremos 10 Categorias de salida
# 10 neuronas de salida
#
# 2 capas ocultas de 50 neuronas cada una
#
# Red neuronal convolucional
#
# 60 mil datos para entrenar la red

import tensorflow as tf
import tensorflow_datasets as tfds
import math
import matplotlib.pyplot as plt
import numpy as np

data, metadata = tfds.load('fashion_mnist', as_supervised=True, with_info=True)

print("Set de datos a usar", metadata)

data_entrenamiento= data['train']
data_pruebas = data['test']
num_ej_entrenamiento = metadata.splits['train'].num_examples
num_ej_prueba = metadata.splits['test'].num_examples

nombres_salidas = metadata.features['label'].names

# Normalizamos los datos de enrtada (0 - 255) --> (0-1)

def normalizar(imagenes, etiquetas):
    imagenes = tf.cast(imagenes, tf.float32)
    imagenes /= 255
    return imagenes, etiquetas

data_entrenamiento = data_entrenamiento.map(normalizar)
data_pruebas = data_pruebas.map(normalizar)

# Agregamos los datos a cache, asi usamos memoria en lugar de disco

data_entrenamiento = data_entrenamiento.cache()
data_pruebas = data_pruebas.cache()

# creamos el modelo
modelo = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28,28,1)),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(50, activation=tf.nn.relu),
    tf.keras.layers.Dense(10, activation=tf.nn.softmax)
])

# Compilamos el modelo
modelo.compile(
    optimizer='adam',
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy']
)

# usamos lotes para mejorar la velocidad de entrenamiento
lotes = 32

data_entrenamiento = data_entrenamiento.repeat().shuffle(num_ej_entrenamiento).batch(lotes)
data_pruebas = data_pruebas.batch(lotes)

# Entrenamos la red
historial = modelo.fit(data_entrenamiento, epochs=5, steps_per_epoch= math.ceil(num_ej_entrenamiento/lotes))

# Se puede usar este grafico para entender cuantos ciclos se necesitaron para llegar al resultado correcto
# plt.xlabel("Numero de ejecución")
# plt.ylabel("Magnitud de perdida")
# plt.plot(historial.history["loss"])
# plt.show()


# TODO: Probamos el algoritmo. Este script solo anda en google colab, si se ejecuta de manera local 
# surge un error con el cache.
#Pintar una cuadricula con varias predicciones, y marcar si fue correcta (azul) o incorrecta (roja)
for imagenes_prueba, etiquetas_prueba in data_pruebas.take(1):
  imagenes_prueba = imagenes_prueba.numpy()
  etiquetas_prueba = etiquetas_prueba.numpy()
  predicciones = modelo.predict(imagenes_prueba)
  
def graficar_imagen(i, arr_predicciones, etiquetas_reales, imagenes):
  arr_predicciones, etiqueta_real, img = arr_predicciones[i], etiquetas_reales[i], imagenes[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  
  plt.imshow(img[...,0], cmap=plt.cm.binary)

  etiqueta_prediccion = np.argmax(arr_predicciones)
  if etiqueta_prediccion == etiqueta_real:
    color = 'blue'
  else:
    color = 'red'
  
  plt.xlabel("{} {:2.0f}% ({})".format(nombres_salidas[etiqueta_prediccion],
                                100*np.max(arr_predicciones),
                                nombres_salidas[etiqueta_real]),
                                color=color)
  
def graficar_valor_arreglo(i, arr_predicciones, etiqueta_real):
  arr_predicciones, etiqueta_real = arr_predicciones[i], etiqueta_real[i]
  plt.grid(False)
  plt.xticks([])
  plt.yticks([])
  grafica = plt.bar(range(10), arr_predicciones, color="#777777")
  plt.ylim([0, 1]) 
  etiqueta_prediccion = np.argmax(arr_predicciones)
  
  grafica[etiqueta_prediccion].set_color('red')
  grafica[etiqueta_real].set_color('blue')
  
filas = 5
columnas = 5
num_imagenes = filas*columnas
plt.figure(figsize=(2*2*columnas, 2*filas))
for i in range(num_imagenes):
  plt.subplot(filas, 2*columnas, 2*i+1)
  graficar_imagen(i, predicciones, etiquetas_prueba, imagenes_prueba)
  plt.subplot(filas, 2*columnas, 2*i+2)
  graficar_valor_arreglo(i, predicciones, etiquetas_prueba)
  
  
# Esta red sirve para casos muy particulares, si se cambia la posicion de la prenda
# en la imagen, o el tamaño, o la posición, o incluso si las prendas no son parecidas
# a las de entrenamiento el test falla.
