# House_Preci_Prediction.py
# Estas es una simple prediccion de de precios basado en el tamaño de la casa implementado con
# tensorflow

import tensorflow as tf
# numpy se utiliza para la informatica cientifica, utilizaremos su generador de numero aleatorios
# y las caracteristicas de conversiones de matriz.
import numpy as np
# math proporciona funciones matematicas.
import math
# matplotlib nos permite trazar y animar nuestros datos.
import matplotlib.pyplot as plt
import matplotlib.animation as animation

# generaremos algunos tamaños de casa aleatorios entre 1000 y 3500
num_house = 160
np.random.seed(42)
house_size = np.random.randint(low = 1000, high = 3500, size = num_house)

# generar precios de la casa desde el tamaño de la casa con un ruido aleatorio agregado
np.random.seed(42)
house_price = house_size * 100.0 + np.random.randint(low = 20000, high = 70000, size = num_house)

# Plot genera casa y tamaño
plt.plot(house_size, house_price, "bx")
plt.ylabel("Precio")
plt.xlabel("Tamaño")
plt.show()

# Docuemntacion

# Ahora podemos implementar nuestro modelo  de prediccion de precio de vivienda entrenando con algunos datos
# y ver que tan bien predice los precios. En tensorflow, para producir y entrenar nuestro modelo necesitamos
# 4 cosas datos que hemos limpiado y preparado para su uso  en la  capacitacion y en la evalucaion del
# desempeño  del modelo  capacitado.
# Conceptos:
# 1.- Una funcion de inferencia  que hace predicciones,
# 2.- Una funcion de inferencia que hace predicciones,
# 3.- una forma de medir la cantidad de las predicciones hechas llamamos la diferencia entre la perdida real
# de valores previstos.
# 4.- Finalmente, necesitamos un metodo para minimizar esta perdida optimizando los paramentros en el modelo
# con estos cuatro pasos en mente.

# Veamos lo que tenemos que condificar.(Implementacion)
# Los datos que usaremos son los datos que creamos con python aqui arriba, al saber el tamaño de la vivienda
# y sus precios asociados. Usaremos el 70% de estos datos para  entrenar a nuestro modelo y el 30% para
# probar nuestro modelo, nuestra funcion de inferencia sera la ecuacion de la linea que se ajusta  a los
# datos, es decir, el precio predictivo es igual a el factor de tamaño
# "Price = (sizeFactor * size) + priceOffset"
# La funcion de perdida  que usaremos para medir con exactitud, se predice que el precio sera el error
# cuadratico medio de la linea ajustada a traves de los datos. Un ajuste perfecto tendria cero error.
# Conseguiremos tan cerca como podemos a conseguir el error cero, ajustando repetidamente los valores para
# factorar, lo haremos utilizando una funcion de optimizando que encontrara los mejores valores para
# minimizar el error. Hay varios optimizadores disponibles en TensorFlow y para este modelo, vamos a utilizar
# el descenso gradiente que se utiliza a menudo en el aprendizaje de la maquina, vamos a llamar repetidamente
# al optimizador y cada ves el optimizador actualiza los valores con mejores valores para minimizar
# el error de estimacion, Ahora que conocemos estos pasos, codifiquemos estos elementos.
 
