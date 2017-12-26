# House_Preci_Prediction.py
# Estas es una simple prediccion de de precios basado en el tamaño de la casa implementado con
# tensorflow

import tensorflow as tf
# numpy se utiliza para la informatica cientifica, utilizaremos su generador de numero aleatorios
# y las caracteristicas de conversiones de matriz. Tiene que instalarlo con pip install numpy
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

# Tenemos los datos house_price y house_size definidos pero antes de usarlo, tenemos que hacer algunos datos
# de preparacion debemos garantizar que el precio y el tamaño estan en una escala similar. Para ello,
# crearemos una funcion para normalizar los datos, basado en los valores pasados adentro. entonces tenemos
# que dividir los datos  en entranamiento.

def normalize(array):
    return (array - array.mean()) / array.std()

# define el numero de formacion simple, 0.7 = 70%. Nosotros podemos tomar el 70% ya que el valor es aleatorio,
# el 70% de los datos se utilizan para mormar el modelo y el 30% se utiliza para ver que tan bueno es nuestro
# modelo.
num_train_samples = math.floor(num_house * 0.7)

# Tomando el 70% de los datos y etiquetarlos como datos de entrenamiento

# Definindo los datos en entranamiento
train_house_size = np.asarray(house_size[:num_train_samples])
train_price = np.asanyarray(house_price[:num_train_samples:])

train_house_size_norm = normalize(train_house_size)
train_price_norm = normalize(train_price)

# Datos de prueba del 30%
test_house_size = np.array(house_size[num_train_samples:])
test_house_price = np.array(house_price[num_train_samples:])

test_house_size_norm = normalize(test_house_size)
test_house_price_norm = normalize(test_house_price)

# Volvemos a nuestro cuadro de elementos que necesitabamos implementar, el primero es preparar la data, para
# usar los datos, necesitamos poner los datos en contenedores de tensores apropiados:

# Tipos de tensores
# 1.- constante: valor constante.
# 2.- variable: valores ajustablesen una grafica
# 3.- PlaceHolder: usapa para pasar datos dentro de la grafica

# ahora volvamos al codigo y veamoslos en accion, tenemos que definir los tensores que se pueden utilizar
# para pasar el precio y los datos del tamaño de la casa a la grafica, ¿Que tipo detensores deben de ser estos?
# Definimos un par de marcadores de posiscion de tensor para recibir datos.

# configurar los marcadores de posición tensorflow que se actualizan a medida que descendemos por el gradiente
tf_house_size = tf.placeholder("float", name="house_size")
tf_price = tf.placeholder("float", name="price")

# Como lo he dicho los marcadores de posiscion son tensores que se alimenta de datos cuando ejecutamos el
# grafico. Para hacer que nuestro modelo aprenda, haremos multiples pases atravesde los datos. Estos marcadores
# de posiscion se utilizan para pasar el precio, y datos del tamañode la casa alalgoritmo del optimizador. El
# algoritmo calcula entonces los valores que reducira la perdida, paremos aqui por un momento y permitanme
# señalar algunos puntos.En primer lugar, observe los paramentros de nombre en los constructores, estos nombres
# establecen el nombre de la operacion del tensor en el grafico de calculo. Asi que intento establecer el
# nombre de todos los tensores y operaciones.
# Segunda: debes recordar que cada vez que creamos en Python una variable que hace referencia a nuestro operador
# o variable de tensorflow, y esta es un nombre enlazado a un operador de TensorFlow o tensor. Es decir se
# refiere a una ubicacion de memoria que contien el objeto, si usted a programado en c o c+, piense en el
# puntero, o si eres de la vieja escuela en Python, es un nombre vinculado a un objeto, y si no esta
# familiarizado con las referencias espero esto haya quedado claro para usted, para hacer todo esto un poco mas
# claro, vamos aagregar una grafica, mostrando como cambia nuestra grafica de calculo.
# En este momento tenemos dos marcadores de posicion definidos en la grafica, con el precio y el tamaño de la
# casa, ahora definimos el tamaño del factor y el precio a compensar.
# Ahora definimos nuestra variables size_factor y price_offset, tensores que vamos a entrenar. Se definen
# como variable tensoras porque sus valores cambian a medida que progrese la capacitacion.

# Definir la variables que participan en el tamaño y precio que establecemos durante la formacion
# Inicializamos algunos valores random basados en la distribucion normal.
tf_size_factor = tf.Variable(np.random.randn(), name="size_factor")
tf_price_offset = tf.Variable(np.random.randn(), name="price_offset")

# despues de esto definimos nuestra funcion de inferencia, que predice el precio de la casa en funcion de la
# casa.

# Define la funcion para la prediccion de valores
# Nota, the uso de tensorflow sumar y multiplicar
# y, los metodos de tensorflow entienden como tratar
# metodos.
tf_price_pred = tf.add(tf.multiply(tf_size_factor, tf_house_size), tf_price_offset)

# aqui el conjunto de operadores es el precio previsto, es igual a size_factor por house_size mas price_offset.
# Tenga en cuenta que utilizamos los operadores TensorFlow para asegurarnos de que estamos singo claros en
# estas operaciones en el entorno de ejecucuon de tensorflow. En algunos casos, TensorFlow sabra usar una suma
# cuando ve un signo +.
# Advertencia: hay metodos similarmente nombrados.
# En otras librerias como numpy, tenga cuidado al utilizar sus metodos cuando queira usar un metodo TensorFlow
# Ahora vamos a definir la funcion de perdida que calcula el error cuadratico medio para los valores predichos
# y los valores reales en el tensor.

# Define la función de pérdida (cuánto error) - error cuadrático medio
tf_cost = tf.reduce_sum(tf.pow(tf_price_pred-tf_price, 2))/(2* num_train_samples)

# Observe lo que esto hace a nuestro grafico, tuvimos mas operadores  que leer  de los marcadores de posicion
# y de la posicion anterior añadir nuevos tensores y combinar todos ellos, con otras operciones y tensores
# y el grafico de compitacion crece. dejare de ilustrar estos cambios en el grafico, pero sabemos que cada
# tensor y operacion que defina en python se añade al grafico de calculo.

# Volviendo al codigo, vamos a definir learning_rate, que es el gradiente de paso. Esto es seguido por el
# optimizador que puede minimizar  el costo.

# Optimizar el rango de aprendizaje. El tamaño  de los pasos debajo de los gradientes
learning_rate = 0.1

# Definimos una gradiente descendente optimizador que minimizará la pérdida definida en la operación "costo"
optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(tf_cost)

# Estamos utilizando el optimizador de descenso de pendientes  de tensorflow, y pasamos la tasa de aprendizaje
# el costo para minimizar.
# En este momento tenemos nuetras variables tensorflow definida en el grafico, pero no existen en el entorno de
# ejecucion y no se ha fijado a los valores iniciales. Asi que si lo hacemos referencia. obtenemos un error
# indefinido. Para evitar esto, definimos la variable init y haga referencia a la operacion de la variable de
# tensorflow. Ahora podemos comenzar nuestra sesion

# Inicializando la variable
init = tf.global_variables_initializer()

# lanzando la grafica en la sesion
with tf.Session() as sess:
    sess.run(init)

# Observe que una vez usamos la instruccion with para definir el alcance de la sesion. Esto puede hacer que el
# codigo sea mas facil de leer y elimina  la necesidad  de cerrar explicitamente la sesion. SIn embargo, en
# algunos ambientes, esto puede ser un poco restrictivo ya que to do el codigo en la sesiondebe hacerse en un
# solo bloque de codigo. En estos casos, puede declarar una variable, haciendo referencia a esta sesion como
# hicimos cuando probamos que tensorflow estaba funcionando y pasar este objeto de sesion alrededor. Una ves que
# estamos en la sesion, Inicializamos nuestras variables con el comando de ejecucion de sesion. Pasamos como
# el codigo para ejecutar el codigo apuntando por init, que el metodo global_variables_initializer, a
# continuacion, establecemos unas constantes en python.

    # establecer con qué frecuencia mostrar el proceso de entrenamiento y el número de interacciones de
    # entrenamiento
    display_every = 2
    num_training_iter = 50

    # y en el bucle for, iteramos los datos de entrenamiento
    for iteration in range(num_training_iter):

        # adaptarse a todos los datos de entrenamiento, el metodo zip rellena los datos de entrenamiento
        # normalizados asi que pasamos el tamaño correcto de la casa y los pares de precios de la casa.
        # Ejecutamos el optimizador y le pasamos un diccionario, definiendo que el marcador de posicion
        # tf_house_size sera reeemplazado por los datos en x y el marcador de posicion tf_price seran
        # los datos en y
        for (x, y) in zip(train_house_size_norm, train_price_norm):
            sess.run(optimizer, feed_dict={tf_house_size: x, tf_price: y})

# el optimizador utiliza estos valores para dar el siguiente paso, medir la perdida y ajustar size_factor
# y price_offset para minizar la perdida.

# para seguir el proceso de informacion, nosotros exhibimos el coste periodicamente, ese es el error, que
# esperamos esta bajando, y la corriente. Valores entrenados para size_factor y price_offset

        # Mostrar el estado actual
        if (iteration + 1) % display_every == 0:
            c = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
            print("iteration #: ", '%04d' % (iteration + 1), "cost = ", "{:.9f}".format(c), \
                "size_factor = ", sess.run(tf_size_factor), "price_offset = ", sess.run(tf_price_offset))

    print("Optimizacion Finalizada!")
    training_cost = sess.run(tf_cost, feed_dict={tf_house_size: train_house_size_norm, tf_price: train_price_norm})
    print("Costo Entrenado=", training_cost,
        "size_factor=", sess.run(tf_size_factor),
        "price_offset=", sess.run(tf_price_offset)), '\n'
# cuando terminemos el bucle se muestra el costo final,  el ultimo factor de tamaño, y el ultimo price_offset

# trazamos los datos de entrenamiento, pruebas

    # diagrama de entrenamiento y datos de prueba, y regresión aprendida

    # obtener valores utilizados para normalizar los datos de modo que podamos desnormalizar los datos a su
    # escala original
    train_house_size_mean = train_house_size.mean()
    train_house_size_std = train_house_size.std()

    train_price_mean = train_price.mean()
    train_price_std = train_price.std()

    # trazar el gráfico
    plt.rcParams["figure.figsize"] = (10, 8)
    plt.figure()
    plt.ylabel("Precio")
    plt.xlabel("Tamaño (sq.ft)")
    plt.plot(train_house_size, train_price, 'go', label = "Datos de entrenamiento")
    plt.plot(test_house_size, test_house_price, 'mo', label = "Data de prueba")
    plt.plot(train_house_size_norm * train_house_size_std + train_house_size_mean,
        (sess.run(tf_size_factor) * train_house_size_norm + sess.run(tf_price_offset))
        * train_price_std + train_price_mean, label="Regresion Aprendida")

    plt.legend(loc='upper left')
    plt.show()

# Hemos trabajado con tensores pero no completamente descrito, repasemos la definicion de un tensor. Como
# dijimos antes un tensor es una matriz n-dimencional de datosy cada tensor tiene tres propiedades definitorias
# que son Rango, forma y tipo de datos, y es importante que entendamos estas propiedades, porque cuando se
# utilizan las operacion, porque cuando se utilizan las operaciones, usted deben asegurar sus datos en un
# tensro, es compatible a traves de estas propiedades, si no, la operacion fallara, o peor aun, completar
# y producir un tensor que no esperaba, echemos un vistazo a cada una de estas propiedades.
# 1.- Primero es el Rango: Este es el numero de dimenciones de un tensor, hablemos de las dimenciones. Ahora
# vamos a formalizar ese conocimiento, si el rango es cero, el tensor es un valor unico, un escalar, si el rango
# es uno el tensor es una lista de valores, un vector. Si el rango es dos, tengo una matriz con filas que contienen
# vectores, Si el rango es tres, tenemos una matriz vectorial como se muestra que es un cubo de datos cuando
# usted ve a tres o mas dimenciones usted puede apenas llamarlo n-tensor, donde n es el grado, y lo dejare a usted
# para averiguar como representar los datos.

# Ejemplo:
# Rango     Descripcion     Ejemplo
# 0         Scalar          s = 145
# 1         Vactor          v = [1, 3, 2, 5, 7]
# 2         Matriz          c = [ [[1,5,6], [5,3,4]], [[9,3,4], [3,4,9]] ]
# 3         3-Tensor(cube)  c = [[[1,5,6], [5,3,4]],[[9,3,5],[3,4,9]], [[4,3,2],[3,6,7]]]

# La propiedad siguiente es la forma. La forma es como se ve la informacion en el tensor en cualquier momento,
# este relacionado con el rango peor esta basado en el arreglo de datos en el tensor. Veamos algunos ejemplos
# para tener esto mas claro, un escala tiene un forma vacia, como se infica en los corchetes vacios, y un rango de
# 0, ya que tiene un solo valor como el vecto r el rango es uno, y tenemos una lista de valores, y la forma es el
# numero de valores de la lista. En este ejemplo, el tensor tiene cinco valores, asi que la forma es 5. Con un
# rango de 2 tenemos una matriz que tiene filas de vectores. En este ejemplo tenemos dos filas de vectores y 3
# valores en cada vector, asi que la forma es de 2,3. Con el rango 3 tenemos un cubo, y para este ejemplo, la forma
# es 3,2,3, bien, hemos hablado de la forma de datos, pero ¿que tipo pueden ser estos datos?

# Forma
# Forma de la data en el tensor. Relacionado al rango.

# Rango     Descripcion     Ejemplo                                                             Forma
# 0         escalar         s = 145                                                             []
# 1         vector          v = [1,3,2,5,7]                                                     [5]
# 2         matriz          m = [[1,5,6], [5,3,4]]                                              [2,3]
# 3         3-tensor(cubo)  c = [ [[1,5,6],[5,3,4]], [[9,3,5],[3,4,9]], [[4,3,2],[3,6,7]] ]     [3,2,3]

# Aqui esta la lista actual de tipos de datos que tensorflow soporta
# float32, float64
# int8, int16, int32, int64
# uint8, uint16
# string
# bool
# complex64, complex128
# qint8, qint16, quint8

# Vemos los tipos esperados, tenemos varios tamaños de numeros de punto flotante, enteros y entereson sin signo.
# Tambien tenemos cadena y booleano, incluso tenemos numero complejosy tambien tenemos numero cuantificados
# firmados y no firmados, si no estas familiarizado con la cuantificacion y numeros cuantificados, estos son los
# numeros que se han escalado para encajar en representaciones mas pequeñas por lo que ocupan menos bits, esta
# reduccion de tamañao  permite a TensorFlow para procesar mas rapido, a veces mucho mas rapido, en algunos casos
# esto puede disminuir el tiempo de procesamiento por el 75% o mas, y es otra caracteristica que nos permite
# escalar hasta tensorflow rendimiento de nivel de produccion, como un aparte, google has ido tan lejos como para
# construir plaas procesadoras especiales y chips llamados TPU. Para unidades de procesamiento TensorFlow que
# utiliza numeros cuantificados para mejorar el rendimiento, ahora conoce las propiedades basicas de los tensores
# en tensorflow, pos supuesto con cualquier buena bliblioteca, tensorflow tiene muchos metodos para permitirle
# trabajar con estas propiedades estos son algunos de los metodos mas utilizados,

# Metodos:
# get_shape() - returns shape
# reshape() - changes shape
# rank - returns rank
# dtype - return data type
# cast - changes data type

# como ves hay metodos que recuperan y alteran las formas de los tipos de datos, estos metodos son muy utiles y
# tendremos oportunidades para usarlos a lo largo de este curso, como recordamos, los valores de los tensores
# son lo que entrenamos cuando optimizamos nuestro modelo.


# En nuestro codigo anterior, utilizamos el descenso en gradiente como un optimizador, en el aprendizaje automatico
# el descenso gradual es una familia popular de metodos para ajustar los valores para reducir el error,
# conceptualmente, se puede pensar en el proceso de aptimiazcion como tratando de encontrar el camino mas empinado
# por una colina. Es decir, encontrar la direccion, que da el cambio mas rapido, comienza en algun punto x y
# calculas los parametros, que proporsiona el camino mas rapido por la colina, que es la reduccion mas rapida en
# la perdida. Toma un paso de tamaño fijo especifico por la tasa de aprendizaje en esa direccion. Desde el punto
# nuevo, se calcula un nuevo conjunto de parametros, que otra ves proporciona la direccion de la pendiente marcadores
# rapida. Esa es la reduccion mas rapida en la perdida y toma el proximo paso de tamaño fijo en la nueva direccion
# repite esto una y otra vez, en algun momento se llega a un minimo, y cuando ustedtoma medidas adicionales el
# valor de perdida ya no disminuye, alemnos asi es como deberia funcionar, si ajusta la tasa de aprendizaje
# demaciado alta, usted puede terminar rebotando alrededor con valores de perdida subiendo y bajando como si estubiese
# saltanto de un lado a otro de la gradiente, para evitar esto no ajustamos las tasa de aprendizaje demaciado alta
# Si estas familiarizado con el calculo, el calculo es la direccion de estos pasos, se determina calculando
# las derivadas de del modelo con respecto a sus caracteristicas, optimizadores basados en descenso de pendientes
# son un metodos frecuentemente usado para minimizar la perdida, y algo que veremos en futuros modelos de TensorFlow
# por lo tanto es importante que aprendamos como funciona, para entender mejor esto vamos a modificar el codigo


# Podemos ver el efecto del optimizador de pendiente de gradiente en accion en nuestro codigo, solo tenemos que hacer
# algunos pequeños cambios en nuestro codigo y añadir animacion con la linea de regresion ajustada. Añadimos
# variables en python para definir el numero de pasos,   
