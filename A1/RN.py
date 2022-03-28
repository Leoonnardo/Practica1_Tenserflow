from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
import csv

# Lectura del dataset con la libreria de CSV
def leerDataset():
    tempX = []
    tempY = []
    tempXY = []

    with open('gatoSets.csv', newline='') as File:  
        reader = csv.reader(File)
        auxLit = []
        auxLit1 = []
        tempX = []
        posicion1 = []
        posicion2 = []
        posicion3 = []
        posicion4 = []
        posicion5 = []
        posicion6 = []
        posicion7 = []
        posicion8 = []
        posicion9 = []
        Ydeseada = []
        for row in reader:
            for datos in row:
                auxLit.append(datos)

            # Añadimos las columnas del dataset en una lista individual
            posicion1.append(auxLit[0:9])
            # posicion2.append(auxLit[1])
            # posicion3.append(auxLit[2])
            # posicion4.append(auxLit[3])
            # posicion5.append(auxLit[4])
            # posicion6.append(auxLit[5])
            # posicion7.append(auxLit[6])
            # posicion8.append(auxLit[7])
            # posicion9.append(auxLit[8])
            Ydeseada.append(auxLit[9])
            auxLit = [] 

    # for i in range(0, 8):
    #     listaX = np.array()
    # tempX.append(np.array(posicion1, dtype=float))
    # tempX.append(np.array(posicion2, dtype=float))
    # tempX.append(np.array(posicion3, dtype=float))
    # tempX.append(np.array(posicion4, dtype=float))
    # tempX.append(np.array(posicion5, dtype=float))
    # tempX.append(np.array(posicion6, dtype=float))
    # tempX.append(np.array(posicion7, dtype=float))
    # tempX.append(np.array(posicion8, dtype=float))
    # tempX.append(np.array(posicion9, dtype=float))
    
    print(posicion1[0])
    print(Ydeseada)
    # Se meten los datos en un array para el manejo de los floats
    tempX = np.array(posicion1, dtype=float)
    tempY = np.array(Ydeseada, dtype=float)

    # Se juntan los arreglos de X y Y
    tempXY = [tempX, tempY]

    print(tempX)
    print(tempY)
    # Y se retorna
    return tempXY

def entrenamiento():
    #1
    # prediccion = [1,1,1,0,2,2,0,0,0]
    #2
    prediccion = [2,2,2,0,1,1,0,0,0]

    # Se obtiene el dataset retornado
    dataset = leerDataset()

    print("==== Inicia iteraciones de entrenamiento ====")
    modelo = implementacionKeras()

    # Compilar el algoritmo obteniendo la perdida cuadratica y realiza la optimizacion de que tanto aprendera el algoritmo
    modelo.compile(
        optimizer = keras.optimizers.Adam(0.1),
        loss='mean_squared_error'
    )

    # Se obtiene el historial del aprendizaje con base a las epocas o numero de iteraciones
    historial = modelo.fit(dataset[0], dataset[1], epochs=500)
    print(historial)

    # Se manda a llamar el interfaz que demostrara la magnitud de perdida y las epocas
    resultados(historial)

    # Demostracion de una prediccion con base a un numero
    print('Prediccion en base a un numero: ', prediccion)
    resultado = modelo.predict([prediccion])
    print('Resultado: ', resultado[0][0])

def implementacionKeras():

    # Red neuronal Keras
    # Funsion de activacion
    capa = keras.layers.Dense(units= 50, input_shape=[9], activation='sigmoid')
    capa1 = keras.layers.Dense(units= 50, activation='sigmoid')
    capa2 = keras.layers.Dense(units= 1, activation='linear')
    # Modelo de la grafica a utilizar
    modelo = keras.Sequential([capa, capa1, capa2])

    return modelo

def resultados(historial):
    plt.xlabel('Numero de epoca')
    plt.ylabel('Magnitud de perdida')
    plt.plot(historial.history['loss'])
 
    plt.show()

entrenamiento()