import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score

previsores = pd.read_csv('base_breast_cancer/datasets/entradas_breast.csv')
classe = pd.read_csv('base_breast_cancer/datasets/saidas_breast.csv')

def criarRede():
    clf = Sequential() 
    clf.add(Dense(units = 32, activation = 'relu', kernel_initializer = 'random_uniform', input_dim = 30))
    #dropout
    clf.add(Dropout(0.15))
    clf.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))
    clf.add(Dropout(0.15))
    clf.add(Dense(units=16, activation='relu', kernel_initializer='random_uniform'))
    clf.add(Dropout(0.15))
    clf.add(Dense(units = 1, activation = 'sigmoid'))

    otimizador = keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
    clf.compile(optimizer = otimizador, loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

    return clf

clf = KerasClassifier(build_fn=criarRede, epochs=150, batch_size=15)

resultados = cross_val_score(estimator=clf, X=previsores, y=classe, cv=10, scoring='accuracy')
print("\nResultados: ",resultados)

media = resultados.mean()
print("\nmedia dos resultados: ",media)

desvio = resultados.std()
print("Desvio padrao: ", desvio)
