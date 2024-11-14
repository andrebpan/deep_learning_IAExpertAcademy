import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout

previsores = pd.read_csv('base_breast_cancer/datasets/entradas_breast.csv')
classe = pd.read_csv('base_breast_cancer/datasets/saidas_breast.csv')


#usando os melhores parametros encontrados
clf = Sequential()
clf.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal', input_dim = 30)) 
#dropout
clf.add(Dropout(0.2)) 
clf.add(Dense(units = 8, activation = 'relu', kernel_initializer = 'normal'))
#dropout
clf.add(Dropout(0.2))
clf.add(Dense(units = 1, activation = 'sigmoid'))
clf.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

clf.fit(previsores, classe, batch_size = 10, epochs = 100)

novo_registro = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = clf.predict(novo_registro)
print(previsao)
previsao = (previsao > 0.5)
print(previsao)


