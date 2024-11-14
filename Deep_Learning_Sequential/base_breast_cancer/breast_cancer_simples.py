import pandas as pd
from sklearn.model_selection import train_test_split
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import confusion_matrix, accuracy_score

previsores = pd.read_csv('base_breast_cancer/datasets/entradas_breast.csv')
classe = pd.read_csv('base_breast_cancer/datasets/saidas_breast.csv')

print(previsores.head())
print(classe.head())

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe, test_size=0.25)

print("\nDistribuição dos dados apos o hold-out: ")

print(previsores_treinamento.shape)
print(previsores_teste.shape)
print(classe_treinamento.shape)
print(classe_teste.shape)

clf = Sequential()
#qtdeOculta = (30 + 1) / 2
clf.add(Dense(units = 16, activation = 'relu',
              kernel_initializer = 'random_uniform', input_dim = 30))
clf.add(Dense(units = 16, activation = 'relu', kernel_initializer = 'random_uniform'))

clf.add(Dense(units = 1, activation = 'sigmoid'))

otimizador = keras.optimizers.Adam(learning_rate = 0.001, decay = 0.0001, clipvalue = 0.5)
clf.compile(optimizer = otimizador, loss = 'binary_crossentropy',
          metrics = ['binary_accuracy'])

#clf.compile(optimizer = 'adam', loss = 'binary_crossentropy',
#           metrics = ['binary_accuracy'])

clf.fit(previsores_treinamento, classe_treinamento,
        batch_size = 10, epochs = 100)

pesos0 = clf.layers[0].get_weights()
print(pesos0)
print(len(pesos0))
pesos1 = clf.layers[1].get_weights()
pesos2 = clf.layers[2].get_weights()


print("\nMetricas no conjunto de testes:")
previsoes = clf.predict(previsores_teste)
previsoes = (previsoes > 0.5)
precisao = accuracy_score(classe_teste, previsoes)
print(f"Acuracia teste = {precisao}")

matriz = confusion_matrix(classe_teste, previsoes)
print(f"Matriz de confusao = {matriz}")

resultado = clf.evaluate(previsores_teste, classe_teste)
print(f"Avaliacao = {resultado}")



