import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical 
from sklearn.metrics import confusion_matrix
import numpy as np

base = pd.read_csv('base_iris/datasets/iris.csv')
previsores = base.iloc[:, 0:4].values #selecionando apenas as colunas de valores, sem o target
classe = base.iloc[:, 4].values #somente a coluna target

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
#iris-setosa 1 0 0 
#iris-virginica 0 1 0
#iris-versicolor 0 0 1
classe_dummy = to_categorical(classe)

previsores_treinamento, previsores_teste, classe_treinamento, classe_teste = train_test_split(previsores, classe_dummy, test_size=0.25)

clf = Sequential()
#qtde neuronios na primeira camada => (4 + 3) / 2 = 3.5 ~ 4
clf.add(Dense(units=4, activation='relu', input_dim=4))
clf.add(Dense(units=4, activation='relu'))

clf.add(Dense(units=3, activation='softmax'))

clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])

#acuracia no treinamento
clf.fit(previsores_treinamento, classe_treinamento, batch_size=10, epochs=1000)

#acuracia na base de teste
resultado = clf.evaluate(previsores_teste, classe_teste)
print("\nresultado: ",resultado)

previsoes = clf.predict(previsores_teste)
previsoes = (previsoes > 0.5)
#percorre as 3 posições e retorna o indice aonde tem o resultado da classificação, o resultado 1
classe_teste2 = [np.argmax(t) for t in classe_teste]
previsoes2 = [np.argmax(t) for t in previsoes]

matriz = confusion_matrix(previsoes2, classe_teste2)
print("\nmatriz de confusao: ",matriz)