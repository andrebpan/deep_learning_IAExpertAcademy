import numpy as np
import pandas as pd
from keras.models import model_from_json

arq = open('base_breast_cancer/rede_salva/classificador_breast.json', 'r')
estrutura_rede = arq.read()
arq.close()

clf = model_from_json(estrutura_rede)
clf.load_weights('base_breast_cancer/rede_salva/classificador_breast.weights.h5')

novo_registro = np.array([[15.80, 8.34, 118, 900, 0.10, 0.26, 0.08, 0.134, 0.178,
                  0.20, 0.05, 1098, 0.87, 4500, 145.2, 0.005, 0.04, 0.05, 0.015,
                  0.03, 0.007, 23.15, 16.64, 178.5, 2018, 0.14, 0.185,
                  0.84, 158, 0.363]])

previsao = clf.predict(novo_registro)
print(previsao)
previsao = (previsao > 0.5)
print(previsao)

#avaliacao em uma base de dados de teste

previsores = pd.read_csv('base_breast_cancer/datasets/entradas_breast.csv')
classe = pd.read_csv('base_breast_cancer/datasets/saidas_breast.csv')

clf.compile(loss='binary_crossentropy', optimizer='adam',
            metrics=['binary_accuracy'])

resultado = clf.evaluate(previsores, classe)
print("\nresultado: ",resultado)
