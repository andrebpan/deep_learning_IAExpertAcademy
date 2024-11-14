import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('base_iris/datasets/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criar_rede(optimizer, kernel_initializer, activation, neurons, dropout_rate):
    clf = Sequential()
    clf.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer, input_dim=4))
    clf.add(Dropout(dropout_rate))
    clf.add(Dense(units=neurons, activation=activation, kernel_initializer=kernel_initializer))
    clf.add(Dropout(dropout_rate))
    clf.add(Dense(units=3, activation='softmax'))
    clf.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return clf

clf = KerasClassifier(build_fn=criar_rede)

parametros = {
    'batch_size': [10, 15],
    'epochs': [100, 200],
    'model__optimizer': ['adam', 'sgd'],
    'model__kernel_initializer': ['random_uniform', 'normal'],
    'model__activation': ['relu', 'tanh'],
    'model__neurons': [8, 16],
    'model__dropout_rate': [0.2, 0.3]
}

grid_search = GridSearchCV(estimator=clf, param_grid=parametros, cv=5)
grid_search = grid_search.fit(previsores, classe)

melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print("\nMelhores parâmetros: ", melhores_parametros)
print("\nMelhor precisão: ", melhor_precisao)