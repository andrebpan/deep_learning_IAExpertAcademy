import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import GridSearchCV


previsores = pd.read_csv('base_breast_cancer/datasets/entradas_breast.csv')
classe = pd.read_csv('base_breast_cancer/datasets/saidas_breast.csv')

def criarRede(optimizer, loos, kernel_initializer, activation, neurons):
    clf = Sequential()
    #qtdeOculta = (30 + 1) / 2  
    clf.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer, input_dim = 30))
    
    #dropout
    clf.add(Dropout(0.2))
    
    clf.add(Dense(units = neurons, activation = activation, kernel_initializer = kernel_initializer))

    #dropout
    clf.add(Dropout(0.2))
    
    clf.add(Dense(units = 1, activation = 'sigmoid'))
    clf.compile(optimizer = optimizer, loss = loos, metrics = ['binary_accuracy'])

    return clf


clf = KerasClassifier(build_fn=criarRede)

parametros = {'batch_size':[10, 30],
              'epochs':[50, 100], 
              'model__optimizer':['adam', 'sgd'],
              'model__loos':['binary_crossentropy', 'hinge'], 
              'model__kernel_initializer':['random_uniform', 'normal'],
              'model__activation':['relu', 'tanh'],
              'model__neurons':[16, 8]}

grid_search = GridSearchCV(estimator=clf,
                           param_grid=parametros,
                           scoring='accuracy',
                           cv=5)

grid_search = grid_search.fit(previsores, classe)
melhores_parametros = grid_search.best_params_
melhor_precisao = grid_search.best_score_

print("\nMelhores parâmetros: ", melhores_parametros)
print("\nMelhor precisão: ", melhor_precisao)

