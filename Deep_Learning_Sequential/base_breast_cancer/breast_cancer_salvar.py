import pandas as pd
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

# Salvar a arquitetura do modelo em JSON
clf_json = clf.to_json()
with open('classificador_breast.json', 'w') as json_file:
    json_file.write(clf_json)

# Salvar os pesos do modelo
clf.save_weights('classificador_breast.weights.h5')