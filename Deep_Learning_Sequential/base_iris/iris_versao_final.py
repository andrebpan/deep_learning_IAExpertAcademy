import pandas as pd
from tensorflow.keras.models import Sequential, model_from_json
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.optimizers import SGD
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('base_iris/datasets/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)

def criar_rede_final():
    clf = Sequential()
    clf.add(Dense(units=16, activation='tanh', kernel_initializer='normal', input_dim=4))
    clf.add(Dropout(0.3))
    clf.add(Dense(units=16, activation='tanh', kernel_initializer='normal'))
    clf.add(Dropout(0.3))
    clf.add(Dense(units=3, activation='softmax'))
    
    otimizador = SGD(learning_rate=0.01)
    clf.compile(optimizer=otimizador, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return clf

clf = criar_rede_final()

clf.fit(previsores, classe, batch_size=10, epochs=200)

#Salvar o modelo da rede neural
clf_json = clf.to_json()
with open('classificador_iris.json', 'w') as json_file:
    json_file.write(clf_json)

# Salvar os pesos do modelo
clf.save_weights('classificador_iris.weights.h5')