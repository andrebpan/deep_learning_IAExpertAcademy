import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf
from keras import backend as K
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from scikeras.wrappers import KerasRegressor

base = pd.read_csv('base_carros/datasets/autos.csv', encoding='ISO-8859-1')

#pré-processamento de dados
base = base.drop(['dateCrawled', 'dateCreated', 'nrOfPictures', 'postalCode', 
                  'lastSeen', 'name', 'seller', 'offerType'], axis=1)

base = base[base.price > 10]
base = base[base.price < 350000]

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}
base = base.fillna(value=valores)

X = base.iloc[:, 1:12].values
y = base.iloc[:,0].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])], remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()

def criar_rede(loss_function):
    K.clear_session()
    regressor = Sequential([
        tf.keras.layers.InputLayer(shape=(316,)),
        tf.keras.layers.Dense(units=158, activation='relu'),
        tf.keras.layers.Dense(units=158, activation='relu'),
        tf.keras.layers.Dense(units=1, activation='linear')
    ])
    regressor.compile(loss=loss_function, optimizer='adam')
    return regressor

#funções de erro a serem testadas
loss_functions = ['mean_squared_error', 'mean_absolute_error', 'mean_absolute_percentage_error', 
                  'mean_squared_logarithmic_error', 'squared_hinge']

resultados = {}

for loss_function in loss_functions:
    regressor = KerasRegressor(model=criar_rede, loss_function=loss_function, epochs=100, batch_size=300)
    scores = cross_val_score(estimator=regressor, X=X, y=y, cv=5, scoring='neg_mean_absolute_error')
    resultados[loss_function] = abs(scores.mean())

for loss_function, resultado in resultados.items():
    print(f"Loss Function: {loss_function} - Resultado: {resultado}")
