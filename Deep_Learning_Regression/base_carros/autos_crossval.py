import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import keras
from keras import backend as k
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense
from sklearn.model_selection import cross_val_score
from sklearn import metrics
from scikeras.wrappers import KerasRegressor
import time

base = pd.read_csv('base_carros/datasets/autos.csv', encoding='ISO-8859-1')

#pré-processamento de dados
base = base.drop('dateCrawled', axis=1)
base = base.drop('dateCreated', axis=1)
base = base.drop('nrOfPictures', axis=1)
base = base.drop('postalCode', axis=1)
base = base.drop('lastSeen', axis=1)
base = base.drop('name', axis=1)
base = base.drop('seller', axis=1)
base = base.drop('offerType', axis=1)

base = base[base.price > 10]
base = base.loc[base.price < 350000]

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}

base = base.fillna(value=valores)

X = base.iloc[:, 1:12].values
y = base.iloc[:,0].values

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X).toarray()

print(X.shape)

def criar_rede():
    k.clear_session()
    regressor = Sequential([
        tf.keras.layers.InputLayer(shape = (316, )),
        tf.keras.layers.Dense(units=158, activation = 'relu'),
        tf.keras.layers.Dense(units=158, activation = 'relu'),
        tf.keras.layers.Dense(units=1, activation = 'linear')])
    regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
    return regressor

regressor = KerasRegressor(model=criar_rede, epochs=100, batch_size=300)

resultados = cross_val_score(estimator=regressor, X=X, y=y, cv=5, scoring='neg_mean_absolute_error')

print(abs(resultados))
print(abs(resultados.mean))

