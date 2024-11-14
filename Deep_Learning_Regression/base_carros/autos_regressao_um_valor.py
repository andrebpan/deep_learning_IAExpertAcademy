import pandas as pd
from sklearn.compose import ColumnTransformer
import tensorflow as tf
import keras
from sklearn.preprocessing import OneHotEncoder
from keras.models import Sequential
from keras.layers import Dense

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

#valores de preços inconsistentes
i1 = base.loc[base.price <= 10]
#print(i1)
base = base[base.price > 10]

i2 = base.loc[base.price > 350000]
#print(i2)
base = base.loc[base.price < 350000]

print(base.loc[pd.isnull(base['vehicleType'])])
base['vehicleType'].value_counts

print(base.loc[pd.isnull(base['gearbox'])])
base['gearbox'].value_counts

print(base.loc[pd.isnull(base['model'])])
base['model'].value_counts

print(base.loc[pd.isnull(base['fuelType'])])
base['fuelType'].value_counts

print(base.loc[pd.isnull(base['notRepairedDamage'])])
base['notRepairedDamage'].value_counts

valores = {'vehicleType': 'limousine',
           'gearbox': 'manuell',
           'model': 'golf',
           'fuelType': 'benzin',
           'notRepairedDamage': 'nein'}

base = base.fillna(value=valores)

print(base.isnull().sum())

X = base.iloc[:, 1:12].values
print(X)

y = base.iloc[:,0].values

base['brand'].value_counts()

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0,1,3,5,8,9,10])], remainder = 'passthrough')
X = onehotencoder.fit_transform(X).toarray()
print(X)

#estrutura da rede neural
regressor = Sequential(
    [tf.keras.layers.InputLayer(shape = (316, )),
    tf.keras.layers.Dense(units=158, activation = 'relu'),
    tf.keras.layers.Dense(units=158, activation = 'relu'),
    tf.keras.layers.Dense(units=1, activation = 'linear')
    ])

print(regressor.summary())

regressor.compile(loss='mean_absolute_error', optimizer='adam', metrics=['mean_absolute_error'])
regressor.fit(X, y, batch_size=300, epochs=100)

previsoes = regressor.predict(X)
print(previsoes)
print(y)
