import pandas as pd
import tensorflow as tf
import sklearn
import numpy as np
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

#pré-processamento dos dados
base = pd.read_csv('base_gamesales/datasets/vgsales.csv')
print(base.head())
print(base.shape)

print(base.columns)
base = base.drop(['Global_Sales', 'Other_Sales'], axis=1)
print(base.shape)

#verificando valores nulos
print(base.isnull().sum())
base = base.dropna(axis=0)
print(base.shape)

#retirando jogos que aparecem apenas uma vez no dataset
base = base.drop('Name',axis=1)
print(base.shape)

#separação dos dados
print(base.columns)
X = base.iloc[:, [1,2,3,4]].values
y_na = base.iloc[:,5].values
y_eu = base.iloc[:,6].values
y_jp = base.iloc[:,7].values

#ps2 1 0 0 0 0...
#xbox360 0 1 0 0 0...
print(base['Platform'].value_counts())

#one-hot encoding nas colunas corretas
onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 2, 3])], remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()
print(X.shape)

#criação da rede neural
camada_entrada = Input(shape=(620,))
#(620+3) / 2 
camada_oculta1 = Dense(units=312, activation='relu')(camada_entrada)
camada_oculta2 = Dense(units=312, activation='relu')(camada_oculta1)
camada_saida1 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida2 = Dense(units=1, activation='linear')(camada_oculta2)
camada_saida3 = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=[camada_saida1, camada_saida2, camada_saida3])

regressor.compile(optimizer='adam', loss='mse')

regressor.fit(X, [y_na, y_eu, y_jp], epochs=500, batch_size=100)

previsao_na, previsao_eu, previsao_jp = regressor.predict(X)

#comparativo na
print(previsao_na)
print(previsao_na.mean())
print(y_na)
print(y_na.mean())

print(mean_absolute_error(y_na, previsao_na))

#comparativo eu
print(previsao_eu)
print(previsao_eu.mean())
print(y_eu)
print(y_eu.mean())

print(mean_absolute_error(y_eu, previsao_eu))

#comparativo jp
print(previsao_jp)
print(previsao_jp.mean())
print(y_jp)
print(y_jp.mean())

print(mean_absolute_error(y_jp, previsao_jp))

