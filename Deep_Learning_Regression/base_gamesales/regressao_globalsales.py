import pandas as pd
import tensorflow as tf
from keras.layers import Dense, Dropout, Activation, Input
from keras.models import Model
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error

base = pd.read_csv('base_gamesales/datasets/vgsales.csv')

#pré-processamento
base = base.drop(['Name', 'NA_Sales', 'EU_Sales', 'JP_Sales', 'Other_Sales'], axis=1)
print(base.shape)

print(base.isnull().sum())
base = base.dropna(axis=0)
print(base.shape)

X = base.iloc[:, [1, 2, 3, 4]].values  # 'Platform', 'Year', 'Genre', 'Publisher'
y = base.iloc[:, 5].values  # 'Global_Sales'

onehotencoder = ColumnTransformer(transformers=[("OneHot", OneHotEncoder(), [0, 2, 3])], remainder='passthrough')
X = onehotencoder.fit_transform(X).toarray()
print(X.shape)

#criação da rede neural
camada_entrada = Input(shape=(620,))
#(620+3) / 2 
camada_oculta1 = Dense(units=312, activation='relu')(camada_entrada)
camada_oculta2 = Dense(units=312, activation='relu')(camada_oculta1)
camada_saida = Dense(units=1, activation='linear')(camada_oculta2)

regressor = Model(inputs=camada_entrada, outputs=camada_saida)
regressor.compile(optimizer='adam', loss='mse')

# Treinamento da rede neural
regressor.fit(X, y, epochs=500, batch_size=100)

previsao = regressor.predict(X)

#comparativo entre previsões e valores reais
print(previsao)
print(previsao.mean())
print(y)
print(y.mean())

print(mean_absolute_error(y, previsao))
