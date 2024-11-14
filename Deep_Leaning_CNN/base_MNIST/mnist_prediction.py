import tensorflow as tf
import keras
import matplotlib
import numpy as np
import pandas as pd
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer, Dense, Flatten, Dropout, Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras import utils as np_utils
import matplotlib.pyplot as plt

(X_treinamento, y_treinamento), (X_teste, y_teste) = mnist.load_data()

print(X_treinamento.shape, X_teste.shape)
#cada imagem possui 784 pixels

#observando uma imagem da base de dados
plt.imshow(X_treinamento[0], cmap='gray')
plt.title('Classe' + str(y_treinamento[0]))
#plt.show()

X_treinamento = X_treinamento.reshape(X_treinamento.shape[0],28,28,1)
X_teste = X_teste.reshape(X_teste.shape[0], 28, 28, 1)

X_treinamento = X_treinamento.astype('float32')
X_teste = X_teste.astype('float32')
X_treinamento /= 255
X_teste /= 255

y_treinamento = np_utils.to_categorical(y_treinamento, 10)
y_teste = np_utils.to_categorical(y_teste, 10)

#estrutura da rede neural convolucional
rede_neural = Sequential()
rede_neural.add(InputLayer(shape = (28, 28, 1)))#28 pixels de altura e largura e 1 canal(imagem em escala de cinza)
rede_neural.add(Conv2D(filters= 32, kernel_size=(3,3), activation='relu'))#camadas de convolução(32), kernel_size(tamanho do detector de caracteristicas)
rede_neural.add(BatchNormalization())
rede_neural.add(MaxPooling2D(pool_size=(2,2)))

rede_neural.add(Conv2D(filters= 32, kernel_size=(3,3), activation='relu'))#camadas de convolução(32), kernel_size(tamanho do detector de caracteristicas)
rede_neural.add(BatchNormalization())
rede_neural.add(MaxPooling2D(pool_size=(2,2)))

rede_neural.add(Flatten())

rede_neural.add(Dense(units=128, activation='relu'))#camada oculta
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units=128, activation='relu'))#camada oculta
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units=10, activation='softmax'))#camada de saida

#visualizando as etapas de convolução
print(rede_neural.summary())

rede_neural.compile(loss= 'categorical_crossentropy', optimizer='adam', metrics = ['accuracy'])
rede_neural.fit(X_treinamento, y_treinamento, batch_size=128,
                epochs=5, validation_data=(X_teste, y_teste))#enviando 128 imagens de cada vez, validation_data mede a acuracia na base de teste a cada época

#selecionando uma imagem 
indice_imagem = 0
imagem_teste = X_teste[indice_imagem]

#exibindo a imagem de teste
plt.imshow(imagem_teste.reshape(28, 28), cmap='gray')
plt.title('Imagem de teste')
plt.show()

predicao = rede_neural.predict(np.expand_dims(imagem_teste, axis=0))

#obtendo a classe prevista (índice da maior probabilidade)
classe_prevista = np.argmax(predicao)

print(f"A imagem pertence à classe: {classe_prevista}")