import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.utils import to_categorical

df = pd.read_csv('Homer_Bart_dataset/personagens.csv')

print(df.head())

#separar características (colunas de cores) e classe: Bart ou Homer
X = df.drop(columns=['classe']) 
y = df['classe'] 

# Codificar o rótulo em valores numéricos
label_encoder = LabelEncoder()
y_codificado = label_encoder.fit_transform(y)
y_categorico = to_categorical(y_codificado)

X_treinamento, X_teste, y_treinamento, y_teste = train_test_split(X, y_categorico, test_size=0.25, random_state=42)

#rede neural densa
rede_neural = Sequential()
rede_neural.add(Dense(units=64, activation='relu', input_shape=(X.shape[1],)))
rede_neural.add(Dense(units=32, activation='relu'))
rede_neural.add(Dropout(0.2))
rede_neural.add(Dense(units=2, activation='softmax'))  #2 classes: Bart e Homer

rede_neural.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

rede_neural.fit(X_treinamento, y_treinamento, epochs=50, batch_size=16, validation_data=(X_teste, y_teste))

loss, accuracy = rede_neural.evaluate(X_teste, y_teste)
print(f'Acurácia no conjunto de teste: {accuracy * 100:.2f}%')
