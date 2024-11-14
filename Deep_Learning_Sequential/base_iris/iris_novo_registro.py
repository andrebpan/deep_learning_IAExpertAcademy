import numpy as np
from tensorflow.keras.models import model_from_json

arq = open('base_iris/classificador_iris.json', 'r')
estrutura_rede = arq.read()
arq.close()
clf = model_from_json(estrutura_rede)

clf.load_weights('base_iris/classificador_iris.weights.h5')

novo_registro = np.array([[5.1, 3.5, 1.4, 0.2]])

#classificar o novo registro
previsao = clf.predict(novo_registro)
classe_prevista = np.argmax(previsao)

print(f"Classe prevista para o novo registro: {classe_prevista}")
