import pandas as pd
from keras.models import Sequential
from keras.layers import Dense
from tensorflow.keras.utils import to_categorical 
from scikeras.wrappers import KerasClassifier
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import LabelEncoder

base = pd.read_csv('base_iris/datasets/iris.csv')
previsores = base.iloc[:, 0:4].values
classe = base.iloc[:, 4].values

labelencoder = LabelEncoder()
classe = labelencoder.fit_transform(classe)
#iris-setosa 1 0 0 
#iris-virginica 0 1 0
#iris-versicolor 0 0 1
classe_dummy = to_categorical(classe)

def criar_rede():
    clf = Sequential()
    
    clf.add(Dense(units=4, activation='relu', input_dim=4)) 
    clf.add(Dense(units=4, activation='relu'))

    clf.add(Dense(units=3, activation='softmax'))

    clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['categorical_accuracy'])
    
    return clf

clf = KerasClassifier(build_fn=criar_rede, epochs=1000, batch_size=10)

resultados = cross_val_score(estimator=clf, X=previsores, y=classe_dummy, cv=10, scoring='accuracy')

media = resultados.mean()
desvio = resultados.std()

print("\nmedia: ",media)
print("\ndesvio padrao: ", desvio)

