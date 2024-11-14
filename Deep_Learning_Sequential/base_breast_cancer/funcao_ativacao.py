import numpy as np

def stepFunction(soma):
    if(soma >= 1):
        return 1
    else:
        return 0
    

def sigmoidFunction(soma):
    return 1 / (1 + np.exp(-soma))

def tanhFunction(soma):
    return (np.exp(soma) - np.exp(-soma)) / (np.exp(soma) + np.exp(-soma))

def relu(soma):
    if soma >= 0:
        return soma
    else:
        return 0

def linearFunction(soma):
    return soma

def softmax(x):
    ex = np.exp(x)
    return ex / ex.sum()

teste = print(stepFunction(-1))
teste2 = print(sigmoidFunction(2.1))
teste3 = print(tanhFunction(2.1))
teste4 = print(relu(2.1))
teste5 = print(linearFunction(2.1))
valores = [7, 2, 1.3]
print(softmax(valores))