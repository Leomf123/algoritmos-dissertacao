# Teste de todo o processo com dados artificiais
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import random

from processar_rotulos import retirar_rotulos 
from utils import gerar_matriz_distancias
from algoritmos_adjacencias import gerar_matriz_adjacencias 
from algoritmos_peso import gerar_matriz_pesos
from algoritmos_classificar import propagar
from processar_rotulos import acuracia

random.seed(200)

dados, rotulos = make_blobs(n_samples=1000, cluster_std = 1 ,centers=3, n_features=2, random_state=0)

# Pré-processamento do rotulos para classificação semissupervisionada
# Primeiro: como o 0 nos rotulos vão representar que não tem rótulo tenho
# que mudar o rótulo de 0
# somando 1, ou seja, 0 será 1, 1 será 2, etc
rotulos = rotulos + 1

# classes
classes = []
for k in range(len(rotulos)):
  if not rotulos[k] in classes:
      classes.append(rotulos[k])

# Como eu preciso de rotulos faltando, que serão classificados
# retiro a quantidade dado uma porcentagem ( de 0 a 1 )
porcentagem_manter = 0.1
rotulos_semissupervisionado = retirar_rotulos(rotulos, porcentagem_manter,classes)

print("------rotulos faltando---------")
print(rotulos_semissupervisionado)

# Preciso transformar o vetor de rotulos numa matriz de 1 coluna
# pra ser processado pelos algoritmos
# Transformar o array em uma matriz onde cada elemento do array é uma linha
rotulos_semissupervisionado = rotulos_semissupervisionado.reshape(-1, 1)
#print(rotulos_semissupervisionado)


# Gerar o grafo com os dados
# print(dados)

print('----------Distancias------------------------------')
matriz_distancias = gerar_matriz_distancias(dados)
#print(matriz_distancias)


print('--------------Adjacencia-------------------')
matriz_adjacencias = gerar_matriz_adjacencias(matriz_distancias,2,'symKNN')
#print(matriz_adjacencias)

print('------------------Pesos----------------------------------')
sigma = 0.2
k = 2
matriz_pesos = gerar_matriz_pesos(matriz_adjacencias,matriz_distancias,sigma,k,"HM")
#print(matriz_pesos)

print('----------Propagação GRF------------------------------')
omega =  np.random.rand(len(classes),1)

parametro_regularizacao = 0.99

rotulos_propagados = propagar(matriz_pesos,rotulos_semissupervisionado,omega,parametro_regularizacao,"LGC")
print('Rotulos originais:')
print(rotulos)
print('Rotulos faltando:')
print(rotulos_semissupervisionado.reshape(1,-1)[0])
print('Rotulos Propagados:')
print(rotulos_propagados.reshape(1,-1)[0])

print('------------acuracia--------------')
print(acuracia(rotulos, rotulos_propagados.reshape(1,-1)[0],rotulos_semissupervisionado.reshape(1,-1)[0]))


plt.scatter(dados[:, 0], dados[:, 1], marker="o", c=rotulos, s=25)
plt.title('rotulos originais')
plt.colorbar(label='cor por classe')
plt.show()

plt.scatter(dados[:, 0], dados[:, 1], marker="o", c=rotulos_semissupervisionado.reshape(1,-1)[0], s=25)
plt.title('rotulos semissupervisionados')
plt.colorbar(label='cor por classe')
plt.show()

plt.scatter(dados[:, 0], dados[:, 1], marker="o", c=rotulos_propagados.reshape(1,-1)[0], s=25)
plt.title('rotulos propagados')
plt.colorbar(label='cor por classe')
plt.show()
