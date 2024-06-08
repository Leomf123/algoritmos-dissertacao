# Teste de todo o processo com dados artificiais
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs
import numpy as np
import random

from processar_rotulos import retirar_rotulos 
from utils import gerar_matriz_distancias, checar_matrix_adjacencias
from algoritmos_adjacencias import gerar_matriz_adjacencias 
from algoritmos_peso import gerar_matriz_pesos
from algoritmos_classificar import propagar
from processar_rotulos import acuracia
from utils import ordem_rotulos_primeiro, divisao_L
from processar_rotulos import one_hot

random.seed(100)

dados, rotulos = make_blobs(n_samples=100, cluster_std = 1 ,centers=3, n_features=2, random_state=0)

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
porcentagem_manter = 0.4
rotulos_semissupervisionado = retirar_rotulos(rotulos, porcentagem_manter, classes)

print("------rotulos faltando---------")
#print(rotulos_semissupervisionado)

# Gerar o grafo com os dados
# print(dados)

print('----------Distancias------------------------------')
medida_distancia = 'euclidean'
matriz_distancias = gerar_matriz_distancias(dados, dados, medida_distancia)
#print(matriz_distancias)


print('--------------Adjacencia-------------------')
k = 2
matriz_adjacencias = gerar_matriz_adjacencias(dados, matriz_distancias, k, 'mutKNN')
#print(matriz_adjacencias)

print('------------------Pesos----------------------------------')
sigma = 0.2
matriz_pesos = gerar_matriz_pesos(dados, matriz_adjacencias, matriz_distancias, sigma, k, "RBF")
#print(matriz_pesos)

checar_matrix_adjacencias(matriz_pesos)

posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos_semissupervisionado)

# Extracao das submatrizes da matriz laplaciana
LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada = divisao_L(matriz_pesos, posicoes_rotulos, ordemObjetos)

matriz_rotulos = one_hot(rotulos_semissupervisionado)
yl = matriz_rotulos[posicoes_rotulos,:]

print('----------Propagação------------------------------')
omega =  np.random.rand(len(classes),1)

parametro_regularizacao = 0.99

rotulos_propagados = propagar(posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada, yl, rotulos_semissupervisionado, matriz_rotulos, omega, parametro_regularizacao, "LGC")
print('Rotulos originais:')
print(rotulos)
print('Rotulos faltando:')
print(rotulos_semissupervisionado)
print('Rotulos Propagados:')
print(rotulos_propagados)

print('------------acuracia--------------')
print(acuracia(rotulos, rotulos_propagados, rotulos_semissupervisionado))


plt.scatter(dados[:, 0], dados[:, 1], marker="o", c=rotulos, s=25)
plt.title('rotulos originais')
plt.colorbar(label='cor por classe')
plt.show()

plt.scatter(dados[:, 0], dados[:, 1], marker="o", c=rotulos_semissupervisionado, s=25)
plt.title('rotulos semissupervisionados')
plt.colorbar(label='cor por classe')
plt.show()

plt.scatter(dados[:, 0], dados[:, 1], marker="o", c=rotulos_propagados, s=25)
plt.title('rotulos propagados')
plt.colorbar(label='cor por classe')
plt.show()
