import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.sparse import csr_matrix

# KNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k e tipo
# saida: matriz de adjacencia
def knn(dados, matriz_distancia, k, tipo):

  print("inicializando KNN... ")

  matriz_adjacencia =  kneighbors_graph(dados, k, mode='connectivity').toarray()

  matriz_adjacencia_transposta = matriz_adjacencia.T

  if tipo == 'mutKNN':
     
    matriz_adjacencia = np.minimum(matriz_adjacencia, matriz_adjacencia_transposta)

    for i in range(matriz_adjacencia.shape[0]):
      # checar se ta isolado
      if np.sum(matriz_adjacencia[i]) == 0:
        # k_indices = np.argsort(matriz_distancia[i])[:2]
        k_indices = np.argpartition(matriz_distancia[i], 2)[:2]
        for k in k_indices:
          if i != k:
            matriz_adjacencia[i][k] = 1
            matriz_adjacencia[k][i] = 1

  elif tipo == 'symKNN':
    matriz_adjacencia = np.maximum(matriz_adjacencia, matriz_adjacencia_transposta)

  elif tipo == 'symFKNN':
    matriz_adjacencia = matriz_adjacencia + matriz_adjacencia_transposta

  return matriz_adjacencia


def gerar_matriz_adjacencias(dados, matriz_distancias, k = 4, algoritmo = 'multKNN'):
  
  if algoritmo in ['mutKNN', 'symKNN', 'symFKNN']:

    return knn(dados, matriz_distancias, k, algoritmo)