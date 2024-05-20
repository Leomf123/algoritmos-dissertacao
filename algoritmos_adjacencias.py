import numpy as np

# KNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k e tipo
# saida: matriz de adjacencia
def knn(matriz_distancia,k,tipo):

  print("inicializando KNN... ")

  matriz_adjacencia = np.zeros((matriz_distancia.shape[0],matriz_distancia.shape[1]))

  for i in range(matriz_distancia.shape[0]):

    # Descobre os k indices os quais i vai ter aresta pra eles
    k_indices = np.argsort(matriz_distancia[i])[:k+1]

    # Monta a matriz de adjacencia: construo a linha de i
    for j in range(matriz_adjacencia.shape[1]):
      if i !=j and j in k_indices:
        matriz_adjacencia[i][j]=1

  matriz_adjacencia_transposta = matriz_adjacencia.T

  if tipo == 'mutKNN':

     for i in range(matriz_adjacencia.shape[0]):
        for j in range(matriz_adjacencia.shape[1]):
          matriz_adjacencia[i][j] = min(matriz_adjacencia[i][j],matriz_adjacencia_transposta[i][j])

  elif tipo == 'symKNN':

     for i in range(matriz_adjacencia.shape[0]):
        for j in range(matriz_adjacencia.shape[1]):
          matriz_adjacencia[i][j] = max(matriz_adjacencia[i][j],matriz_adjacencia_transposta[i][j])

  elif tipo == 'symFKNN':

     for i in range(matriz_adjacencia.shape[0]):
        for j in range(matriz_adjacencia.shape[1]):
          matriz_adjacencia[i][j] = matriz_adjacencia[i][j] + matriz_adjacencia_transposta[i][j]

  return matriz_adjacencia


def gerar_matriz_adjacencias(matriz_distancia,k = 4,tipo = 'mutKNN',algoritmo = 'KNN'):
  
  if algoritmo == 'KNN':

    return knn(matriz_distancia,k,tipo)