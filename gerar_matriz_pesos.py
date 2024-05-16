import numpy as np


# RBF Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias, sigma
# saída: matriz de pesos
def RBF(matriz_adjacencia,matriz_distancia,sigma):

  print("Iniciando RBF...")

  matriz_pesos = np.zeros((matriz_adjacencia.shape[0],matriz_adjacencia.shape[1]))

  for i in range(matriz_adjacencia.shape[0]):
    for j in range(matriz_adjacencia.shape[1]):
      if matriz_adjacencia[i][j] >= 1:
        matriz_pesos[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/2*np.power(2,sigma))

  return matriz_pesos

# HM Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e k
# saída: matriz de pesos
def HM(matriz_adjacencia,matriz_distancia,k):

  print("Iniciando HM...")

  matriz_pesos = np.zeros((matriz_adjacencia.shape[0],matriz_adjacencia.shape[1]))

  for i in range(matriz_adjacencia.shape[0]):
    psi_i = matriz_distancia[i][k]
    for j in range(matriz_adjacencia.shape[1]):
      psi_j = matriz_distancia[j][k]
      if matriz_adjacencia[i][j] >= 1:
        matriz_pesos[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/np.power(2,max(psi_i,psi_j)))

  return matriz_pesos


def matriz_pesos(matriz_adjacencia,matriz_distancia,sigma = 0.2,k = 2, algoritmo = "RBF"):
  
  if algoritmo == "RBF":
    return RBF(matriz_adjacencia,matriz_distancia,sigma)
  
  elif algoritmo == "HM":
    return HM(matriz_adjacencia,matriz_distancia,k)