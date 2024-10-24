from scipy.optimize import minimize, nnls
import numpy as np

from LAE import LAE

# RBF Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e sigma
# saída: matriz de pesos
def RBF(matriz_distancias, sigma):

  #print("inicializando RBF", end="... ")
  n = matriz_distancias.shape[0]

  matriz_kernel = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      matriz_kernel[i][j] = np.exp(-1*(matriz_distancias[i][j]**2)/(2*(sigma**2)))
  
  #print("feito")

  return matriz_kernel

# HM Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e k
# saída: matriz de pesos
def HM(matriz_distancias, k):

  #print("Inicializando HM", end="... ")
  n = matriz_distancias.shape[0]

  matriz_kernel = np.zeros((n, n))
  for i in range(n):
    psi_i = np.sort(matriz_distancias[i])[:k+1]
    for j in range(n):
      psi_j = np.sort(matriz_distancias[j])[:k+1]
      matriz_kernel[i][j] = np.exp(-1*(matriz_distancias[i][j]**2)/(max(psi_i[-1],psi_j[-1])**2))
  
  #print("feito")

  return matriz_kernel

def LLE(dados, matriz_adjacencia):

    matriz_pesos = LAE(dados, dados, matriz_adjacencia)
    
    symFKNN = np.any(matriz_adjacencia == 2)
    if symFKNN:
        matriz_pesos = matriz_pesos * matriz_adjacencia

    matriz_pesos = 1/2*(matriz_pesos + matriz_pesos.T)

    return matriz_pesos

def gerar_matriz_pesos(dados, matriz_adjacencias, matriz_distancias, sigma = 0.2, k = 2, algoritmo = "RBF"):
  
  if algoritmo == "RBF":
    return matriz_adjacencias * RBF(matriz_distancias, sigma)
  
  elif algoritmo == "HM":
    return matriz_adjacencias * HM(matriz_distancias, k)
  
  elif algoritmo == "LLE":
    return LLE(dados, matriz_adjacencias)
