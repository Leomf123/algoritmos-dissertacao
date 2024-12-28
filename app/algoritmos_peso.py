from scipy.optimize import minimize, nnls
import numpy as np

from LAE import LAE

# RBF Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e sigma
# saída: matriz de pesos
def RBF(matriz_distancias, sigma):

  return np.exp(-0.5 * (matriz_distancias ** 2) / (sigma ** 2))

# HM Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e k
# saída: matriz de pesos
def HM(matriz_distancias, k):
    
  # k menores valores por linha, excluindo ele mesmo
  psi = np.sort(matriz_distancias, axis=1)[:, k]
  # Construir a matriz de máximos dos psi
  psi_max = np.maximum.outer(psi, psi)

  return np.exp(-1 * (matriz_distancias**2) / (psi_max**2))

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
