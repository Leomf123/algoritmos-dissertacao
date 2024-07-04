from numpy.linalg import solve
import numpy as np


# RBF Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e sigma
# saída: matriz de pesos
def RBF(matriz_distancias, sigma):

  #print("inicializando RBF", end="... ")
  n = matriz_distancias.shape[0]

  matriz_kernel = np.zeros((n, n))
  for i in range(n):
    for j in range(n):
      matriz_kernel[i][j] = np.exp(-1*np.power(2,matriz_distancias[i][j])/2*np.power(2,sigma))
  
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
    psi_i = np.partition(matriz_distancias[i], k)[:k]
    for j in range(n):
      psi_j = np.partition(matriz_distancias[j], k)[:k]
      matriz_kernel[i][j] = np.exp(-1*np.power(2,matriz_distancias[i][j])/np.power(2,max(psi_i[-1],psi_j[-1])))
  
  #print("feito")

  return matriz_kernel

def LLE(dados, matriz_adjacencia):

  #print("Inicializando LLE", end="... " )

  matriz_pesos = np.zeros((matriz_adjacencia.shape[0],matriz_adjacencia.shape[1]))

  X = dados.T

  for i in range(X.shape[1]):
    
    # Criar matriz Z com os vizinhos de Xi
    posicoes = np.where(matriz_adjacencia[i,:] != 0)[0]
    Z = np.array(X[:,posicoes])
    # Subtrair Xi de Z
    for j in range(Z.shape[1]):
      Z[:,j] = Z[:,j]-X[:,i]
    
    # Variancia Local
    C = Z.T @ Z
    
    if C.dtype == 'int' or C.dtype == 'int64':
      C = C.astype(np.float64)

    C += np.eye(C.shape[0]) + 0.00001
    
    # Resolve o sistema linear C * w = 1
    ones = np.ones(len(posicoes))
    w = solve(C, ones)
   
    for j in range(len(w)):
      matriz_pesos[i][posicoes[j]] = w[j]/ np.sum(w)
    
  symFKNN = np.any(matriz_adjacencia == 2)
  if symFKNN:
    matriz_pesos = matriz_pesos * matriz_adjacencia

  matriz_pesos = 1/2*(matriz_pesos + matriz_pesos.T)

  #print("feito")

  return matriz_pesos


def gerar_matriz_pesos(dados, matriz_adjacencias, matriz_distancias, sigma = 0.2, k = 2, algoritmo = "RBF"):
  
  if algoritmo == "RBF":
    return matriz_adjacencias * RBF(matriz_distancias, sigma)
  
  elif algoritmo == "HM":
    return matriz_adjacencias * HM(matriz_distancias, k)
  
  elif algoritmo == "LLE":
    return LLE(dados, matriz_adjacencias)
