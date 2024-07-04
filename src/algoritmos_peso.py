from numpy.linalg import solve
import numpy as np


# RBF Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e sigma
# saída: matriz de pesos
def RBF(matriz_adjacencia, matriz_distancia, sigma):

  #print("inicializando RBF", end="... ")

  matriz_pesos = np.zeros((matriz_adjacencia.shape[0],matriz_adjacencia.shape[1]))

  for i in range(matriz_adjacencia.shape[0]):
    for j in range(matriz_adjacencia.shape[1]):
      matriz_pesos[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/2*np.power(2,sigma))
  
  #print("feito")

  return matriz_pesos * matriz_adjacencia

# HM Kernel para calcular a matriz de pesos
# entrada: matriz de adjacencias, matriz de distancias e k
# saída: matriz de pesos
def HM(matriz_adjacencia,matriz_distancia,k):

  #print("Inicializando HM", end="... ")

  matriz_pesos = np.zeros((matriz_adjacencia.shape[0],matriz_adjacencia.shape[1]))

  for i in range(matriz_adjacencia.shape[0]):
    psi_i = np.partition(matriz_distancia[i], k)[:k]
    for j in range(matriz_adjacencia.shape[1]):
      psi_j = np.partition(matriz_distancia[j], k)[:k]
      matriz_pesos[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/np.power(2,max(psi_i[-1],psi_j[-1])))
  
  #print("feito")

  return matriz_pesos * matriz_adjacencia

def LLE(dados,matriz_adjacencia):

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


def gerar_matriz_pesos(dados,matriz_adjacencia,matriz_distancia,sigma = 0.2,k = 2, algoritmo = "RBF"):
  
  if algoritmo == "RBF":
    return RBF(matriz_adjacencia, matriz_distancia, sigma)
  
  elif algoritmo == "HM":
    return HM(matriz_adjacencia ,matriz_distancia, k)
  
  elif algoritmo == "LLE":
    return LLE(dados, matriz_adjacencia)