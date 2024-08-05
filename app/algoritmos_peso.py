from scipy.optimize import minimize, nnls
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
    psi_i = np.sort(matriz_distancias[i])[:k+1]
    for j in range(n):
      psi_j = np.sort(matriz_distancias[j])[:k+1]
      matriz_kernel[i][j] = np.exp(-1*np.power(2,matriz_distancias[i][j])/np.power(2,max(psi_i[-1],psi_j[-1])))
  
  #print("feito")

  return matriz_kernel

def LLE(dados, matriz_adjacencia):

    #print("Inicializando LLE", end="... " )

    matriz_pesos = np.zeros((matriz_adjacencia.shape[0],matriz_adjacencia.shape[1]))

      
    for i in range(dados.shape[0]):
    
        # Criar matriz Z com os vizinhos de Xi
        posicoes = np.where(matriz_adjacencia[i,:] != 0)[0]
        Z = dados[posicoes] - dados[i]
    
        # Variancia Local
        C = np.dot(Z, Z.T).astype(float)

        C += np.eye(len(posicoes)) * 0.001

        try:

            # Resolve o sistema linear C * w = 1
            ones = np.ones(len(posicoes))
            w, _ = nnls(C, ones)
            # Normaliza os pesos
            w /= np.sum(w)
        
        except Exception as e:
            
            # Função objetivo para a otimização
            def obj(w):
                return w @ C @ w
        
            # Restrições para o problema
            # Soma dos pesos deve ser 1
            cons = [
                {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
            ]
        
            # Limites para os pesos (não negativos)
            bounds = [(0, None)] * len(posicoes)
        
            # Resolver o problema de otimização
            x0 = np.zeros(len(posicoes))
            w = minimize(obj, x0, bounds=bounds, constraints=cons).x

        matriz_pesos[i, posicoes] = w
    
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
