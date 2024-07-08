import numpy as np
from sklearn.neighbors import kneighbors_graph
from utils import primMST

# KNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k e tipo
# saida: matriz de adjacencia
def knn(dados, matriz_distancia, k, tipo):

  #print("inicializando " + tipo, end="... ")

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
  
  #print("feito")

  return matriz_adjacencia

def MST(matriz_distancias, mpts):
    
    #print("inicializando MST", end="... ")

    # 1- Calcular a core distance
    core_distance = np.zeros(matriz_distancias.shape[0])
    for i in range(matriz_distancias.shape[0]):
        # Descobre os k indices os quais i vai ter aresta pra eles - incluo ele mesmo
        # vizinhos = np.sort(matriz_distancias[i])[:mpts]
        vizinhos = np.partition(matriz_distancias[i], mpts)[:mpts]
        core_distance[i] = np.max(vizinhos)

    # 2- Criar grafo de Mutual Reachability Distance
    grafoMRD = np.zeros((matriz_distancias.shape[0],matriz_distancias.shape[1]))
    for i in range(matriz_distancias.shape[0]):
        for j in range(matriz_distancias.shape[1]):
            grafoMRD[i][j] = max(core_distance[i], core_distance[j], matriz_distancias[i][j])

    # 3- Gerar MST: Aplicar Prim
    MST = np.array(primMST(grafoMRD))

    MST[MST != 0] = 1

    #print("feito")

    return MST

def gerar_matriz_adjacencias(dados, matriz_distancias, k = 4, algoritmo = 'mutKNN'):
  
  if algoritmo in ['mutKNN', 'symKNN', 'symFKNN']:
    return knn(dados, matriz_distancias, k, algoritmo)
  
  elif algoritmo == "MST":
    return MST(matriz_distancias, k)
