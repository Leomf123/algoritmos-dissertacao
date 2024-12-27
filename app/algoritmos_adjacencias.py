import numpy as np
from sklearn.neighbors import kneighbors_graph

from utils import primMST

def tratamento_no_isolado(matriz_adjacencia, matriz_distancia):

  # Tratamento de pontos isolados
  matriz = matriz_adjacencia.copy()
  pontos_isolados = np.where(matriz_adjacencia.sum(axis=1) == 0)[0]

  for i in pontos_isolados:
    k_indices = np.argsort(matriz_distancia[i])[:2]
    for k in k_indices:
      if i != k:
        matriz[i][k] = 1
        matriz[k][i] = 1

  return matriz

# KNN para calcular a matriz de adjacencias
# entrada: matriz de distancia, k e tipo
# saida: matriz de adjacencia
def knn(dados, matriz_distancia, medida_distancia, k, tipo):

  # Converte em bool para usar a rogerstanimoto sem da warning
  if medida_distancia == 'rogerstanimoto':
     dados = dados.astype(bool)

  matriz_adjacencia =  kneighbors_graph(dados, k, mode='connectivity',  metric = medida_distancia, include_self=False).toarray()

  matriz_adjacencia_transposta = matriz_adjacencia.T

  if tipo == 'mutKNN':
    matriz_adjacencia = np.minimum(matriz_adjacencia, matriz_adjacencia_transposta)

    matriz_adjacencia = tratamento_no_isolado(matriz_adjacencia, matriz_distancia)

  elif tipo == 'symKNN':
    matriz_adjacencia = np.maximum(matriz_adjacencia, matriz_adjacencia_transposta)

  elif tipo == 'symFKNN':
    matriz_adjacencia = matriz_adjacencia + matriz_adjacencia_transposta

  return matriz_adjacencia

def MST(matriz_distancias, mpts):
    
    #print("inicializando MST", end="... ")

    # 1- Calcular a core distance
    core_distance = np.zeros(matriz_distancias.shape[0])
    for i in range(matriz_distancias.shape[0]):
        # Descobre os k indices os quais i vai ter aresta pra eles - incluo ele
        distancias_vizinhos = np.sort(matriz_distancias[i])[:mpts]
        #vizinhos = np.partition(matriz_distancias[i], mpts)[:mpts]
        core_distance[i] = distancias_vizinhos[-1]

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

def gerar_matriz_adjacencias(dados, matriz_distancias, medida_distancia, k = 4, algoritmo = 'mutKNN'):
  
  if algoritmo in ['mutKNN', 'symKNN', 'symFKNN']:
    return knn(dados, matriz_distancias, medida_distancia, k, algoritmo)
  
  elif algoritmo == "MST":
    return MST(matriz_distancias, k)
