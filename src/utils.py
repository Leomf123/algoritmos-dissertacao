from scipy.spatial.distance import cdist
import numpy as np
import heapq


def gerar_matriz_distancias(X, Y, medida_distancia = 'euclidean'):

  matriz = cdist(X, Y, medida_distancia )

  return matriz


def checar_matrix_adjacencias(matriz_adjacencias):

    simetrica = True
    conectado = True
    for i in range(matriz_adjacencias.shape[0]):
        if np.sum(matriz_adjacencias[i]) == 0:
            conectado = False
        for j in range(matriz_adjacencias.shape[1]):
            if matriz_adjacencias[i][j] != matriz_adjacencias[j][i]:
                simetrica = False
                print(matriz_adjacencias[i][j])
                print(matriz_adjacencias[j][i])

    if simetrica:
        print('simetrica')
    else:
        print('não simetrica')
    
    if conectado:
        print('conectado')
    else:
        print('isolado')


def primMST(grafo):
    V = len(grafo)  
    pai = [-1] * V  
    chave = [float('inf')] * V  
    V_bool = [False] * V 

    chave[0] = 0
    min_heap = [(0, 0)] 

    while min_heap:
        
        _, u = heapq.heappop(min_heap)
        V_bool[u] = True

        
        for v in range(V):
            
            if grafo[u][v] > 0 and not V_bool[v] and chave[v] > grafo[u][v]:
                chave[v] = grafo[u][v]
                pai[v] = u
                heapq.heappush(min_heap, (chave[v], v))

    
    MST = [[0] * V for _ in range(V)]
    for v in range(1, V):
        u = pai[v]
        MST[u][v] = grafo[u][v]
        MST[v][u] = grafo[u][v]

    return MST
