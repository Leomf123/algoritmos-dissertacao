import numpy as np
from utils import Graph

def MST_ext(matriz_distancias, mpts):


    # 1- Calcular a core distance
    core_distance = np.zeros(matriz_distancias.shape[0])
    for i in range(matriz_distancias.shape[0]):
        # Descobre os k indices os quais i vai ter aresta pra eles - incluo ele mesmo
        vizinhos = np.sort(matriz_distancias[i])[:mpts]
        core_distance[i] = max(vizinhos)

    # 2- Criar grafo de Mutual Reachability Distance
    grafoMRD = np.zeros((matriz_distancias.shape[0],matriz_distancias.shape[1]))
    for i in range(matriz_distancias.shape[0]):
        for j in range(matriz_distancias.shape[1]):
            grafoMRD[i][j] = max(core_distance[i], core_distance[j], matriz_distancias[i][j])
    
    
    # 3- Gerar MST: Aplicar Prim
    g = Graph(grafoMRD.shape[0])
    g.graph = grafoMRD
    g.primMST()
    MST = g.MST

    # 4- Gerar MST-ext: Colocar auto-arestas ???



    return MST




        

    

