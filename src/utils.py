from scipy.spatial.distance import cdist
import sys
import numpy as np

# KNN para calcular a matriz de distancias
# entrada: dados, medida de similaridade
# saída: matriz de distancias
def gerar_matriz_distancias(X, Y, medida_distancia = 'euclidean'):

  matriz = cdist(X, Y, medida_distancia )

  return matriz



 
class Graph():
    def __init__(self, vertices):
        self.V = vertices
        self.graph = [[0 for column in range(vertices)]
                      for row in range(vertices)]
 
    # A utility function to print 
    # the constructed MST stored in parent[]
    def printMST(self, parent):
       # print("Edge \tWeight")
       #for i in range(1, self.V):
            #print(parent[i], "-", i, "\t", self.graph[i][parent[i]])
        
        self.MST = np.zeros((self.graph.shape[0],self.graph.shape[1]))
        for i in range(self.graph.shape[0]):
          if parent[i] != -1:
            self.MST[i][parent[i]] = self.graph[i][parent[i]]
            self.MST[parent[i]][i] = self.graph[parent[i]][i]
 
    # A utility function to find the vertex with
    # minimum distance value, from the set of vertices
    # not yet included in shortest path tree
    def minKey(self, key, mstSet):
 
        # Initialize min value
        min = sys.maxsize
 
        for v in range(self.V):
            if key[v] < min and mstSet[v] == False:
                min = key[v]
                min_index = v
 
        return min_index
 
    # Function to construct and print MST for a graph
    # represented using adjacency matrix representation
    def primMST(self):
 
        # Key values used to pick minimum weight edge in cut
        key = [sys.maxsize] * self.V
        parent = [None] * self.V  # Array to store constructed MST
        # Make key 0 so that this vertex is picked as first vertex
        key[0] = 0
        mstSet = [False] * self.V
 
        parent[0] = -1  # First node is always the root of
 
        for cout in range(self.V):
 
            # Pick the minimum distance vertex from
            # the set of vertices not yet processed.
            # u is always equal to src in first iteration
            u = self.minKey(key, mstSet)
 
            # Put the minimum distance vertex in
            # the shortest path tree
            mstSet[u] = True
 
            # Update dist value of the adjacent vertices
            # of the picked vertex only if the current
            # distance is greater than new distance and
            # the vertex in not in the shortest path tree
            for v in range(self.V):
 
                # graph[u][v] is non zero only for adjacent vertices of m
                # mstSet[v] is false for vertices not yet included in MST
                # Update the key only if graph[u][v] is smaller than key[v]
                if self.graph[u][v] > 0 and mstSet[v] == False \
                and key[v] > self.graph[u][v]:
                    key[v] = self.graph[u][v]
                    parent[v] = u
 
        self.printMST(parent)




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

    