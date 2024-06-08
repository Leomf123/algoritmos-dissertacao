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


def ordem_rotulos_primeiro(rotulos):

  #pegar posições existe rotulo
  posicoes_rotulos =  np.where( rotulos != 0)[0]

  # Reordenar as posições para os rotulados vim primeiro
  ordemObjetos = np.arange(rotulos.shape[0])
  # Retira dos indices os que são rotulados
  ordemObjetos = np.setdiff1d(ordemObjetos, posicoes_rotulos)
  # ordemObjetos será uma lista onde os indices dos objetos rotulados
  # vem primeiro, depois o resto em ordem crescente de indice
  ordemObjetos = np.concatenate((posicoes_rotulos,ordemObjetos))

  return posicoes_rotulos, ordemObjetos

def divisao_L(matriz_pesos, posicoes_rotulos, ordemObjetos):

    # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
    D = np.zeros(matriz_pesos.shape)
    np.fill_diagonal(D, np.sum(matriz_pesos, axis=1))
    # Calculo da matriz laplaciana
    L= 1.01*D - matriz_pesos

    # Calculo da laplaciana normalizada
    matriz_identidade = np.eye(matriz_pesos.shape[0])
    D_inv_raiz = np.diag(1 / np.sqrt(np.diag(D)))
    L_normalizada = 1.01*matriz_identidade - D_inv_raiz.dot(matriz_pesos.dot(D_inv_raiz))

    ## Reordenacao da matriz Laplaciana, para manter os dados rotulados à frente
    L = L[ordemObjetos,:]
    L = L[:, ordemObjetos]

    # Armazenar numero de objetos rotulados, nao rotulados e total de objetos
    nRotulado = len(posicoes_rotulos)
    nNaoRotulado = L.shape[0]-nRotulado
    nObjetos = L.shape[0]

    # Extracao das submatrizes da matriz laplaciana
    LRotulado = L[0:nRotulado, 0:nRotulado]
    LNaoRotuladoRotulado = L[nRotulado:nObjetos, 0:nRotulado]
    LNaoRotulado = L[nRotulado:nObjetos, nRotulado:nObjetos]

    return LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada
