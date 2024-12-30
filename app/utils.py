from scipy.spatial.distance import cdist
import numpy as np
import heapq
import pandas as pd


def gerar_matriz_distancias(X, Y, medida_distancia ):

  matriz = cdist(X, Y, medida_distancia )

  return matriz


def checar_matrix_adjacencias(matriz_adjacencias):

    simetrica = True
    conectado = True
    positivo = True
    for i in range(matriz_adjacencias.shape[0]):
        if np.sum(matriz_adjacencias[i]) == 0:
            conectado = False
        for j in range(matriz_adjacencias.shape[1]):
            if matriz_adjacencias[i][j] != matriz_adjacencias[j][i]:
                simetrica = False
    
    if (np.any(matriz_adjacencias < 0)):
        positivo = False

    return simetrica, conectado, positivo

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

    # Máscara para identificar elementos rotulados
    mascara_rotulos = rotulos != 0

    # Índices de rótulos e não rótulos
    posicoes_rotulos = np.flatnonzero(mascara_rotulos)
    nao_rotulados = np.flatnonzero(~mascara_rotulos)

    # Combina diretamente os índices
    ordemObjetos = np.hstack((posicoes_rotulos, nao_rotulados))

    return posicoes_rotulos, ordemObjetos

def laplacianas(matriz_pesos):

    n = matriz_pesos.shape[0]
    
    # Matriz diagonal (grau dos vértices)
    D = np.diag(np.sum(matriz_pesos, axis=1))
    
    # Matriz laplaciana
    L = 1.01 * D - matriz_pesos
    
    # Laplaciana normalizada
    D_inv_raiz = np.diag(1 / np.sqrt(np.diag(D)))
    L_normalizada = 1.01 * np.eye(n) - D_inv_raiz @ matriz_pesos @ D_inv_raiz

    return L, L_normalizada
    

def processar_laplacianas(L, posicoes_rotulos, ordemObjetos, yl):

    # Reordenar matriz laplaciana
    L = L[ordemObjetos, :]
    L = L[:, ordemObjetos]
    
    # Divisão em submatrizes
    nRotulado = len(posicoes_rotulos)
    LRotulado = L[:nRotulado, :nRotulado]
    LNaoRotuladoRotulado = L[nRotulado:, :nRotulado]
    LNaoRotulado = L[nRotulado:, nRotulado:]

    LNaoRotulado_inv = np.linalg.inv(LNaoRotulado)
    formula_comum_grf_rmgt = LNaoRotulado_inv.dot(LNaoRotuladoRotulado).dot(yl)

    return LRotulado, LNaoRotulado_inv, formula_comum_grf_rmgt



def gravar_resultados(test_ID, nome_dataset, k, adjacencia, simetrica, conectado, positivo, ponderacao, r, e, propagacao, seed, nRotulos, acuracia, f_measure):
    
    if test_ID == 0: 

        # Criando um dataframe
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Ponderacao': ponderacao, 'Simetrica': simetrica, 'Conectado': conectado, 'Positivo': positivo, 'Propagacao': propagacao, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        df = pd.DataFrame(dados)
        # salvo arquivo csv
        df.to_csv('Resultados.csv', index=False)

    else:
        
        # leio arquivo csv existente e salvo df
        df = pd.read_csv('Resultados.csv')
  
        # Adicionando dados
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Ponderacao': ponderacao, 'Simetrica': simetrica, 'Conectado': conectado, 'Positivo': positivo, 'Propagacao': propagacao, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        dados = pd.DataFrame(dados)
        df = pd.concat([df, dados], ignore_index=True)

        # salvo arquivo csv mesmo lugar do outro
        df.to_csv('Resultados.csv', index=False)


def definir_medida_distancia(nome_dado):

    datasets_euclidean = [
        "autoPrice.data",
        "banknote-authentication.data",
        "chscase_geyser1.data",
        "diggle_table.data",
        "iris.data",
        "seeds.data",
        "segmentation-normcols.data",
        "stock.data",
        "transplant.data",
        "wdbc.data",
        "wine-187.data",
        "yeast_Galactose.data",
        "mfeat-factors.data",
        "mfeat-karhunen.data",
        "cardiotocography.data"
    ]    
    datasets_tanimoto = [
        "ace_ECFP_4.data",
        "ace_ECFP_6.data",
        "cox2_ECFP_6.data",
        "dhfr_ECFP_4.data",
        "dhfr_ECFP_6.data",
        "fontaine_ECFP_4.data",
        "fontaine_ECFP_6.data",
        "m1_ECFP_4.data",
        "m1_ECFP_6.data",
    ]

    datasets_cosine = [
        "articles_1442_5.data",
        "articles_1442_80.data",
        "analcatdata_authorship-458.data",
        "armstrong2002v1.data",
        "chowdary2006.data",
        "gordon2002.data",
        "semeion.data"
    ]

    if nome_dado in datasets_tanimoto:
        return 'rogerstanimoto'
    elif nome_dado in datasets_cosine:
        return  'cosine'
    else:
        return 'euclidean'

def normalizar_dados(nome_dado, dados):

    datasets_Normalizar = [
        "autoPrice.data",
        "banknote-authentication.data",
        "stock.data",
        "transplant.data"
    ]    

    dados_normalizados = np.array(dados)

    if nome_dado in datasets_Normalizar:
        mean = np.mean(dados, axis=0)
        std = np.std(dados, axis=0)
        dados_normalizados = (dados - mean) / std
    
    return dados_normalizados

def retornar_sigma(matriz_distancias, k):
    
    n = matriz_distancias.shape[0]
    sigma = 0
    for i in range(n):
        ik = np.sort(matriz_distancias[i])[:k+1]
        sigma += ik[-1]

    return sigma / (3*n)

def retornar_omega(classes):

    omega = np.ones((len(classes), 1))/len(classes)

    return omega