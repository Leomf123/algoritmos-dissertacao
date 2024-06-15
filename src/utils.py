from scipy.spatial.distance import cdist
import numpy as np
import heapq
import pandas as pd


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
    
    return simetrica, conectado

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

    return L, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada



def gravar_resultados(test_ID, nome_dataset, k, adjacencia, simetrica, conectado, ponderacao, r, e, propagacao, seed, tempo, nRotulos, acuracia, f_measure):
    
    if test_ID == 0: 

        # Criando um DataFrame vazio
        df = pd.DataFrame(columns=['test_ID', 'Dataset', 'Adjacencia', 'k', 'Ponderacao', 'Propagacao', 'PorcRot', 'NumExp', 'SeedExp', 'TempExp (min)', 'NumNRot', 'Acuracia', 'F_measure' ])
        # Adicionando dados
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Simetrica': simetrica, 'Conectado': conectado, 'Ponderacao': ponderacao, 'Propagacao': propagacao, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'TempExp': tempo, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        dados = pd.DataFrame(dados)
        df = pd.concat([df, dados], ignore_index=True)
        # salvo arquivo csv
        df.to_csv('Resultados.csv', index=False)

    else:
        
        # leio arquivo csv existente e salvo df
        df = pd.read_csv('Resultados.csv')
  
        # Adicionando dados
        dados = [{'test_ID': test_ID, 'Dataset': nome_dataset, 'Adjacencia': adjacencia, 'k': k, 'Ponderacao': ponderacao, 'Propagacao': propagacao, 'PorcRot': r, 'NumExp': e, 'SeedExp': seed, 'TempExp': tempo, 'NumNRot': nRotulos, 'Acuracia': acuracia, 'F_measure': f_measure}]

        dados = pd.DataFrame(dados)
        df = pd.concat([df, dados], ignore_index=True)

        # salvo arquivo csv mesmo lugar do outro
        df.to_csv('Resultados.csv', index=False)

    #print("Gravação concluída")

def definir_medida_distancia(nome_dado):

    datasets_euclidean = [
        "autoPrice.data",
        "banknote-authentication.data",
        "cardiotocography.data",
        "chscase_geyser1.data",
        "diggle_table.data",
        "iris.data",
        "seeds.data",
        "segmentation-normcols.data",
        "stock.data",
        "transplant.data",
        "wdbc.data",
        "wine-187.data",
        "yeast_Galactose.data"
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
        "semeion.data",
        "mfeat-factors.data",
        "mfeat-karhunen.data",
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
        "transplant.data",
        "diggle_table.data",
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
        ik = np.partition(matriz_distancias[i], k)[:k]
        sigma += ik[-1]/(3*n)
    
    return sigma

def retornar_omega(classes):

    omega = np.ones((len(classes), 1))/len(classes)

    return omega