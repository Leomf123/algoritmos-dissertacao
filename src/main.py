import numpy as np
import pandas as pd
import random
import time

from utils import gerar_matriz_distancias
from algoritmos_adjacencias import gerar_matriz_adjacencias
from algoritmos_peso import gerar_matriz_pesos
from processar_rotulos import retirar_rotulos, medidas_qualidade
from algoritmos_classificar import propagar
from utils import ordem_rotulos_primeiro, divisao_L, gravar_resultados, definir_medida_distancia
from utils import normalizar_dados, retornar_sigma, retornar_omega
from processar_rotulos import one_hot

datasets = [
    "ace_ECFP_4.data",
    "ace_ECFP_6.data",
    "analcatdata_authorship-458.data",
    "armstrong2002v1.data",
    "articles_1442_5.data",
    "articles_1442_80.data",
    "autoPrice.data",
    "banknote-authentication.data",
    "cardiotocography.data",
    "chowdary2006.data",
    "chscase_geyser1.data",
    "cox2_ECFP_6.data",
    "dhfr_ECFP_4.data",
    "dhfr_ECFP_6.data",
    "diggle_table.data",
    "fontaine_ECFP_4.data",
    "fontaine_ECFP_6.data",
    "gordon2002.data",
    "iris.data",
    "m1_ECFP_4.data",
    "m1_ECFP_6.data",
    "mfeat-factors.data",
    "mfeat-karhunen.data",
    "seeds.data",
    "segmentation-normcols.data",
    "semeion.data",
    "stock.data",
    "transplant.data",
    "wdbc.data",
    "wine-187.data",
    "yeast_Galactose.data"
]

K = [2, 4, 6, 8, 10, 12, 14, 16]

Adjacencia = ["mutKNN", "symKNN", "symFKNN", "MST"]

Ponderacao = ["RBF", "HM", "LLE"]

Quantidade_rotulos = [0.02, 0.05, 0.08, 0.1]

Quantidade_experimentos = 30

Propagacao = ["GRF", "RMGT", "LGC"]

test_ID = 0

inicio_geral = time.time()
# 1 - Para cada dataset
for nome_dataset in datasets:
    inicio = time.time()
    tempo = 0
    print("Datset: ", nome_dataset)
    # Lendo dados
    df = pd.read_csv('data/' + nome_dataset, header=None)

    # Conversão para numpy
    dados = df.to_numpy()
    # Separando rótulos dos dados
    ultima_coluna = dados.shape[1] - 1
    rotulos = np.array(dados[:,ultima_coluna], dtype='int64')
    dados = np.array(dados[:,:ultima_coluna])
    # Pegar classes
    classes = np.unique(rotulos)

    # Normalizar dados
    dados = normalizar_dados(nome_dataset, dados)

    # medida_distancia = 'euclidean'
    medida_distancia = definir_medida_distancia(nome_dataset)
    matriz_distancias = gerar_matriz_distancias(dados, dados, medida_distancia)
   
    # Usado no RMGT
    omega = retornar_omega(classes)

    # 2 - Para cada valor de K
    for k in K:
        # Usado no RBF
        sigma = retornar_sigma(matriz_distancias, k)

        # 3 - Para cada algoritmo de adjacencia
        for adjacencia in Adjacencia:
            print(adjacencia)
            # Gerar matriz de adjacencia
            matriz_adjacencias = gerar_matriz_adjacencias(dados, matriz_distancias, k, adjacencia)
            
            # 4 - Para cada ponderação
            for ponderacao in Ponderacao:
                print(ponderacao)
                # Gerar matriz pesos
                matriz_pesos = gerar_matriz_pesos(dados, matriz_adjacencias , matriz_distancias, sigma, k, ponderacao)
                
                # 5 - Para cada quantidade de rotulos
                for r in Quantidade_rotulos:

                    #Gerar os seeds
                    seeds = random.sample(range(1, 200), Quantidade_experimentos )

                    # 6 - Quantidade de experimentos
                    for e in range(Quantidade_experimentos):

                        # Retirar quantidade de rotulos
                        rotulos_semissupervisionado = retirar_rotulos(rotulos, r, classes, seeds[e])

                        posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos_semissupervisionado)

                        # Extracao das submatrizes da matriz laplaciana
                        L, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada = divisao_L(matriz_pesos, posicoes_rotulos, ordemObjetos)
                        
                        matriz_rotulos = one_hot(rotulos_semissupervisionado)
                        yl = matriz_rotulos[posicoes_rotulos,:]

                        #7 - Para cada algoritmo de classificação semi
                        for propagacao in Propagacao:
                            print(propagacao)
                            # Propagar rotulos
                            lambda_k = 0.1
                            lambda_u = 0.1
                            # Usado no LGC
                            parametro_regularizacao = 0.99
                            rotulos_propagados = propagar(dados, L, posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada, yl, rotulos_semissupervisionado, matriz_rotulos, classes, medida_distancia, k, lambda_k, lambda_u, omega, parametro_regularizacao, propagacao)

                            # Usar medidas de qualidade
                            acuracia, f_measure, nRotulos = medidas_qualidade(posicoes_rotulos, ordemObjetos, rotulos, rotulos_propagados)

                            # Gravar tempo que levou
                            fim = time.time()
                            tempo = fim - ( inicio + tempo )

                            # gravar resultado em uma linha usando pandas
                            gravar_resultados(test_ID, nome_dataset, k, adjacencia, ponderacao, r, e, propagacao, seeds[e], tempo, nRotulos, acuracia, f_measure)

                            print("test_ID: ", test_ID)

                            test_ID += 1
fim_geral = time.time()
tempo_geral = fim_geral - inicio_geral
print("Tempo geral execução (min): ", tempo_geral/60)
