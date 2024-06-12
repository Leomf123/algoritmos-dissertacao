import numpy as np
import pandas as pd
import random
import time

from utils import gerar_matriz_distancias
from algoritmos_adjacencias import gerar_matriz_adjacencias
from algoritmos_peso import gerar_matriz_pesos
from processar_rotulos import retirar_rotulos, medidas_qualidade
from algoritmos_classificar import propagar
from utils import ordem_rotulos_primeiro, divisao_L, gravar_resultados
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

K = [2, 4, 6, 8, 10]

Adjacencia = ["multKNN", "symKNN", "symFKNN", "MST"]

Ponderacao = ["RBF", "HM", "LLE"]

Quantidade_rotulos = [0.02, 0.05, 0.08, 0.1]

Quantidade_experimentos = 30

Propagacao = ["GRF", "RMGT", "LGC", "LapRLS", "LapSVM"]

test_ID = 0

# 1 - Para cada dataset
for nome_dataset in datasets:
    inicio = time.time()

    # Lendo dados
    df = pd.read_csv('data/' + nome_dataset, header=None)
    # Conversão para numpy
    dados = df.to_numpy()
    # Separando rótulos dos dados
    ultima_coluna = dados.shape[1] - 1
    classes = dados[:,dados.shape[1] - 1].unique()
    rotulos = np.array(dados[:,dados.shape[1] - 1], dtype='int64')
    dados = np.array(dados[:,:dados.shape[1] - 1])

    # Pegar classes

    # Precisa normalizar os dados?

    medida_distancia = 'euclidean'
    matriz_distancias = gerar_matriz_distancias(dados, dados, medida_distancia)

    # 2 - Para cada valor de K
    for k in K:

        # 3 - Para cada algoritmo de adjacencia
        for adjacencia in Adjacencia:

            # Gerar matriz de adjacencia
            matriz_adjacencias = gerar_matriz_adjacencias(dados, matriz_distancias, k, adjacencia)

            # 4 - Para cada ponderação
            for ponderacao in Ponderacao:

                # Gerar matriz pesos
                # to do: otimizar algoritimos de peso
                sigma = 0.2
                matriz_pesos = gerar_matriz_pesos(dados, matriz_adjacencias , matriz_distancias, sigma, k, ponderacao)
                
                # 5 - Para cada quantidade de rotulos
                for r in Quantidade_rotulos:

                    #Gerar os seeds
                    seeds = random.sample(range(1, 200), Quantidade_experimentos )

                    # 6 - Quantidade de experimentos
                    for e in range(Quantidade_experimentos):

                        # Retirar quantidade de rotulos
                        # to do: otimizar retirar_rotulos
                        rotulos_semissupervisionado = retirar_rotulos(rotulos, r, classes, seeds[e])

                        posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos_semissupervisionado)

                        # Extracao das submatrizes da matriz laplaciana
                        L, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada = divisao_L(matriz_pesos, posicoes_rotulos, ordemObjetos)
                        
                        matriz_rotulos = one_hot(rotulos_semissupervisionado)
                        yl = matriz_rotulos[posicoes_rotulos,:]

                        #7 - Para cada algoritmo de classificação semi
                        for propagacao in Propagacao:

                            # Propagar rotulos
                            lambda_k = 0.1
                            lambda_u = 0.1
                            omega = 0
                            parametro_regularizacao = 0.1
                            rotulos_propagados = propagar(dados, L, posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada, yl, rotulos_semissupervisionado, matriz_rotulos, classes, medida_distancia, k, lambda_k, lambda_u, omega, parametro_regularizacao, propagacao)

                            # Usar medidas de qualidade
                            acuracia, f_measure, nRotulos = medidas_qualidade(posicoes_rotulos, ordemObjetos, rotulos, rotulos_propagados)

                            test_ID += 1

                            # Gravar tempo que levou
                            fim = time.time()
                            tempo = fim - inicio

                            # gravar resultado em uma linha usando pandas
                            gravar_resultados(test_ID, nome_dataset, k, adjacencia, ponderacao, r, e, propagacao, seeds[e], tempo, nRotulos, acuracia, f_measure)

                            print(test_ID)









    
