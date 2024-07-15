import sys
import os

# Caminho absoluto para o diretório que contém o módulo
caminho_diretorio = os.path.join(os.path.dirname(__file__), 'algoritmos-dissertacao', 'app')
# Adiciona o diretório ao sys.path
sys.path.append(caminho_diretorio)

# Agora podemos importar o módulo diretamente
from teste import teste


def main():
    
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

    Propagacao = ["GRF", "RMGT", "LGC", "LapRLS", "LapSVM"]
    
    teste(datasets, K, Adjacencia, Ponderacao, Quantidade_rotulos, Quantidade_experimentos, Propagacao)


if __name__ == "__main__":
    main()
