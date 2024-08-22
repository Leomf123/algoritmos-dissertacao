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
        "armstrong2002v1.data",
        "chowdary2006.data",
        "ace_ECFP_4.data",
        "ace_ECFP_6.data",
        "cox2_ECFP_6.data",
        "dhfr_ECFP_4.data",
        "dhfr_ECFP_6.data",
        "fontaine_ECFP_4.data",
        "fontaine_ECFP_6.data",
        "m1_ECFP_4.data",
        "m1_ECFP_6.data",
        "transplant.data",
        "autoPrice.data",
        "seeds.data",
        "chscase_geyser1.data",
        "diggle_table.data",
        "gordon2002.data",
        "articles_1442_5.data",
        "articles_1442_80.data",
        "iris.data",
        "analcatdata_authorship-458.data",
        "wine-187.data",
        "banknote-authentication.data",
        "yeast_Galactose.data",
        "semeion.data",
        "wdbc.data",
        "mfeat-karhunen.data",
        "mfeat-factors.data",
        "stock.data",
        "segmentation-normcols.data",
        "cardiotocography.data",
    ]

    K = [2, 4, 6, 8, 10, 12, 14, 16]

    Adjacencia = ["mutKNN", "symKNN", "symFKNN", "MST"]

    Ponderacao = ["LLE"]

    Quantidade_rotulos = [0.02, 0.05, 0.08, 0.1]

    Quantidade_experimentos = 1

    Propagacao = ["GRF", "LGC"]
    
    teste(datasets, K, Adjacencia, Ponderacao, Quantidade_rotulos, Quantidade_experimentos, Propagacao)


if __name__ == "__main__":
    main()
