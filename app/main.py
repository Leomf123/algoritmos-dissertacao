from teste import teste


def main():
    
    datasets = [
        "iris.data",
    ]

    K = [2, 4, 6, 8, 10, 12, 14, 16]

    Adjacencia = ["mutKNN", "symKNN", "symFKNN", "MST"]

    Ponderacao = ["RBF", "HM", "LLE"]

    Quantidade_rotulos = [0.02, 0.05, 0.08, 0.1]

    Quantidade_experimentos = 20

    Propagacao = ["GRF", "RMGT", "LGC", "LapRLS", "LapSVM"]
    
    teste(datasets, K, Adjacencia, Ponderacao, Quantidade_rotulos, Quantidade_experimentos, Propagacao)


if __name__ == "__main__":
    main()
