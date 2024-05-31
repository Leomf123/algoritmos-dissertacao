from scipy.spatial.distance import cdist

# KNN para calcular a matriz de distancias
# entrada: dados, medida de similaridade
# saída: matriz de distancias
def gerar_matriz_distancias(X, Y, medida_distancia = 'euclidean'):

  matriz = cdist(X, Y, medida_distancia )

  return matriz
