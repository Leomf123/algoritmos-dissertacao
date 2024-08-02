import numpy as np
from sklearn.neighbors import kneighbors_graph
from scipy.spatial.distance import cdist

from utils import retornar_sigma

# Classe LapRLS adaptada da classe LapRLS do repositório:
# https://github.com/HugoooPerrin/semi-supervised-learning


class LapRLS(object):

    def __init__(self, L, distancy, k, lambda_k, lambda_u):
        
        self.L = L
        self.distancy = distancy
        self.k = k
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        

    def labelDiffusion(self, X, X_no_label, Y):
       
        self.l = X.shape[0]
        u = X_no_label.shape[0]
        n = self.l + u
        c = Y.shape[1]

        self.X = np.concatenate([X, X_no_label], axis=0)
        self.Y = np.vstack([Y, np.zeros((u, c))])
        
        # Memory optimization
        del X_no_label

        self.matriz_kernel = self.kernel(self.X, self.X)
        
        J = np.diag(np.concatenate([np.ones(self.l), np.zeros(u)]))
        
        #final = (J.dot(self.matriz_kernel) + self.lambda_k * self.l * np.identity(self.l + u) + ((self.lambda_u * self.l) / (self.l + u) ** 2) * self.L.dot(self.matriz_kernel))
        final = (J.dot(self.matriz_kernel) + self.lambda_k * self.l * np.identity(self.l + u) + self.lambda_u * self.l * self.L.dot(self.matriz_kernel))
        
        self.alpha = np.linalg.inv(final).dot(self.Y)
        
        del self.Y, J

        predictions = self.predict()

        return predictions

    def predict(self):
       
        # Computing K_new for X
        new_K = self.matriz_kernel[:, self.l:]
        #new_K = self.kernel(self.X, Xtest)
        f = self.alpha.T.dot(new_K)

        return f

    def kernel(self, X, Y ):

        matriz_distancia = cdist(X, Y, self.distancy )
        
        matriz_kernel = np.zeros((matriz_distancia.shape[0],matriz_distancia.shape[1]))

        sigma = retornar_sigma(matriz_distancia, self.k)
        
        for i in range(matriz_distancia.shape[0]):
            for j in range(matriz_distancia.shape[1]):
                matriz_kernel[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/2*np.power(2,sigma))

        return matriz_kernel


def propagar_LapRLS(dados, L, posicoes_rotulos, ordemObjetos, rotulos, Yl, medida_distancia, k, lambda_k, lambda_u):

    ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]

    dados_rotulados = dados[posicoes_rotulos,:]
    dados_nao_rotulados = dados[ordemNaoRotulado,:]
        
    propagacao_LapRLS = LapRLS( L, medida_distancia, k, lambda_k, lambda_u )

    f = propagacao_LapRLS.labelDiffusion(dados_rotulados, dados_nao_rotulados, Yl)
    f = f.T
    
    resultado = np.array(rotulos) 
    for i in range(f.shape[0]):
        rotulo = np.argmax(f[i,:]) + 1
        resultado[ordemNaoRotulado[i]] = rotulo
    
    return resultado
