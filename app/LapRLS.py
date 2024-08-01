import numpy as np
from scipy.spatial.distance import cdist

from utils import retornar_sigma

# Classe LapRLS adaptada da classe LapRLS do repositório:
# https://github.com/HugoooPerrin/semi-supervised-learning

class LapRLS(object):

    def __init__(self, L, distancy, k, lambda_k, lambda_u):
        """
        Laplacian Regularized Least Square algorithm

        Parameters
        ----------
        lambda_k : float
        lambda_u : float
        solver : string ('closed-form')
            The method to use when solving optimization problem
        """
        self.L = L
        self.distancy = distancy
        self.k = k
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        

    def labelDiffusion(self, X, X_no_label, Y):
        """
        Fit the model
        
        Parameters
        ----------
        X : ndarray shape (n_labeled_samples, n_features)
            Labeled data
        X_no_label : ndarray shape (n_unlabeled_samples, n_features)
            Unlabeled data
        Y : ndarray shape (n_labeled_samples,)
            Labels
        """
        #print("inicializando LapRLS fit", end="... ")
        # Storing parameters
        self.l = X.shape[0]
        u = X_no_label.shape[0]
        n = self.l + u

        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        self.Y = np.concatenate([Y, np.zeros(u)])
        
                
        # Memory optimization
        del X_no_label

        # Computing K with k(i,j) = kernel(i, j)
        #print('Computing kernel matrix', end='...')
        self.matriz_kernel = self.kernel(self.X, self.X)
        #print('done')

        # Creating matrix J (diag with l x 1 and u x 0)
        J = np.diag(np.concatenate([np.ones(self.l), np.zeros(u)]))
           
        # Computing final matrix
        #print('Computing final matrix', end='...')
        #final = (J.dot(self.matriz_kernel) + self.lambda_k * self.l * np.identity(self.l + u) + ((self.lambda_u * self.l) / (self.l + u) ** 2) * self.L.dot(self.matriz_kernel))
        final = (J.dot(self.matriz_kernel) + self.lambda_k * self.l * np.identity(self.l + u) + self.lambda_u * self.l * self.L.dot(self.matriz_kernel))
        #print('done')
        
        # Solving optimization problem
        #print('Computing closed-form solution', end='...')
        self.alpha = np.linalg.inv(final).dot(self.Y)
        #print('done')
            
        # Memory optimization
        del self.Y, J
                                  
        # Finding optimal decision boundary b using labeled data
        indices_X = np.arange(self.l)  
        #new_K = self.kernel(self.X, X)
        new_K = self.matriz_kernel[:, indices_X]
        f = np.squeeze(np.array(self.alpha)).dot(new_K)

        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == Y) / len(predictions))

        bs = np.linspace(0, 1, num=1000001)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]

        #print("feito")

        predictions = self.predict()

        return predictions

    def predict(self):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
            
        Returns
        -------
        predictions : ndarray shape (n_samples, )
            Predicted labels for Xtest
        """
        #print("inicializando LapRLS predict", end="... ")
        # Computing K_new for X
        new_K = self.matriz_kernel[:, self.l:]
        #new_K = self.kernel(self.X, Xtest)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        predictions = np.array((f > self.b) * 1)
        
        #print("feito")
        return predictions

    def kernel(self, X, Y ):

        matriz_distancia = cdist(X, Y, self.distancy )
        
        matriz_kernel = np.zeros((matriz_distancia.shape[0],matriz_distancia.shape[1]))

        sigma = retornar_sigma(matriz_distancia, self.k)
        
        for i in range(matriz_distancia.shape[0]):
            for j in range(matriz_distancia.shape[1]):
                matriz_kernel[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/2*np.power(2,sigma))

        return matriz_kernel



def propagar_LapRLS(dados, L, posicoes_rotulos, ordemObjetos, rotulos, classes, medida_distancia, k, lambda_k, lambda_u):

    # Reordenar as posições para os rotulados vim primeiro
    posicoes_sem_rotulos = np.arange(rotulos.shape[0])
    # Retira dos indices os que são rotulados
    posicoes_sem_rotulos = np.setdiff1d(posicoes_sem_rotulos, posicoes_rotulos)

    dados_rotulados = dados[posicoes_rotulos,:]
    dados_nao_rotulados = dados[posicoes_sem_rotulos,:]
    Yl = rotulos[posicoes_rotulos]
    
    propagacao_LapRLS = LapRLS( L, medida_distancia, k, lambda_k, lambda_u )

    resultado = np.zeros((rotulos.shape[0]),dtype=int)

    # Os rotulos originais continuam da mesma forma
    for i in range(len(posicoes_rotulos)):
        resultado[ordemObjetos[i]] = Yl[i]

    # Pela quantidade de classes, one-vesus-all
    for i in range(len(classes)):
        # rotulo da vez
        rotulo = classes[i]
        # transforma rotulo da vez 1, resto -1
        yl_one_versus_all = np.zeros((Yl.shape[0]))
        for j in range(Yl.shape[0]):
            if Yl[j] == rotulo:
                yl_one_versus_all[j] = 1
            else:
                yl_one_versus_all[j] = 0
        

        rotulos_propagados = propagacao_LapRLS.labelDiffusion(dados_rotulados, dados_nao_rotulados, yl_one_versus_all)

        # Formatacao dos dados nao rotulados
        ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
        for i in range(len(ordemNaoRotulado)):
        # O que propagou foi o positivo (1)
            if rotulos_propagados[i]  == 1:
                resultado[ordemNaoRotulado[i]]= rotulo   
    
    return resultado

