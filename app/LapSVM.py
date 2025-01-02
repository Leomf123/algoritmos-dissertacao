import numpy as np
from scipy.spatial.distance import cdist
import scipy.optimize as sco

from utils import retornar_sigma

# Classe LapSVM adaptada da classe LapSVM do repositório:
# https://github.com/HugoooPerrin/semi-supervised-learning

class LapSVM(object):

    def __init__(self, L, distancy, k, lambda_k, lambda_u):
        """
        Laplacian Support Vector Machines

        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        """
        self.L = L
        self.k = k
        self.distancy = distancy
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
        #print("inicializando LapSVM fit", end="... ")
        # Storing parameters
        self.l = X.shape[0]
        u = X_no_label.shape[0]
        n = self.l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        Y = np.diag(Y)
        
        # Memory optimization
        del X_no_label
        
        # Computing K with k(i,j) = kernel(i, j)
        #print('Computing kernel matrix', end='...')
        self.matriz_kernel = self.kernel(self.X, self.X)
        #print('done')

        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(self.l), np.zeros(self.l * u).reshape(self.l, u)], axis=1)

        ###########################################################################
        
        # Computing "almost" alpha
        #print('Inverting matrix', end='...')
        almost_alpha = np.linalg.inv(2 * self.lambda_k * np.identity(self.l + u) \
                                     + ((2 * self.lambda_u) / (self.l + u) ** 2) * self.L.dot(self.matriz_kernel)).dot(J.T).dot(Y)
        
        # Computing Q
        Q = Y.dot(J).dot(self.matriz_kernel).dot(almost_alpha)
        #print('done')
        
        # Memory optimization
        del J
        
        # Solving beta using scypy optimize function
        
        #print('Solving beta', end='...')
        
        e = np.ones(self.l)
        q = -e
        
        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)
        
        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))
        
        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / self.l) for _ in range(self.l)]
        
        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))
        
        def constraint_grad(beta):
            return np.diag(Y)

        
        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}
        
        # ===== Solving =====
        x0 = np.zeros(self.l)
        
        beta_hat = sco.minimize(objective_func, x0, jac=objective_grad, \
                                constraints=cons, bounds=bounds, method='SLSQP')['x']
                
        #print('done')
        
        # Computing final alpha
        #print('Computing alpha', end='...')
        self.alpha = almost_alpha.dot(beta_hat)
        #print('done')
        
        del almost_alpha, Q
        
        ###########################################################################
        
        # Finding optimal decision boundary b using labeled data
        #new_K = self.kernel(self.X, X)
        indices_X = np.arange(self.l)
        #new_K = self.kernel(self.X, X)
        new_K = self.matriz_kernel[:, indices_X]
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == np.diag(Y)) / len(predictions))
        
        bs = np.linspace(0, 1, num=101)
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
        #print("inicializando LapSVM predict", end="... ")
        # Computing K_new for X
        #new_K = self.kernel(self.X, Xtest)
        new_K = self.matriz_kernel[self.l:, :]
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        predictions = np.array((f > self.b) * 1)
        #print("feito")
        return predictions
    
    def kernel(self, X, Y ):

        matriz_distancias = cdist(X, Y, self.distancy )
        
        sigma = retornar_sigma(matriz_distancias, self.k)
        
        return np.exp(-0.5 * (matriz_distancias ** 2) / (sigma ** 2))
    


def propagar_LapSVM(dados, L, posicoes_rotulos, ordemObjetos, rotulos, classes, medida_distancia, k, lambda_k, lambda_u):


    # Reordenar as posições para os rotulados vim primeiro
    posicoes_sem_rotulos = np.arange(rotulos.shape[0])
    # Retira dos indices os que são rotulados
    posicoes_sem_rotulos = np.setdiff1d(posicoes_sem_rotulos, posicoes_rotulos)

    dados_rotulados = dados[posicoes_rotulos,:]
    dados_nao_rotulados = dados[posicoes_sem_rotulos,:]
    Yl = rotulos[posicoes_rotulos]

    propagacao_LapSVM = LapSVM( L, medida_distancia, k, lambda_k, lambda_u )

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
                yl_one_versus_all[j] = -1
        
        rotulos_propagados = propagacao_LapSVM.labelDiffusion(dados_rotulados, dados_nao_rotulados, yl_one_versus_all)

        # Formatacao dos dados nao rotulados
        ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
        for i in range(len(ordemNaoRotulado)):
        # O que propagou foi o positivo (1)
            if rotulos_propagados[i]  == 1:
                resultado[ordemNaoRotulado[i]]= rotulo

    return resultado
