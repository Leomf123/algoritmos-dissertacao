#!/usr/bin/env python
# -*- coding: utf-8 -*-



#=========================================================================================================
#================================ 0. MODULE


import numpy as np
from scipy.spatial.distance import cdist
import scipy.optimize as sco

from utils import retornar_sigma

#=========================================================================================================
#================================ 1. ALGORITHM


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
    

    def fit(self, X, X_no_label, Y):
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
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        Y = np.diag(Y)
        
        # Memory optimization
        del X_no_label
        
        # Computing K with k(i,j) = kernel(i, j)
        #print('Computing kernel matrix', end='...')
        K = self.kernel(self.X, self.X)
        #print('done')

        # Creating matrix J [I (l x l), 0 (l x (l+u))]
        J = np.concatenate([np.identity(l), np.zeros(l * u).reshape(l, u)], axis=1)

        ###########################################################################
        
        # Computing "almost" alpha
        #print('Inverting matrix', end='...')
        almost_alpha = np.linalg.inv(2 * self.lambda_k * np.identity(l + u) \
                                     + ((2 * self.lambda_u) / (l + u) ** 2) * self.L.dot(K)).dot(J.T).dot(Y)
        
        # Computing Q
        Q = Y.dot(J).dot(K).dot(almost_alpha)
        #print('done')
        
        # Memory optimization
        del J
        
        # Solving beta using scypy optimize function
        
        #print('Solving beta', end='...')
        
        e = np.ones(l)
        q = -e
        
        # ===== Objectives =====
        def objective_func(beta):
            return (1 / 2) * beta.dot(Q).dot(beta) + q.dot(beta)
        
        def objective_grad(beta):
            return np.squeeze(np.array(beta.T.dot(Q) + q))
        
        # =====Constraint(1)=====
        #   0 <= beta_i <= 1 / l
        bounds = [(0, 1 / l) for _ in range(l)]
        
        # =====Constraint(2)=====
        #  Y.dot(beta) = 0
        def constraint_func(beta):
            return beta.dot(np.diag(Y))
        
        def constraint_grad(beta):
            return np.diag(Y)

        def hessian_zero(beta):
            return np.zeros((len(beta), len(beta)))
        
        cons = {'type': 'eq', 'fun': constraint_func, 'jac': constraint_grad}
        
        # ===== Solving =====
        x0 = np.zeros(l)
        
        beta_hat = sco.minimize(objective_func, x0, jac=objective_grad, hess=hessian_zero, \
                                constraints=cons, bounds=bounds, method='trust-constr')['x']
        #beta_hat = sco.minimize(objective_func, x0, jac=objective_grad, hess=objective_hess,
        #                constraints=cons, bounds=bounds, method='trust-constr')['x']
        #print('done')
        
        # Computing final alpha
        #print('Computing alpha', end='...')
        self.alpha = almost_alpha.dot(beta_hat)
        #print('done')
        
        del almost_alpha, Q
        
        ###########################################################################
        
        # Finding optimal decision boundary b using labeled data
        #new_K = self.kernel(self.X, X)
        indices_X = np.arange(l)  
        #new_K = self.kernel(self.X, X)
        new_K = K[:, indices_X]
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == np.diag(Y)) / len(predictions))
        
        bs = np.linspace(0, 1, num=101)
        res = np.array([to_minimize(b) for b in bs])
        self.b = bs[res == np.min(res)][0]
        
        #print("feito")

    def predict(self, Xtest):
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
        new_K = self.kernel(self.X, Xtest)
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        predictions = np.array((f > self.b) * 1)
        #print("feito")
        return predictions
    

    def accuracy(self, Xtest, Ytrue):
        """
        Parameters
        ----------
        Xtest : ndarray shape (n_samples, n_features)
            Test data
        Ytrue : ndarray shape (n_samples, )
            Test labels
        """
        predictions = self.predict(Xtest)
        accuracy = sum(predictions == Ytrue) / len(predictions)
        print('Accuracy: {}%'.format(round(accuracy * 100, 2)))

    
    def kernel(self, X, Y ):

        matriz_distancia = cdist(X, Y, self.distancy )
        
        matriz_kernel = np.zeros((matriz_distancia.shape[0],matriz_distancia.shape[1]))

        sigma = retornar_sigma(matriz_distancia, self.k)
        
        for i in range(matriz_distancia.shape[0]):
            for j in range(matriz_distancia.shape[1]):
                matriz_kernel[i][j] = np.exp(-1*np.power(2,matriz_distancia[i][j])/2*np.power(2,sigma))

        return matriz_kernel
    


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
                yl_one_versus_all[j] = 0
        

        propagacao_LapSVM.fit(dados_rotulados, dados_nao_rotulados, yl_one_versus_all)

        rotulos_propagados = propagacao_LapSVM.predict(dados_nao_rotulados)

        # Formatacao dos dados nao rotulados
        ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
        for i in range(len(ordemNaoRotulado)):
        # O que propagou foi o positivo (1)
            if rotulos_propagados[i]  == 1:
                resultado[ordemNaoRotulado[i]]= rotulo

    return resultado
