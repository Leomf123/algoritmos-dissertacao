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

class LapRLS(object):

    def __init__(self, L, distancy, k, lambda_k, lambda_u,
                 learning_rate=0.1, n_iterations=100 , solver='closed-form'):
        """
        Laplacian Regularized Least Square algorithm

        Parameters
        ----------
        n_neighbors : integer
            Number of neighbors to use when constructing the graph
        lambda_k : float
        lambda_u : float
        Learning_rate: float
            Learning rate of the gradient descent
        n_iterations : integer
        solver : string ('closed-form' or 'gradient-descent' or 'L-BFGS-B')
            The method to use when solving optimization problem
        """
        self.L = L
        self.distancy = distancy
        self.k = k
        self.lambda_k = lambda_k
        self.lambda_u = lambda_u
        self.learning_rate = learning_rate
        self.n_iterations = n_iterations
        self.solver = solver
        

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
        #print("inicializando LapRLS fit", end="... ")
        # Storing parameters
        l = X.shape[0]
        u = X_no_label.shape[0]
        n = l + u
        
        # Building main matrices
        self.X = np.concatenate([X, X_no_label], axis=0)
        self.Y = np.concatenate([Y, np.zeros(u)])
        
                
        # Memory optimization
        del X_no_label

        # Computing K with k(i,j) = kernel(i, j)
        #print('Computing kernel matrix', end='...')
        K = self.kernel(self.X, self.X)
        #print('done')

        # Creating matrix J (diag with l x 1 and u x 0)
        J = np.diag(np.concatenate([np.ones(l), np.zeros(u)]))
        
        if self.solver == 'closed-form':
            
            # Computing final matrix
            #print('Computing final matrix', end='...')
            final = (J.dot(K) + self.lambda_k * l * np.identity(l + u) + ((self.lambda_u * l) / (l + u) ** 2) * self.L.dot(K))
            #print('done')
        
            # Solving optimization problem
            #print('Computing closed-form solution', end='...')
            self.alpha = np.linalg.inv(final).dot(self.Y)
            #print('done')
            
            # Memory optimization
            del self.Y, J
        
        elif self.solver == 'gradient-descent':
            """
            If solver is Gradient-descent then a learning rate and an iteration number must be provided
            """
            
            #print('Performing gradient descent...')
            
            # Initializing alpha
            self.alpha = np.zeros(n)

            # Computing final matrices
            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(self.L)).dot(K)

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
                        
            # Memory optimization
            del self.Y, J
        
            for i in range(self.n_iterations + 1):
                
                # Computing gradient & updating alpha
                self.alpha -= self.learning_rate * RLS_grad(self.alpha)
                
                #if i % 50 == 0:
                    #print("\r[%d / %d]" % (i, self.n_iterations) ,end = "")
                    
            #print('\n')
        
        elif self.solver == 'L-BFGS-B':
            
            #print('Performing L-BFGS-B', end='...')
            
            # Initializing alpha
            x0 = np.zeros(n)

            # Computing final matrices
            grad_part1 = -(2 / l) * K.dot(self.Y)
            grad_part2 = ((2 / l) * K.dot(J) + 2 * self.lambda_k * np.identity(l + u) + \
                        ((2 * self.lambda_u) / (l + u) ** 2) * K.dot(self.L)).dot(K)

            def RLS(alpha):
                return np.squeeze(np.array((1 / l) * (self.Y - J.dot(K).dot(alpha)).T.dot((self.Y - J.dot(K).dot(alpha))) \
                        + self.lambda_k * alpha.dot(K).dot(alpha) + (self.lambda_u / n ** 2) \
                        * alpha.dot(K).dot(self.L).dot(K).dot(alpha)))

            def RLS_grad(alpha):
                return np.squeeze(np.array(grad_part1 + grad_part2.dot(alpha)))
            
            self.alpha, _, _ = sco.fmin_l_bfgs_b(RLS, x0, RLS_grad, args=(), pgtol=1e-30, factr =1e-30)
            
            #print('done')
                                  
        # Finding optimal decision boundary b using labeled data
        indices_X = np.arange(l)  
        #new_K = self.kernel(self.X, X)
        new_K = K[:, indices_X]
        f = np.squeeze(np.array(self.alpha)).dot(new_K)
        
        def to_minimize(b):
            predictions = np.array((f > b) * 1)
            return - (sum(predictions == Y) / len(predictions))

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
        #print("inicializando LapRLS predict", end="... ")
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
        

        propagacao_LapRLS.fit(dados_rotulados, dados_nao_rotulados, yl_one_versus_all)

        rotulos_propagados = propagacao_LapRLS.predict(dados_nao_rotulados)


        # Formatacao dos dados nao rotulados
        ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
        for i in range(len(ordemNaoRotulado)):
        # O que propagou foi o positivo (1)
            if rotulos_propagados[i]  == 1:
                resultado[ordemNaoRotulado[i]]= rotulo   
    
    return resultado


