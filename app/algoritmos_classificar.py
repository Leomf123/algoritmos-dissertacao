from processar_rotulos import one_hot
import numpy as np
from LapRLS import propagar_LapRLS
from LapSVM import propagar_LapSVM

# GRF para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos
# saída: matriz de rótulos propagados
def GRF(posicoes_rotulos, ordemObjetos, formula_comum_grf_rmgt, rotulos):

  #print("inicializando GRF", end="... ")

  f = -formula_comum_grf_rmgt

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
  
  resultado = np.array(rotulos) 
  for i in range(f.shape[0]):
    rotulo = np.argmax(f[i,:]) + 1
    resultado[ordemNaoRotulado[i]] = rotulo
  
  #print("feito")

  return resultado


# RMGT para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e omega
# saída: matriz de rótulos propagados
def RMGT(posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotulado_inv, formula_comum_grf_rmgt, yl, rotulos, omega):

  #print("inicializando RMGT", end="... ")

  vetor_1 = np.ones((LNaoRotulado_inv.shape[0],1),dtype=int)
  vetor_2 = np.ones((LRotulado.shape[0],1),dtype=int)

  f1 = (LNaoRotulado_inv.dot(vetor_1)) / (vetor_1.T.dot(LNaoRotulado_inv).dot(vetor_1))

  f2 = len(ordemObjetos)*omega.T - vetor_2.T.dot(yl) + vetor_1.T.dot(formula_comum_grf_rmgt)

  f = -formula_comum_grf_rmgt + f1.dot(f2)

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]

  resultado = np.array(rotulos) 
  for i in range(f.shape[0]):
    rotulo = np.argmax(f[i,:]) + 1
    resultado[ordemNaoRotulado[i]] = rotulo
  
  #print("feito")

  return resultado


# LGC para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e parametro regularização
# saída: vetor de rótulos propagados
def LGC(L_normalizada, matriz_rotulos, ordemObjetos, posicoes_rotulos, rotulos, parametro_regularizacao):

  #print("inicializando LGC", end="... ")

  # Calculo da laplaciana normalizada
  matriz_identidade = np.eye(matriz_rotulos.shape[0])
 
  f = np.linalg.inv(matriz_identidade + L_normalizada/parametro_regularizacao).dot(matriz_rotulos)

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]

  resultado = np.array(rotulos) 
  for i in range(ordemNaoRotulado.shape[0]):
    rotulo = np.argmax(f[ordemNaoRotulado[i],:]) + 1
    resultado[ordemNaoRotulado[i]] = rotulo

  #print("feito")

  return resultado

def propagar(dados, L, posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotulado_inv, formula_comum_grf_rmgt, L_normalizada, yl, rotulos, matriz_rotulos, classes, medida_distancia, k, lambda_k, lambda_u, omega = 0, parametro_regularizacao = 0.99, algoritmo = "GRF"):
   
   if algoritmo == "GRF":
      return GRF(posicoes_rotulos, ordemObjetos, formula_comum_grf_rmgt, rotulos)
   
   elif algoritmo == "RMGT":
      return RMGT(posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotulado_inv, formula_comum_grf_rmgt, yl, rotulos, omega)
   
   elif algoritmo == "LGC":
      return LGC(L_normalizada, matriz_rotulos, ordemObjetos, posicoes_rotulos, rotulos, parametro_regularizacao)
   
   elif algoritmo == "LapRLS":
      return propagar_LapRLS(dados, L, posicoes_rotulos, ordemObjetos, rotulos, yl, medida_distancia, k, lambda_k, lambda_u)
   
   elif algoritmo == "LapSVM":
      return propagar_LapSVM(dados, L, posicoes_rotulos, ordemObjetos, rotulos, classes, medida_distancia, k, lambda_k, lambda_u)
