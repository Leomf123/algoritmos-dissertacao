from processar_rotulos import one_hot
import numpy as np
from LapRLS import propagar_LapRLS
from LapSVM import propagar_LapSVM

# GRF para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos
# saída: matriz de rótulos propagados
def GRF(posicoes_rotulos, ordemObjetos, LNaoRotuladoRotulado, LNaoRotulado, yl, rotulos):

  print("inicializando GRF", end="... ")

  resultado = np.zeros((rotulos.shape[0]),dtype=int)
  # Os rotulos originais continuam da mesma forma
  resultado[ordemObjetos] = rotulos[ordemObjetos]

  f = -np.linalg.inv(LNaoRotulado).dot(LNaoRotuladoRotulado.dot(yl))

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
  for i in range(len(ordemNaoRotulado)):
    posicao = np.argmax(f[i,:])
    resultado[ordemNaoRotulado[i]]= posicao + 1
  
  print("feito")

  return resultado


# RMGT para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e omega
# saída: matriz de rótulos propagados
def RMGT(posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, yl, rotulos, omega):

  print("inicializando RMGT", end="... ")

  resultado = np.zeros((rotulos.shape[0]),dtype=int)

  # Os rotulos originais continuam da mesma forma
  resultado[ordemObjetos] = rotulos[ordemObjetos]


  vetor_1 = np.ones((LNaoRotulado.shape[0],1),dtype=int)
  vetor_2 = np.ones((LRotulado.shape[0],1),dtype=int)

  f1 = -np.linalg.inv(LNaoRotulado).dot(LNaoRotuladoRotulado.dot(yl))

  f2 = ((np.linalg.inv(LNaoRotulado).dot(vetor_1))/(vetor_1.T.dot((-np.linalg.inv(LNaoRotulado)).dot(vetor_1))))

  f3 = (len(ordemObjetos)*omega.T-vetor_2.T.dot(yl+vetor_1.T.dot((-np.linalg.inv(LNaoRotulado)).dot(LNaoRotuladoRotulado.dot(yl)))))
    
  f4 = f2.dot(f3)

  f = f1 + f4 

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
  for i in range(len(ordemNaoRotulado)):
    posicao = np.argmax(f[i,:])
    resultado[ordemNaoRotulado[i]]= posicao + 1
  
  print("feito")

  return resultado


# LGC para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e parametro regularização
# saída: vetor de rótulos propagados
def LGC(L_normalizada, matriz_rotulos, rotulos, parametro_regularizacao):

  print("inicializando LGC", end="... ")

  # Calculo da laplaciana normalizada
  matriz_identidade = np.eye(matriz_rotulos.shape[0])
 
  f = np.linalg.inv(matriz_identidade + L_normalizada/parametro_regularizacao).dot(matriz_rotulos)

  resultado = np.zeros((rotulos.shape[0]),dtype=int)

  for i in range(f.shape[0]):
    posicao = np.argmax(f[i,:])
    resultado[i]= posicao + 1
  
  print("feito")

  return resultado

def propagar(dados, L, posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, L_normalizada, yl, rotulos, matriz_rotulos, classes, medida_distancia, k, lambda_k, lambda_u, omega = 0, parametro_regularizacao = 0.99, algoritmo = "GRF"):
   
   if algoritmo == "GRF":
      return GRF(posicoes_rotulos, ordemObjetos, LNaoRotuladoRotulado, LNaoRotulado, yl, rotulos)
   
   elif algoritmo == "RMGT":
      return RMGT(posicoes_rotulos, ordemObjetos, LRotulado, LNaoRotuladoRotulado, LNaoRotulado, yl, rotulos, omega)
   
   elif algoritmo == "LGC":
      return LGC(L_normalizada, matriz_rotulos, rotulos, parametro_regularizacao)
   
   elif algoritmo == "LapRLS":
      return propagar_LapRLS(dados, L, posicoes_rotulos, ordemObjetos, rotulos, classes, medida_distancia, k, lambda_k, lambda_u)
   
   elif algoritmo == "LapSVM":
      return propagar_LapSVM(dados, L, posicoes_rotulos, ordemObjetos, rotulos, classes, medida_distancia, k, lambda_k, lambda_u)



   """
   
# LGC para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e parametro regularização
# saída: vetor de rótulos propagados
def LGC(W,rotulos,parametro_regularizacao):

  print("inicializando LGC...")

  # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
  D = np.zeros(W.shape)
  np.fill_diagonal(D, np.sum(W, axis=1))
  # Calculo da matriz laplaciana
  L = D - W

  # Calculo da laplaciana normalizada
  matriz_identidade = np.eye(W.shape[0])
  D_inv_raiz = np.diag(1 / np.sqrt(np.diag(D)))
  L_normalizada = matriz_identidade - D_inv_raiz.dot(W.dot(D_inv_raiz))

  resultado = np.zeros((rotulos.shape[0],1),dtype=int)

  # classes
  classes = []
  for k in range(len(rotulos)):
    if rotulos[k][0] != 0:
      if not rotulos[k][0] in classes:
        classes.append(rotulos[k][0])

  # Pela quantidade de classes, one-vesus-all
  for i in range(len(classes)):
    # rotulo da vez
    rotulo = classes[i]
    print(rotulo)
    # transforma rotulo da vez 1, resto -1
    y_one_versus_all = np.zeros((rotulos.shape[0],1))
    for j in range(rotulos.shape[0]):
        if rotulos[j][0] == rotulo:
          y_one_versus_all[j][0] = 1
        else:
          y_one_versus_all[j][0] = -1
  
    f = np.linalg.inv(matriz_identidade + L_normalizada/parametro_regularizacao).dot(y_one_versus_all)

    print(f)

    for i in range(f.shape[0]):
       # O que propagou foi o positivo (1)
       if 1*np.sign(f[i,0]) >= 0:
          resultado[i]= rotulo
       else:
          # Ele não tinha rotulo?
          if resultado[i] == 0:
            resultado[i] = -1
          # se já tem rotulo continua o antigo pq já foi propagado antes,
          # o que faz sentido ser -1 agora
          # pq o antigo agora faz parte do -1

  return resultado
   """

#GRF com CMN
'''
def GRF(W,rotulos):

  print("inicializando GRF... ")

  # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
  D = np.zeros(W.shape)
  np.fill_diagonal(D, np.sum(W, axis=1))
  # Calculo da matriz laplaciana
  L= 1.01*D - W

  posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos)

  # Extracao das submatrizes da matriz laplaciana
  LRotulado, LNaoRotuladoRotulado, LNaoRotulado = divisao_L(L,rotulos,posicoes_rotulos,ordemObjetos)

  # da forma: [[],[],[],[],...,[]]
  yl = rotulos[posicoes_rotulos]
  resultado = np.zeros((rotulos.shape[0],1),dtype=int)

  # classes
  classes = []
  for k in range(len(yl)):
    if not yl[k][0] in classes:
      classes.append(yl[k][0])

  # Os rotulos originais continuam da mesma forma
  for i in range(len(posicoes_rotulos)):
    resultado[ordemObjetos[i]] = yl[i,0]

  # Pela quantidade de classes, one-vesus-all
  for i in range(len(classes)):
    # rotulo da vez
    rotulo = classes[i]
    print(rotulo)
    # q vai ser usado no CNM
    q = 0
    for r in range(len(posicoes_rotulos)):
       if rotulos[posicoes_rotulos[r]] == rotulo:
          q += 1
    print(q)
    print(len(posicoes_rotulos))
    q = 1 / len(classes)

    print(q)
       
    # transforma rotulo da vez 1, resto 0
    yl_one_versus_all = np.zeros((yl.shape[0],1))
    for j in range(yl.shape[0]):
        if yl[j][0] == rotulo:
          yl_one_versus_all[j][0] = 1    

    f = -np.linalg.inv(LNaoRotulado).dot(LNaoRotuladoRotulado.dot(yl_one_versus_all))
    
    print(f.shape)
    f_sum1 = np.sum(f)
    print(f_sum1)
    f_sum2 = np.sum(1-f)
    print(f_sum2)

    # Formatacao dos dados nao rotulados
    ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
    for i in range(len(ordemNaoRotulado)):
       # O que propagou foi o positivo (1)
       # 1*np.sign(f[i,0])
       # CNM:
       if q * (f[i,0] / f_sum1) > (1 - q)* ((1 - f[i,0])/f_sum2):
       #if f[i,0] > 0.5:
          resultado[ordemNaoRotulado[i]]= rotulo
       #else:
          # Ele não tinha rotulo?
          #if resultado[ordemNaoRotulado[i]] == 0:
          #  resultado[ordemNaoRotulado[i]] = -1
      
          # se já tem rotulo continua o antigo pq já foi propagado antes,
          # o que faz sentido ser -1 agora
          # pq o antigo agora faz parte do -1    

  return resultado
'''

#GRF com sign
'''
def GRF(W,rotulos):

  print("inicializando GRF... ")

  # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
  D = np.zeros(W.shape)
  np.fill_diagonal(D, np.sum(W, axis=1))
  # Calculo da matriz laplaciana
  L= 1.01*D - W

  posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos)

  # Extracao das submatrizes da matriz laplaciana
  LRotulado, LNaoRotuladoRotulado, LNaoRotulado = divisao_L(L,rotulos,posicoes_rotulos,ordemObjetos)

  # da forma: [[],[],[],[],...,[]]
  yl = rotulos[posicoes_rotulos]
  resultado = np.zeros((rotulos.shape[0],1),dtype=int)

  # classes
  classes = []
  for k in range(len(yl)):
    if not yl[k][0] in classes:
      classes.append(yl[k][0])

  # Os rotulos originais continuam da mesma forma
  for i in range(len(posicoes_rotulos)):
    resultado[ordemObjetos[i]] = yl[i,0]

  # Pela quantidade de classes, one-vesus-all
  for i in range(len(classes)):
    # rotulo da vez
    rotulo = classes[i]
    print(rotulo)
    # transforma rotulo da vez 1, resto -1
    yl_one_versus_all = np.zeros((yl.shape[0],1))
    for j in range(yl.shape[0]):
        if yl[j][0] == rotulo:
          yl_one_versus_all[j][0] = 1
        else:
          yl_one_versus_all[j][0] = -1
    

    f = -np.linalg.inv(LNaoRotulado).dot(LNaoRotuladoRotulado.dot(yl_one_versus_all))

    # Formatacao dos dados nao rotulados
    ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
    for i in range(len(ordemNaoRotulado)):
       # O que propagou foi o positivo (1)
       if 1*np.sign(f[i,0]) >= 0:
          resultado[ordemNaoRotulado[i]]= rotulo
       else:
          # Ele não tinha rotulo?
          if resultado[ordemNaoRotulado[i]] == 0:
            resultado[ordemNaoRotulado[i]] = -1
          # se já tem rotulo continua o antigo pq já foi propagado antes,
          # o que faz sentido ser -1 agora
          # pq o antigo agora faz parte do -1    

  return resultado
'''