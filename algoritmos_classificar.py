from processar_rotulos import one_hot
import numpy as np

def ordem_rotulos_primeiro(rotulos):

  #pegar posições existe rotulo
  posicoes_rotulos = []
  for i in range(rotulos.shape[0]):
    if not np.all(rotulos[i] == 0):
        posicoes_rotulos.append(i)

  # Reordenar as posições para os rotulados vim primeiro
  ordemObjetos = np.arange(rotulos.shape[0])
  # Retira dos indices os que são rotulados
  ordemObjetos = np.setdiff1d(ordemObjetos, posicoes_rotulos)
  # ordemObjetos será uma lista onde os indices dos objetos rotulados
  # vem primeiro, depois o resto em ordem crescente de indice
  ordemObjetos = np.concatenate((posicoes_rotulos,ordemObjetos))

  return posicoes_rotulos, ordemObjetos

def divisao_L(L,rotulos,posicoes_rotulos, ordemObjetos):

    ## Reordenacao da matriz Laplaciana, para manter os dados rotulados à frente
    L = L[ordemObjetos,:]
    L = L[:, ordemObjetos]

    # Armazenar numero de objetos rotulados, nao rotulados e total de objetos
    nRotulado = len(posicoes_rotulos)
    nNaoRotulado = L.shape[0]-nRotulado
    nObjetos = L.shape[0]

    # Extracao das submatrizes da matriz laplaciana
    LRotulado = L[0:nRotulado, 0:nRotulado]
    LNaoRotuladoRotulado = L[nRotulado:nObjetos, 0:nRotulado]
    LNaoRotulado = L[nRotulado:nObjetos, nRotulado:nObjetos]

    return LRotulado, LNaoRotuladoRotulado, LNaoRotulado

# GRF para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos
# saída: matriz de rótulos propagados
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


# RMGT para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e omega
# saída: matriz de rótulos propagados
def RMGT(W,rotulos,omega):

  print("inicializando RMGT... ")

  # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
  D = np.zeros(W.shape)
  np.fill_diagonal(D, np.sum(W, axis=1))
  # Calculo da matriz laplaciana
  L= 1.01*D - W

  posicoes_rotulos, ordemObjetos = ordem_rotulos_primeiro(rotulos)

  # Extracao das submatrizes da matriz laplaciana
  LRotulado, LNaoRotuladoRotulado, LNaoRotulado = divisao_L(L,rotulos,posicoes_rotulos,ordemObjetos)

  # da forma: [[],[],[],[],...,[]]
  matriz_rotulos = one_hot(rotulos)
  yl = matriz_rotulos[posicoes_rotulos,:]


  resultado = np.zeros((rotulos.shape[0],1),dtype=int)

  # Os rotulos originais continuam da mesma forma
  for i in range(len(posicoes_rotulos)):
    resultado[ordemObjetos[i]] = rotulos[ordemObjetos[i],0]


  vetor_1 = np.ones((LNaoRotulado.shape[0],1),dtype=int)
  vetor_2 = np.ones((LRotulado.shape[0],1),dtype=int)

  f1 = -np.linalg.inv(LNaoRotulado).dot(LNaoRotuladoRotulado.dot(yl))

  f2 = ((np.linalg.inv(LNaoRotulado).dot(vetor_1))/(vetor_1.T.dot((-np.linalg.inv(LNaoRotulado)).dot(vetor_1))))

  f3 = (W.shape[0]*omega.T-vetor_2.T.dot(yl+vetor_1.T.dot((-np.linalg.inv(LNaoRotulado)).dot(LNaoRotuladoRotulado.dot(yl)))))
    
  f4 = f2.dot(f3)

  f = f1 + f4 

  print(f)

  # Formatacao dos dados nao rotulados
  ordemNaoRotulado = ordemObjetos[len(posicoes_rotulos):]
  for i in range(len(ordemNaoRotulado)):
    posicao = np.argmax(f[i,:])
    resultado[ordemNaoRotulado[i]]= posicao+1

  return resultado


# LGC para calcular a matriz de rótulos propagados
# entrada: matriz de pesos, rótulos e parametro regularização
# saída: vetor de rótulos propagados
def LGC(W,rotulos,parametro_regularizacao):

  print("inicializando LGC...")

  # Calculo da matriz diagonal: Uma matriz de grau de cada um dos vertices
  D = np.zeros(W.shape)
  np.fill_diagonal(D, np.sum(W, axis=1))
  # Calculo da matriz laplaciana
  L = 1.01*D - W

  # Calculo da laplaciana normalizada
  matriz_identidade = np.eye(W.shape[0])
  D_inv_raiz = np.diag(1 / np.sqrt(np.diag(D)))
  L_normalizada = 1.01*matriz_identidade - D_inv_raiz.dot(W.dot(D_inv_raiz))

  # da forma: [[],[],[],[],...,[]]
  matriz_rotulos = one_hot(rotulos)
  
  f = np.linalg.inv(matriz_identidade - L_normalizada*parametro_regularizacao).dot(matriz_rotulos)

  print(f)

  resultado = np.zeros((rotulos.shape[0],1),dtype=int)

  for i in range(f.shape[0]):
    posicao = np.argmax(f[i,:])
    resultado[i]= posicao + 1

  return resultado

def propagar(W,rotulos,omega = 0, parametro_regularizacao = 0.99, algoritmo = "GRF"):
   
   if algoritmo == "GRF":
      return GRF(W,rotulos)
   
   elif algoritmo == "RMGT":
      return RMGT(W,rotulos,omega)
   
   elif algoritmo == "LGC":
      return LGC(W,rotulos,parametro_regularizacao)




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
