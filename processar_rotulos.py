import numpy as np
import random

# função para retirar rotulos dado uma porcentagem
# entrada: vetor numpy de rótulos e uma porcetagem pra manter
# saída: vertor numpy de rotulos com alguns não rotulados
def retirar_rotulos(rotulos, porcentagem_manter,classes):
 
 quantidade_retirar = int(len(rotulos)*(1 - porcentagem_manter))
 if len(rotulos)-quantidade_retirar < len(classes):
   quantidade_retirar = len(rotulos)-len(classes)

 rotulos_semissupervisionado = np.array(rotulos)
 
 # Garante que guardo pelo menos 1 de cada classe
 posicoes_protegidas = np.zeros(len(classes))
 for i in range(len(classes)):
   posicoes = [j for j, v in enumerate(rotulos) if v == classes[i]]
   posicao_aleatoria = random.randint(0, len(posicoes)-1)
   posicoes_protegidas[i] = posicoes[posicao_aleatoria]
 
 for i in range(quantidade_retirar):
  while(True):
    posicao_aleatoria = random.randint(0, len(rotulos)-1)
    if rotulos_semissupervisionado[posicao_aleatoria] != 0 and not posicao_aleatoria in posicoes_protegidas:
      break

  rotulos_semissupervisionado[posicao_aleatoria] = 0

 return rotulos_semissupervisionado


def acuracia(rotulos_originais, rotulos_propagados,rotulos_semissupervisionado):

  denominador = 0
  numerador = 0

  for i in range(rotulos_originais.shape[0]):
    if rotulos_semissupervisionado[i] == 0:
      denominador += 1
      if rotulos_originais[i] == rotulos_propagados[i]:
        numerador = numerador + 1

  print(denominador)
  print(numerador)
  
  return numerador/denominador


# função para one-hot rótulos
# entrada: vetor de rótulos
# saída: matriz de rótulos one-hot
def one_hot(rotulos):
  
  rotulos_reshape = rotulos.reshape(1,-1)[0]

  matriz_one_hot = np.zeros((len(rotulos_reshape),np.max(rotulos_reshape)))

  for i in range(len(rotulos_reshape)):
    if rotulos_reshape[i] !=0:
      matriz_one_hot[i][rotulos_reshape[i]-1] = 1

  return matriz_one_hot

# -------------------------------------------------
# função para rótulos
# entrada: matriz de rótulos one-hot
# saída: vetor de rótulos
def reverso_one_hot(matriz_one_hot):

  rotulos = np.zeros(matriz_one_hot.shape[0])

  print(rotulos.shape)

  for i in range(matriz_one_hot.shape[0]):
    if not np.all(np.equal(matriz_one_hot[i], 0)):
      valor = np.where(matriz_one_hot[i] == 1)
      print(valor)
      rotulos[i]= valor[0][0]+1

    else:
      rotulos[i] =0

  return rotulos

