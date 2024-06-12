import numpy as np
import random
from sklearn.metrics import balanced_accuracy_score, f1_score

# função para retirar rotulos dado uma porcentagem
# entrada: vetor numpy de rótulos e uma porcetagem pra manter
# saída: vertor numpy de rotulos com alguns não rotulados
def retirar_rotulos(rotulos, porcentagem_manter, classes, seed = 100):
 
 random.seed(seed)
 
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
  denominador2 = 0
  numerador2 = 0

  for i in range(rotulos_originais.shape[0]):
    if rotulos_semissupervisionado[i] == 0:
      denominador += 1
      if rotulos_originais[i] == rotulos_propagados[i]:
        numerador = numerador + 1
    else:
      denominador2 +=1
      if rotulos_originais[i] == rotulos_propagados[i]:
        numerador2 += 1

  print("rotulados já")
  print(denominador2)
  print(numerador2)
  print(numerador2/denominador2)
  print("nao rotulados já")
  print(denominador)
  print(numerador)
  
  return numerador/denominador


# função para one-hot rótulos
# entrada: vetor de rótulos
# saída: matriz de rótulos one-hot
def one_hot(rotulos):

  matriz_one_hot = np.zeros((len(rotulos),np.max(rotulos)))

  for i in range(len(rotulos)):
    if rotulos[i] !=0:
      matriz_one_hot[i][rotulos[i]-1] = 1

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

def medidas_qualidade(posicoes_rotulos, ordemObjetos, rotulos, rotulos_propagados):
  '''
  rotulos: rotulos originais
  rotulos_propagados: rotulos propagados
  '''
  posicao_sem_rotulos = ordemObjetos[len(posicoes_rotulos):]
  y_true = rotulos[posicao_sem_rotulos]
  y_pred = rotulos_propagados[posicao_sem_rotulos]

  nRotulos = np.count_nonzero(y_pred == 0)

  acuracia = balanced_accuracy_score(y_true, y_pred)
  f_measure = f1_score(y_true, y_pred, average='macro')

  return acuracia, f_measure, nRotulos

  

