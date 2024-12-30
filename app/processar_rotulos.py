import numpy as np
import random
from collections import defaultdict
from sklearn.metrics import balanced_accuracy_score, f1_score

# função para retirar rotulos dado uma porcentagem
# entrada: vetor numpy de rótulos e uma porcetagem pra manter
# saída: vertor numpy de rotulos com alguns não rotulados
def retirar_rotulos(rotulos, porcentagem_manter, classes, seed=100):

  random.seed(seed)
  np.random.seed(seed)

  # Calcula a quantidade de rótulos a remover
  quantidade_retirar = int(len(rotulos) * (1 - porcentagem_manter))
  if len(rotulos) - quantidade_retirar < len(classes):
    quantidade_retirar = len(rotulos) - len(classes)

  # Pré-processa as posições de cada classe
  classe_para_posicoes = defaultdict(list)
  for i, r in enumerate(rotulos):
    classe_para_posicoes[r].append(i)

  # Garante que pelo menos 1 rótulo por classe será mantido
  posicoes_protegidas = []
  for classe in classes:
    posicoes_protegidas.append(random.choice(classe_para_posicoes[classe]))

  # Define as posições para remoção
  todas_posicoes = set(range(len(rotulos)))
  posicoes_removiveis = list(todas_posicoes - set(posicoes_protegidas))

  if quantidade_retirar > 0:
    posicoes_removidas = np.random.choice(posicoes_removiveis, quantidade_retirar, replace=False)
    rotulos_semi = np.array(rotulos)
    rotulos_semi[posicoes_removidas] = 0

  return rotulos_semi

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
  f_measure = f1_score(y_true, y_pred, average='weighted')

  return acuracia, f_measure, nRotulos

  

