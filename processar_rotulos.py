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
