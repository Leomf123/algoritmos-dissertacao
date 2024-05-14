import numpy as np
import random

# função para retirar rotulos dado uma porcentagem
# entrada: vetor de rótulos rotulados e uma porcetagem pra manter
# saída: vertor de rotulos com alguns não rotulados
def retirar_rotulos(rotulos, porcentagem_manter):

 quantidade_retirar = int(len(rotulos)*(1 - porcentagem_manter))

 rotulos_semissupervisionado = np.array(rotulos)
 for i in range(quantidade_retirar):
  while(True):
    posicao_aleatoria = random.randint(0, len(rotulos)-1)
    if rotulos_semissupervisionado[posicao_aleatoria] != 0:
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