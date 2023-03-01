import numpy as np
import imutils
import matplotlib.pyplot as plt
import cv2


################################################################################

# img a ser aberta
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5testDigital.jpg' # 
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5woutNumbers.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5reverse.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5vert.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5rot3degree.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5rotminus8degree.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5rot86degree.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5rot99degree.jpg' # CERTO V
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5other.jpg' # CERTO V  
#IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real1.jpg' # CERTO V
IMAGES = 'image.jpg' # perspectiva traduz errado, pp fica ruim
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real3.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real4.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real5.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real6.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real7.jpg' # 2 codigos juntos nao funciona, junta os dois
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real8.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real9.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real10.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real11.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real12.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real13.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real14.jpg' # CERTO V 
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real15.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real16.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real17.jpg' # CERTO V
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real18.jpg' # perspectiva faz ele pegar errado o codigo
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real19.jpg' # CERTO V 
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real20.jpg' # da errado -> traduz o chao (+ variacao q o barcode)
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real21.jpg' # da errado -> traduz o pano (+ variacao q o barcode)
##IMAGES = 'drive/MyDrive/Colab Notebooks/imgBarCodeInt2of5real22.jpg' # CERTO V 

################################################################################

def getPP(image, axis, thresh):
    # projection profile -> conta o numero de pixels escuros por row ou col
    # thresh vem de otsu
    # axis == 0 para vertical
    # axis == 1 para horizontal
    img = image.copy()
    img = cv2.erode(img, None, iterations = 4)

    img[image < thresh] = 1 # abaixo do limiar vira 1 (escuro vira 1)
    img[image >= thresh] = 0 # acima do limiar vira 0 (claro vira 0)

    proj = np.sum(img, axis=axis)
    return proj

def getBarCodeBox(img, show):
  # calculando os gradientes
  gradX = cv2.Sobel(img, ddepth = cv2.CV_32F, dx = 1, dy = 0, ksize = -1)
  gradY = cv2.Sobel(img, ddepth = cv2.CV_32F, dx = 0, dy = 1, ksize = -1)

  # subtraindo um gradiente do outro
  grad = cv2.subtract(gradX, gradY)
  grad = cv2.convertScaleAbs(grad)

  # borra a img (o classico)
  blurr = cv2.blur(grad, (3, 3))

  # limiarizacao (o classico)
  _, thresh = cv2.threshold(blurr, 225, 255, cv2.THRESH_BINARY) 

  # fechando os buracos
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (19, 9))
  morph = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
  kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 19))
  morph = cv2.morphologyEx(morph, cv2.MORPH_CLOSE, kernel)
  erod = cv2.erode(morph, None, iterations = 4)
  dilat = cv2.dilate(erod, None, iterations = 4)

  # para mostrar o processo
  if show:
    print("gradient")
    cv2.imshow("gradiente", grad)
    print('blurr')
    cv2.imshow("borrado", blurr)
    print("threshed")
    cv2.imshow("tresh", thresh)
    print("morph")
    cv2.imshow("morfologias", morph)
    print("erode / dilate")
    cv2.imshow("erosão e dilatação", dilat)

  # acha os contornos e ordena pela area, ficando com o de maior area
  (cnt, _) = cv2.findContours(dilat, cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  cnt = sorted(cnt, key = cv2.contourArea, reverse = True)
  rows, cols = img.shape
  
  # pega o blob com maior contorno, dese q caiba na img
  for i in range(len(cnt)):
    boxInImg = 0
    bar = cnt[i]
    minRec = cv2.minAreaRect(bar)
    box = np.int0(cv2.boxPoints(minRec))
    for coords in box:
      if coords[0] < 0 or coords[1] < 0:
        boxInImg = 0
        break
      if coords[0] > cols or coords[1] > rows:
        boxInImg = 0
        break
      else: 
        boxInImg = 1
    if boxInImg == 1:
      return box

def showPP(vpp, hpp): 
  # mostra a projecao vert e horiz num grafico
  plt.plot(hpp)
  plt.plot(vpp)

  plt.xlabel('line / column')
  plt.ylabel('amount of black pixels')
  plt.title('projection profile')
  plt.show()
  return

def rotateImg(image, angle):
    # pega o centro
    (height, width) = image.shape[:2]
    (centerX, centerY) = (width / 2, height / 2)

    # pega a matriz de rotacao, o sin e o cos 
    M = cv2.getRotationMatrix2D((centerX, centerY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # novas dimensoes da img
    newWidth = int((height * sin) + (width * cos))
    newHeight = int((height * cos) + (width * sin))

    # ajusta a matriz de rotacao considerando a translacao para nao cortar a img
    M[0, 2] += (newWidth / 2) - centerX
    M[1, 2] += (newHeight / 2) - centerY

    # faz o giro e pode o fundo branco (se fosse preto iria contar na projecao)
    rotated = cv2.warpAffine(image, M, (newWidth, newHeight), borderValue=(255,255,255))

    return rotated

def getScore(vpp, hpp):
  # pegando os valores do vpp
  norm_vpp = vpp / np.linalg.norm(vpp.astype(int)) # normalizando para dar para comparar
  deriv_vpp = np.diff(norm_vpp) # deriv de vpp

  sumVpp = ((deriv_vpp.sum())).astype(float)  # deve ser pequena (alto delta, valores se cancelam (vale cancela topo))
  absVpp = ((sum(abs(value) for value in deriv_vpp))).astype(float)  # deve ser grande (muitas variacoes)
  sumVpp = round(sumVpp, 5) 
  absVpp = round(absVpp, 5)

  # pegando os valores do hpp   
  norm_hpp = hpp / np.linalg.norm(hpp.astype(int)) # normalizando para dar para comparar
  deriv_hpp = np.diff(norm_hpp) # devir de hpp

  sumHpp = ((deriv_hpp.sum())).astype(float) # deve ser pequena (baixo delta, nao deve ter grandes variacoes)
  absHpp = ((sum(abs(value) for value in deriv_hpp))).astype(float)  # deve ser pequena (poucas variacoes)
  sumHpp = round(sumHpp, 5)
  absHpp = round(absHpp, 5)

  score = [abs(sumVpp), abs(absVpp), abs(sumHpp), abs(absHpp)]
  return score

def getScoreNew(vpp, hpp):
  score = 0
  maxVpp = max(vpp)
  minVpp = min(vpp)

  # pegando os valores do vpp
  norm_vpp = vpp / np.linalg.norm(vpp.astype(int))
  deriv_vpp = np.diff(norm_vpp)

  noVariationV = (deriv_vpp == 0).sum()
  highVariationV = (deriv_vpp == max(deriv_vpp)).sum()
  highVariationV += (deriv_vpp == min(deriv_vpp)).sum()

  # pegando os valores do hpp   
  norm_hpp = hpp / np.linalg.norm(hpp.astype(int))
  deriv_hpp = np.diff(norm_hpp)

  noVariationH = (deriv_hpp == 0).sum()
  
  score += noVariationH + highVariationV + noVariationV
  return score

def findBestDegree(img, vertical, thresh, show):
  # buscando o melhor angulo para "endireitar" o barcode
  bestDegree = 0 # degree with the highest score
  bestScore = 0 # para o jeito antigo
  best_hpp = []
  best_vpp = []

  # barcode na horiz
  minD = -20 # inicio do teste da rotacao
  maxD = 21 # fim do teste da rotacao
  if vertical == 1: # se for na vert, testa com +90 graus pra ficar na horiz
    minD += 90
    maxD += 90

  for degrees in range(minD, maxD): # angulos a serem testados
    rotated = rotateImg(img.copy(), degrees) # gira sem dar crop, com bkg branco

    vpp = getPP(rotated[:,:], 0, thresh) # pp vertical
    hpp = getPP(rotated[:,:], 1, thresh) # pp horizontal
    score = getScore(vpp, hpp) # calcula o score dessa rotacao
                     
    if degrees == minD: # se for o primeiro
      bestDegree = degrees
      bestScore = score
      best_hpp = hpp
      best_vpp = vpp
    
    else:
      better = 0 # flag para saber se eh melhor ou n
      if score[1] > bestScore[1]: # vendo a soma abs da deriv do vpp
        better = 1
      elif score[1] == bestScore[1]:
        if score[0] < bestScore[0]: # vendo a soma da deriv do vpp # INVERTI O SINAL
          better = 1
        elif score[0] == bestScore[0]:
          if score[2] < bestScore[2]: # vendo a soma da deriv do hpp
            better = 1
          elif score[2] == bestScore[2]:
            if score[3] < bestScore[3]: # vendo a soma abs da deriv do hpp
              better = 1 

      if better == 1: # se for melhor, troca
        bestDegree = degrees
        bestScore = score
        best_hpp = hpp
        best_vpp = vpp
      
    # para ver os barcode rodados e os pp deles:
    if show:
      print('dg: ', degrees)
      cv2.imshow("rotacionado", rotated)
      print('score: ', score)
      showPP(vpp, hpp)

  return bestDegree, best_vpp, best_hpp

def getBoxCrop(box):
  # pega os pontos da box para cortar na img
  for i in range(4):
    if i == 0:
      min0 = box[i][0]
      min1 = box[i][1]
      max0 = box[i][0]
      max1 = box[i][1]
    else:
      if box[i][0] < min0: min0 = box[i][0]
      if box[i][0] > max0: max0 = box[i][0]
      if box[i][1] < min1: min1 = box[i][1]
      if box[i][1] > max1: max1 = box[i][1]

  return min0, max0, min1, max1

def getBarValues(vpp, show):
  # pega o nmr de pixels de cada barra (preta ou branca)
  values = []
  split = vpp.mean()
  
  # para mostrar o criterio de divisao (media)
  if show:
    print("splitter: ", split) 

  for value in vpp:
    if value > split: values.append(1) # acima da media 1, vai ser preto
    else: values.append(0) # abaixo da media 0, vai ser branco
  return values

def getSeqValues(barValues):
  # vai contar o tamanho das seq de 0 ou 1 
  # seq de 0 -> barra branca, marca como negativo
  # seq de 1 -> barra preta, marca como positivo
  barCount = []
  valueBef = 0 # valor anterior ao q vai ser analisado
  aux = 0 # contador da sequencia
  for values in barValues:
    if values == 1: # se o valor eh 1 (preto)
      if valueBef == 0: # se o anterior era branco
        barCount.append(aux) # coloca na count o contador do branco
        aux = 1 # contador vira 1
      else: # se o anterior tbm era preto
        aux += 1 # incrementa contador

    else: # se o valor eh 0 (branco)
      if valueBef == 1: # se o anterior era preto
        barCount.append(aux) # coloca na count o contador do preto
        aux = -1 # contador vira -1
      else: # se o anterior tbm era branco 
        aux -= 1 # decrementa contador

    valueBef = values # anterior se torna o atual

  barCount.append(aux) # colocando o valor da ultima barra
  return barCount

def getNarrowWide(barSeqValues):
  # vai ver se a barra (preta ou branca) eh barrinha (narrow) ou barrona (wide)
  # wide sera 2 (preto) ou -2 (branco)
  # narrow sera 1 (preto) ou -1 (branco)
  barNW = [] #bar narrow and wide

  maxBlk = 0
  minBlk = 0
  maxWht = 0
  minWht = 0

  # separa o max e min preto (positivo) e o max e min branco (negativos)
  for value in barSeqValues:
    if value > 0: # preto
      if maxBlk == 0: maxBlk = value # se nao tiver valor, seta como max
      elif value > maxBlk: maxBlk = value # se for maior q o max, vira max
      if minBlk == 0: minBlk = value # se nao tiver valor, seta como min
      elif value < minBlk: minBlk = value # se for menor q o min, vira min
    if value < 0: # branco
      absvalue = abs(value)
      if maxWht == 0: maxWht = absvalue # se nao tiver valor, seta como max
      elif absvalue > maxWht: maxWht = absvalue # se for maior q o max, vira max
      if minWht == 0: minWht = absvalue # se nao tiver valor, seta como min
      elif absvalue < minWht: minWht = absvalue # se for menor q o min, vira min

  # varre os valores da seq:
  for values in barSeqValues:
    if values != 0:
      if values > 0: # black bar -> diferenca abs para saber se eh Wide ou Narrow
        difMax = abs(values - maxBlk)
        difMin = abs(values - minBlk)
        if difMax < difMin: # diferenca menor ate o topo (+ perto do max) -> Wide
          barNW.append(2)
        else: barNW.append(1) # diferenca menor ate a base (+ perto do min) -> Narrow

      else: # white bar -> diferenca abs para saber se eh Wide ou Narrow
        difMax = abs(-values - maxWht)
        difMin = abs(-values - minWht)
        if difMax < difMin: # diferenca menor ate o topo (+ perto do max) -> Wide
          barNW.append(-2)
        else: barNW.append(-1) # diferenca menor ate a base (+ perto do min) -> Narrow

  return barNW

def getWeight(position):
  # retorna o peso do valor da barra wide, de acordo com as regras do 
  # interleaved 2 of 5
  if position == 0: return 1
  if position == 1: return 2
  if position == 2: return 4
  if position == 3: return 7
  if position == 4: return 0

def getTranslation(barNW):
  # 5 barras pretas (?) == 1 numero
  # 1	2	4	7	0 --> peso delas, soma == numero, se der 11, vale 0
  # start = nnnn (barra, espaco, barra, espaco)
  translation = []
  blck5 = []
  whte5 = []

  for i in range(4, len(barNW)): 
    if barNW[i] > 0: blck5.append(barNW[i])
    else: whte5.append(barNW[i])

    if len(blck5) == 5:
      # podemos traduzir esse bloco
      aux = 0
      for j in range(5):
        if blck5[j] == 2: 
          aux += getWeight(j)
      if aux == 11: aux = 0
      translation.append(aux)
      blck5 = []
    if len(whte5) == 5:
      # podemos traduzir esse bloco
      aux = 0
      for j in range(5):
        if whte5[j] == -2: 
          aux += getWeight(j)
      if aux == 11: aux = 0
      translation.append(aux)
      whte5 = []

  return translation

def translateBar(vpp, show):
  # traduz a barra partindo do vpp
  barValues = getBarValues(vpp, show) # pegando os valores se sao pretos (1) ou brancos (0)
  barSeqValues = getSeqValues(barValues) # pega os valores sequenciais

  # corta o comeco e o final se forem brancos -> barra comeca e termina com preto
  if barSeqValues[0] <= 0: del barSeqValues[0]
  if barSeqValues[len(barSeqValues)-1] <= 0: del barSeqValues[len(barSeqValues)-1]

  barNW = getNarrowWide(barSeqValues) # separa os valores em narrow (1 ou -1) e wide (2 ou -2)

  # traduzindo seguindo a regra do tipo interleaved 2 of 5:
  start = [1, -1, 1, -1] # condicao de inicio de leitura do barcode
  if barNW[:4] == start: # se comecar com o start a barra nao esta invertida
    barTranslation = getTranslation(barNW)
  else: # se nao comecar com o start a barra esta invertida
    reversed_barNW = barNW[::-1]
    barTranslation = getTranslation(reversed_barNW)

  # para ver os valores de cada parte da traducao
  if show:
    print(barValues)
    print(barSeqValues)
    print(barNW)
    print(barTranslation)

  return barTranslation

def showTranslation(translation):
  # printa a traducao da barra
  for values in translation:
    print(values, end=' ')
  print('\n')
  return

def main():
  showMiddle = False # para ajudar no debug
  showEnd = False # para ver mais do q apenas a traducao

  # abrindo a img
  image = cv2.imread(IMAGES)
  imageOriginal = cv2.imread(IMAGES)
  img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  box = getBarCodeBox(img, showMiddle) # acha o codigo na img, com as coord do contorno dele
  boxCoord = getBoxCrop(box) # pega os extremos da coord da box
  
  # ve se o box esta deitado ou em pe
  vertical = 0 # 0 deitado, 1 em pe
  barWidth = abs(boxCoord[0] - boxCoord[1])
  barHeight = abs(boxCoord[2] - boxCoord[3])
  if(barHeight > barWidth): vertical = 1 # altura maior q largura -> barcode em pe 

  barImg = image[boxCoord[2]:boxCoord[3], boxCoord[0]:boxCoord[1]] # recorta o barcode da img
  barImg = cv2.cvtColor(barImg, cv2.COLOR_BGR2GRAY) # escala de cinza

  otsu, _ = cv2.threshold(barImg.copy(), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU) # vamos usar o thresh vindo do otsu
  bestdegree, bvpp, bhpp = findBestDegree(barImg, vertical, otsu, showMiddle) # acha o melhor grau de rotacao para traduzir o barcode

  # para ver mais detalhes do resultado
  if showEnd:
    print('melhor angulo de rotacao do barcode: ', bestdegree)
    rotatedOriginal = rotateImg(barImg, bestdegree)
    print('codigo na img original')
    cv2.imshow ("original", barImg)
    print('melhor rotacao - img toda')
    cv2.imshow("melhor rotação", rotateImg(image, bestdegree))
    print('melhor rotacao - barcode')
    cv2.imshow("rotação", rotatedOriginal)
    print('grafico do barcode com melhor rotacao')
    showPP(bvpp, bhpp)

  print('img original')
  cv2.imshow ("imagem original", imageOriginal)

  print('Traducao: ')
  translation = translateBar(bvpp, showMiddle) # traduz a barra  
  showTranslation(translation) # mostra a traducao


  cv2.waitKey(0)

if __name__ == '__main__':
  main ()

from google.colab import drive
drive.mount('/content/drive')

# apresentacao de uns 12 minutos
# sem explicar codigo, so ele rodando e os metodos para resolver o problema
# pipeline de como foi solucionado 
# sem explicar muito coisas que ja foram explicadas em aula
