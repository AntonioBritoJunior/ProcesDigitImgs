
from statsmodels.stats.weightstats import DescrStatsW
from scipy import stats
import matplotlib.pyplot as plt
import sys
import statistics
import timeit
import numpy as np
import cv2
import math as mt

# ===============================================================================
# ALUNOS
# Antonio Neves de Brito Junior 2236605
# Christoffer Nogueira Spring 2241471
# Chroma Key
# -------------------------------------------------------------------------------
# Professor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# ===============================================================================


IMAGES = '0.BMP', \
          '1.bmp',\
          '2.bmp',\
          '3.bmp',\
          '4.bmp',\
          '5.bmp',\
          '6.bmp',\
          '7.bmp',\
          '8.bmp' 

BACKGROUND = 'Wind Waker GC.bmp'

def normalize(img):
  normalized = (img - np.min(img)) / (np.max(img) - np.min(img))
  return normalized

def whereIsGreen(hsv):
  lwr_green = np.array([40, 100, 50]) # 40, 110, 70
  upr_green = np.array([75, 255, 255]) #85, 255, 255

  # seleciona pixeis verdes
  mask = cv2.inRange(hsv, lwr_green, upr_green)
  return mask

def findGreeness(hsv, mask):
  wMask = np.zeros(mask.shape)
  height, width = mask.shape
  for y in range(height):
    for x in range(width):
      if mask[y, x] != 0:
        pixel = hsv[y, x, :]
        sum = (((pixel[0]-40) // 10) / 10) * 5 # matiz 
        sum += (((pixel[1]-100) // 10) / 10) * 10 # + alto = - branco / - alto = + branco
        sum += (((pixel[2]-50) // 10) / 10) * 10 # + alto = - preto / - alto = + preto    
        wMask[y, x] = sum / (5 + 10 + 10) 
  return wMask

def main ():
    sys.setrecursionlimit(1000000)
    # abrindo o background
    bkg = cv2.imread (BACKGROUND, cv2.IMREAD_COLOR)
    if bkg is None:
      print ('Erro abrindo o background.\n')
      sys.exit ()

    bkgHeight = bkg.shape[0]
    bkgWidth = bkg.shape[1]
    bkg = bkg.reshape (bkgHeight, bkgWidth, bkg.shape [2])

    # abrindo as img a imagem em escala de cinza
    for img_input in IMAGES:
      img = cv2.imread (img_input, cv2.IMREAD_COLOR)
      if img is None:
        print ('Erro abrindo a imagem.\n')
        sys.exit ()

      img = img.reshape (img.shape [0], img.shape [1], img.shape[2])
      bkg = cv2.resize(bkg, (img.shape[1], img.shape[0]))
      height, width, channels = img.shape

      # passando para hsv
      hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

      # preparando as mascaras
      mask = whereIsGreen(hsv)
      wMask = findGreeness(hsv, mask)
      wMask = normalize(wMask)

      mean = np.mean(wMask) * 255
      q1 = np.percentile(wMask, 25) * 255
      q2 = np.percentile(wMask, 50) * 255
      q3 = np.percentile(wMask, 75) * 255
      max = np.max(wMask) 

      # wMask --> indica um fator de verdice
      bkgg = bkg.copy()
      bkgg *= 255
      pixel = np.zeros(img.shape)
      for y in range(height):
        for x in range(width):
          # wMask --> + branco, + alto ==> + verde
          if wMask[y, x] <= 0.1: # nao eh verde
            # pega da img
            pixel[y, x, :] = img[y, x, :]

            # se for verde demais, pega o canal g como a media do b e r
            if 2*pixel[y, x, 1] >= (pixel[y, x, 0] + pixel[y, x, 2]):
              pixel[y, x, 1] = (pixel[y, x, 0] + pixel[y, x, 2]) / 2
        
          elif wMask[y, x] >= 0.9: # eh verde
            # pega do fundo 
            pixel[y, x, :] = bkg[y, x, :] 

          else: # esverdeado
            # pega o fundo escurecido
            k = wMask[y, x] * 255
            aux = max
            aux2 = aux

            thresholdLower = mean - 70
            thresholdMiddle = mean
            thresholdUpper = q3
            
            # pegando o fundo multiplicado, sendo um "escurecedor" do fundo
            if thresholdUpper <= k:  
              pixel[y, x, :] = (bkg[y, x, :]) * (max)
              if wMask[y, x] < aux: aux = wMask[y, x]
            elif thresholdMiddle <= k < thresholdUpper:
              pixel[y, x, :] = (bkg[y, x, :]) * (aux)
              if wMask[y, x] < aux2: aux2 = wMask[y, x]
            elif thresholdLower <= k < thresholdMiddle: 
              pixel[y, x, :] = (bkg[y, x, :]) * (aux2)
              if wMask[y, x] < aux2: aux2 = wMask[y, x]
            else: 
              pixel[y, x, :] = (bkg[y, x, :]) * wMask[y, x]
                 
      print('img:')
      cv2.imshow('img:',img)
      print('bkg:')
      cv2.imshow('bkg:',bkg)
      print('result:')
      cv2.imshow('result:',pixel/255)
      cv2.imwrite('resultado'+img_input+'.png',pixel)

      bkg = cv2.resize(bkg, (bkgWidth, bkgHeight)) 

    cv2.waitKey ()
    cv2.destroyAllWindows ()


if __name__ == '__main__':
  main ()