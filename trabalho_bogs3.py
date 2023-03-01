# ===============================================================================
# ALUNOS
# Antonio Neves de Brito Junior 2236605
# Christoffer Nogueira Spring 2241471
# Bloom
# -------------------------------------------------------------------------------
# Autor: Bogdan T. Nassu
# Universidade Tecnológica Federal do Paraná
# ===============================================================================

import sys
import numpy as np
import cv2

# ===============================================================================

INPUT_IMAGE = 'GT2.BMP'
ALPHA = 0.8
BETA = 0.2

# ===============================================================================


def brightpass(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    thres = 127 / 255
    T, img_th = cv2.threshold(gray, thres, 255, cv2.THRESH_TOZERO)
    return img_th


def mask(img):
    ksize = (15, 15)
    mask = np.zeros((img.shape[0], img.shape[1]), np.float32)
    temp = img.copy()
    for x in range(0, 4):  # para cada gaussiano
        for i in range(0, 10):  # alguns da media
            temp = cv2.blur(img, ksize)
        mask += temp

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    return mask


def gauss(img):
    sigma = 5
    ksize = (0, 0)
    gss = np.zeros((img.shape[0], img.shape[1]), np.float32)
    temp = img.copy()
    for x in range(0, 4):
        temp = cv2.GaussianBlur(img, ksize, (sigma*(2**x)), temp)
        gss += temp

    gss = cv2.cvtColor(gss, cv2.COLOR_GRAY2BGR)
    return gss


def main():

    # Abre a imagem.
    img = cv2.imread(INPUT_IMAGE, cv2.IMREAD_COLOR)
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img is None:
        print('Erro abrindo a imagem.\n')
        sys.exit()

    # É uma boa prática manter o shape com 3 valores, independente da imagem ser
    # colorida ou não. Também já convertemos para float32.
    img = img.reshape((img.shape[0], img.shape[1], img.shape[2]))
    img = img.astype(np.float32) / 255
    cv2.imshow('entrada', img)
    cv2.imwrite('entrada.png', img*255)

    img_bp = brightpass(img)
    img_mk = mask(img_bp)
    img_gs = gauss(img_bp)

    # tentar achar as consts para ficar bonito
    img_out = (ALPHA * img) + (BETA * img_mk)
    img_outgs = (ALPHA * img) + (BETA * img_gs)

    cv2.imshow('03out', img_out)
    cv2.imshow('03outgs', img_outgs)

    cv2.imwrite('03out.png', img_out*255)
    cv2.imwrite('03out.png', img_outgs*255)

    cv2.waitKey()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()

# ===============================================================================
