


import numpy as np
from numpy.fft import fft2,fftshift
from PIL import Image
import TME1
import matplotlib.pyplot as plt


def computeFT(I):
    """ Array -> Array[complex] """
    return np.fft.fftshift(np.fft.fft2(I))

def toVisualizeFT(If):
    """ Array[complex] -> Array[float] """
    return abs(If)

def toVisualizeLogFT(If):
    """ Array[complex] -> Array[float] """
    return np.log(1 + abs(If))



def blend(I1,I2,alpha):
    """ Array**2*float -> Array """
    return alpha*I1+(1-alpha)*I2



def mainOrientation(I):
    """ Array -> tuple[Iori,float]
        return image of orientation (32 bins) and the main orientation (degree) from a Fourier transform module
    """
    n, m = I.shape

    size = 32
    x = np.array(range(size))
    ori = np.vstack((np.cos(np.pi*x/size), np.sin(np.pi*x/size))).T

    Iori = np.zeros((n, m))
    orients = np.zeros((size))

    for i in range(1,n+1):
        for j in range(1,m+1):
            if I[i-1, j-1] > 0:
                v = np.array([j-m/2, -i + n/2])
                if i > n/2:
                    v = -v
                    prod = np.matmul(ori, v)
                    maxi = prod.max()
                    if maxi > 0:
                        imax = np.nonzero(prod == maxi)
                        Iori[i-1, j-1] = imax[0]
                        orients[imax] += 1

    maxori = np.nonzero(orients == orients.max())[0][0]
    return (Iori, 180*maxori/size-90)

def rotateImage(I,a):
    """ Array*float -> Array 
        return a rotation of angle a (degree) of image I
    """
    return np.array(Image.fromarray(I).rotate(a, expand=True, fillcolor=127))

def rectifyOrientation(I):
    I2 = TME1.thresholdImage(toVisualizeFT(computeFT(I)), 3e5)
    I2, ori = mainOrientation(I2)
    rectified = rotateImage(I, -ori)
    return rectified
 

def thresholdFT(If, s):
    return np.where(np.abs(If)<s, If, 0)

