#!/usr/bin/env python
# coding: utf-8

import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def openImage(fname):
    """ str -> Array 
    (notation above means the function gets a string argument and returns an Array object)
    """
    return np.array(Image.open(fname))

def countPixelsLoop(I,k):
    """ Array*int -> int"""
    cpt_k = 0
    N, M = I.shape
    for i in range(N):
        for j in range(M):
            if I[i,j] == k:
                cpt_k += 1
    return cpt_k

def countPixels(I,k):
    return (I==k).sum()


def replacePixels(I,k1,k2):
    """ Array*int*int -> Array """
    
    return np.where(I==k1, k2, I)

def normalizeImageLoop(I,k1,k2):
    """ Array*int*int -> Array """
    newI = np.copy(I)
    kmax = I.max()
    kmin = I.min()
    i1 = kmax - kmin
    i2 = k2 - k1
    N, M = I.shape
    for i in range(N):
        for j in range(M):
            newI[i, j] = np.int64((I[i,j]-kmin)/i1 *i2 + k1)
    return newI

def inverteImage(I):
    """ Array -> Array """
    invert = lambda x: 255-x
    return invert(I)

def computeHistogram(I):
    """ Array -> list[int] """
    kmax=I.max()
    kmin=I.min()
    H = np.zeros(256)
    return [np.count_nonzero(I == k) for k in range(256)]


def thresholdImage(I,s):
    """ Array*int -> Array """
    return np.where(I<s, 0, 255)


def histogramEqualization(I,h):
    """ Array * (list[int] -> Array """
    h = np.array(h)
    equalized = np.copy(I)
    kmax=I.max()
    kmin=I.min()
    L = kmax - kmin
    coef = L/I.size
    N, M = I.shape
    for k in range(256):
        equalized = replacePixels(equalized, k, coef*h[:k+1].sum())
    return equalized
  
