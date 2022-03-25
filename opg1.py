from random import random
import numpy as np
from scipy import rand
import scipy.linalg


def rot2d(a, b):
    factor = 1/(np.sqrt(a**2+b**2))
    matrix = np.array(((a, b), (-b, a)))
    return np.multiply(matrix, factor)


def approx_svd(A, j):
    if j == 0:
        return A
    prevA = approx_svd(A, j-1)
    if j % 2:
        return prevA @ rot2d(prevA[0][0], prevA[0][1]).T
    return rot2d(prevA[0][0], prevA[1][0]) @ prevA


def sigma(a):
    U, s, vH = np.linalg.svd(a)
    return s
