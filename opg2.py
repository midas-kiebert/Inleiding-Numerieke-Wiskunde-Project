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
        return rot2d(prevA[0][0], prevA[1][0]) @ prevA
    return prevA @ rot2d(prevA[0][0], prevA[1][0])


def sigma(a):
    return np.linalg.svd(a)


def rot(n, a, b, i):
    boven = np.eye(i)
    onder = np.eye(n-i-2)
    return scipy.linalg.block_diag(boven, rot2d(a, b), onder)


def nul_links(A, i, j):
    a = A[i-2][j-1]
    b = A[i-1][j-1]
    return rot(len(A), a, b, (i-2))


def rot_rechts(n, a, b, i):
    boven = np.eye(i)
    onder = np.eye(n-i-2)
    return scipy.linalg.block_diag(boven, rot2d(a, b).T, onder)


def nul_rechts(A, i, j):
    c = A[i-1][j-2]
    d = A[i-1][j-1]
    return rot_rechts(len(A[0]), c, d, (j-2))

A = np.random.rand(2,8)
print(A)
print(nul_rechts(A, 1, 3))
print(A@nul_rechts(A, 1, 3))