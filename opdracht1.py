from random import random
import numpy as np
from scipy import rand
import scipy.linalg


def rot2d(a, b):
    factor = 1/(np.sqrt(a**2+b**2))
    matrix = np.array(((a, b), (-b, a)))
    return np.multiply(matrix, factor)

# COMMENT


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
    return rot_rechts(len(A[0]), c, d, (i))


def bidiagonaliseer(A):
    matrix = A
    for i in range(2, len(A)+1)[::-1]:
        matrix = nul_links(matrix, i, 1)@matrix
    matrix = matrix@nul_rechts(matrix, 1, 3)
    for i in range(3, len(A)+1)[::-1]:
        matrix = nul_links(matrix, i, 2)@matrix

    matrix = nul_links(matrix, 4, 3)@matrix

    return matrix


test = np.array(((1, 2), (2, 3)))

tessst = nul_rechts(np.array(((1, 2, 3, 4, 4), (4, 3, 2, 1, 2),
                              (3, 2, 1, 3, 5), (3, 4, 7, 4, 3), (3, 4, 5, 3, 2))), 1, 2)

mattie = np.array(((1, 2, 3, 4, 4), (4, 3, 2, 1, 2),
                   (3, 2, 1, 3, 5), (3, 4, 7, 4, 3), (3, 4, 5, 3, 2)))

randommatrix = np.random.rand(4, 3)
nul_matrix = nul_rechts(randommatrix, 1, 3)
# print(nul_matrix)

print(bidiagonaliseer(randommatrix))


# print(mattie@tessst)

# Voor nul links werken alleen waardes in de onderdriehoek, voor nul rechts
# alleen bovendriehoeks. En voor beide werkt de diagonaal niet.
