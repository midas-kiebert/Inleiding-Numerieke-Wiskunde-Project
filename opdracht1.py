import numpy as np

def rot2d(a,b):
    factor = 1/(np.sqrt(a**2+b**2))
    matrix = np.array(((a,b),(-b,a)))
    return np.multiply(matrix, factor)

def approx_svd(A, j):
    if j == 0:
        return A
    prevA = approx_svd(A, j-1)
    if j%2:
        return rot2d(prevA[0][0], prevA[1][0])@prevA
    return prevA@rot2d(prevA[0][0], prevA[1][0])

test = np.array(((1, 2),(2, 3)))

print(approx_svd(test, 10))
