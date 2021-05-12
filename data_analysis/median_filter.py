import numpy as np
from scipy import signal


def median_filter(X, k):
    X_1 = np.zeros_like(X)
    n = len(X)
    s = int((k - 1) / 2) # Parte intera
    Z = np.hstack((np.zeros(s), X, np.zeros(s)))
    i = 0 # Cambiato inizio

    while i <= (n-1): # Cambiato indice
        X_1[i] = median(Z[i:i + k]) # Rimuovere + 1
        i = i + 1
    return X_1

def median(X):
    n = len(X)
    X = np.sort(X)
    return X[n // 2] # controllare indice


A = np.array((25, 1, 2, 4, 25, 2, 3))
print(median_filter(A, 3))
print(signal.medfilt(A, kernel_size=3))

A = np.array((25,-1,2,2,2,2,-1,25))
print(median_filter(A, 3))
print(signal.medfilt(A, kernel_size=3))

A = np.array((25, 1))
print(median_filter(A, 3))
print(signal.medfilt(A, kernel_size=3))

A = np.array((25, 1, 2, 4, 25, 2, 3))
print(median_filter(A, 5))
print(signal.medfilt(A, kernel_size=5))

A = np.array((25, 1, 2, 4, 25, 2, 3))
print(median_filter(A, 7))
print(signal.medfilt(A, kernel_size=9))

