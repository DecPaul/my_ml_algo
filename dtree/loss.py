import numpy as np

#loss function

def mse(y):
    return np.mean((y-np.mean(y)) ** 2)

def entropy(y):
    hist = np.bincount(y)
    ps = hist / np.sum(hist)
    return -np.sum([p * np.log2(p) for p in ps if p > 0])

def gini(y):
    hist = np.bincount(y)
    N = np.sum(hist)
    return 1 - sum([(i / N) ** 2 for i in hist])