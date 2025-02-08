import numpy as np

def l2(X, y):
    return np.sum(np.abs(X - y)**2, axis=1)

def l1(X, y):
    return np.sum(np.abs(X - y), axis=1)

def cosine(X, y):
    return 1 - np.dot(X, y) / (np.linalg.norm(X) * np.linalg.norm(y))

distance_metrics = {
    'l1': l1,
    'l2': l2,
    'cosine': cosine
}
