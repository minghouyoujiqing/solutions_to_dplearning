import numpy as np
def normalizeRows(x):
    norm = np.linalg.norm(x,axis=1, keepdims= True)
    return x/norm

x = np.array([
    [0, 3, 4],
    [1, 6, 4]])
print("normalizeRows(x) = " + str(normalizeRows(x)))