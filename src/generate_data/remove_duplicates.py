import numpy as np


data = np.load('dataset.npz')
u, counts = np.unique(data['arr_0'], axis=0, return_counts=True)
print(u.shape, counts[counts>1])