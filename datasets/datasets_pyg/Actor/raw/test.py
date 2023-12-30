import numpy as np
data = np.load('film_split_0.6_0.2_0.npz')
print(np.sum(data['train_mask']))  # 7600
print(np.sum(data['test_mask']))   # 7600
print(np.sum(data['val_mask']))