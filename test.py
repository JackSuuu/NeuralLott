import numpy as np

Y_true = np.array([[1] for _ in range(20)])
Y_false = np.array([[0] for _ in range(28)])
Y = np.concatenate((Y_true, Y_false), axis=0)

print(Y)