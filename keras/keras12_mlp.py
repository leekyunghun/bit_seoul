#1. 데이터
import numpy as np

x = np.array([range(1, 101), range(711,811), range(100)])
y = np.array([range(101, 201), range(311,411), range(100)])

x_T = x.T
y_T = y.T

print(x.shape)
print(y.shape)

print(x)

print(x_T.shape)
print(y_T.shape)

print(x_T)
print(y_T)