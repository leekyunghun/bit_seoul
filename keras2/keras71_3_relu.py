import numpy as np
import matplotlib.pyplot as plt

def relu(x):
    return np.maximum(0, x)

def leakyRelu(x):
    return np.maximum(0.01 * x, x)

def elu(alpha, x):
    a = []
    for i in range(len(x)):
        if x[i] < 0:
            a.append(alpha * (np.exp(x[i]) -1))
        else:
            a.append(x[i])
    return a

x = np.arange(-5, 5, 1)
y = elu(1,x)
plt.plot(x, y)
plt.grid()
plt.show()

# relu 친구들
# elu, leakyRelu