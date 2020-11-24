import numpy as np
from tensorflow.keras.datasets import mnist
from sklearn.decomposition import PCA

(x_train, _), (x_test, _) = mnist.load_data()

print(x_train.shape, x_test.shape)  # (60000, 28, 28), (10000, 28, 28)

x = np.append(x_train, x_test, axis = 0)
x = x.reshape(x.shape[0], x.shape[1] * x.shape[2])

pca1 = PCA(n_components=333)
x2d = pca1.fit_transform(x)
print(x2d.shape)

pca_EVR = pca1.explained_variance_ratio_
print(pca_EVR)
print(sum(pca_EVR))

# pca2 = PCA()
# pca2.fit(x)
# cumsum = np.cumsum(pca2.explained_variance_ratio_)  
# for i in range(len(cumsum)):
#     print(i+1, "번째 cumsum : ", cumsum[i])

# d = np.argmax(cumsum >= 0.95) + 1
# print(cumsum >= 0.95)
# print(d)

# import matplotlib.pyplot as plt
# plt.plot(cumsum)
# plt.grid()              
# plt.show()
