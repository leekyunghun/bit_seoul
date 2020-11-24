import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_diabetes

datasets = load_diabetes()
x = datasets.data       # (442, 10)
y = datasets.target     # (442, )

# pca = PCA(n_components=8)
# x2d = pca.fit_transform(x)
# print(x2d.shape)

# pca_EVR = pca.explained_variance_ratio_
# print(pca_EVR)
# print(sum(pca_EVR))

pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)       # 1개부터 10개까지 축소했을때의 가치값을 보여준다.
print(cumsum)

d = np.argmax(cumsum >= 0.95) + 1
print(cumsum >= 0.95)
print(d)

import matplotlib.pyplot as plt
plt.plot(cumsum)
plt.grid()              
plt.show()